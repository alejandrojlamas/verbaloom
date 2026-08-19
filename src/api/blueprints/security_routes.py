"""
Security and file upload routes
"""
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app

from src.utils.security import SecureFileHandler, rate_limiter, get_client_ip, SecurityError
from src.utils.language_detector import LanguageDetector
from src.api.services.path_validator import PathValidator


def _detect_uploaded_file_language(file_data, original_filename, secure_path):
    """Detect language using the saved filename first, then the browser filename.

    Android/share-sheet uploads sometimes arrive without an extension in
    ``original_filename`` even when the saved secure filename has the content
    extension restored. LanguageDetector chooses PDF/EPUB extractors from the
    filename, so prefer the saved path name.
    """
    candidates = []
    if secure_path:
        candidates.append(Path(secure_path).name)
    if original_filename:
        candidates.append(original_filename)

    seen = set()
    for filename in candidates:
        if not filename or filename in seen:
            continue
        seen.add(filename)
        detected_language, confidence = LanguageDetector.detect_language_from_file(
            file_data,
            filename,
        )
        if detected_language:
            return detected_language, confidence
    return None, 0.0


def _probe_uploaded_readable_text(secure_path, file_type):
    """Return compact readable-text metadata for an uploaded source file."""
    label = str(file_type or "file").upper()
    try:
        from src.core.output_formats import extract_readable_text

        readable_text = extract_readable_text(Path(secure_path))
    except Exception as exc:
        return {
            "readable": False,
            "readable_characters": 0,
            "message": f"Could not extract readable text from this {label} file: {exc}",
        }

    char_count = len((readable_text or "").strip())
    if char_count <= 0:
        return {
            "readable": False,
            "readable_characters": 0,
            "message": (
                f"This {label} file has no readable text. If it is a scanned PDF "
                "or image-only document, run OCR first and upload the OCR text/PDF."
            ),
        }
    return {
        "readable": True,
        "readable_characters": char_count,
        "message": "",
    }


def _delete_failed_upload(secure_path):
    try:
        path = Path(secure_path)
        if path.exists() and path.is_file():
            path.unlink()
    except Exception as exc:
        current_app.logger.warning("Failed to remove unreadable upload %s: %s", secure_path, exc)


def _infer_uploaded_book_profile(original_filename, secure_path):
    """Return compact profile metadata inferred from upload names/paths."""
    try:
        from src.core.book_profiles import infer_profile_id_from_metadata, load_book_profile

        profile_id = infer_profile_id_from_metadata({
            "original_filename": original_filename,
            "input_filename": original_filename,
            "file_path": str(secure_path or ""),
            "secure_filename": Path(secure_path).name if secure_path else "",
        })
        if not profile_id:
            return {}
        profile = load_book_profile(profile_id, allow_missing=True)
        if profile is None:
            return {"profile_id": profile_id}
        return {
            "profile_id": profile.profile_id,
            "profile_name": profile.name,
            "target_locale": profile.target_locale,
            "approved_count": profile.approved_count,
            "pending_count": profile.pending_count,
            "generated_profile": bool(profile.raw_config.get("generated_profile", False)),
        }
    except Exception as exc:
        current_app.logger.warning("Could not infer upload book profile for %s: %s", original_filename, exc)
        return {}


def create_security_blueprint(output_dir):
    """
    Create and configure the security blueprint

    Args:
        output_dir: Base directory for file operations
    """
    bp = Blueprint('security', __name__)

    # Initialize secure file handler
    upload_dir = Path(output_dir) / 'uploads'
    secure_file_handler = SecureFileHandler(upload_dir)

    @bp.route('/api/upload', methods=['POST'])
    def upload_file():
        """Secure file upload with comprehensive validation"""

        # Rate limiting
        client_ip = get_client_ip(request)
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({
                "error": "Rate limit exceeded. Please wait before uploading again.",
                "remaining_requests": rate_limiter.get_remaining_requests(client_ip)
            }), 429

        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Security check: limit filename length in request
        if len(file.filename) > 255:
            return jsonify({"error": "Filename too long"}), 400

        try:
            # Read at most one byte beyond the documented limit. Flask's
            # MAX_CONTENT_LENGTH rejects oversized request bodies earlier;
            # this bound also protects direct blueprint tests and clients that
            # omit Content-Length.
            file_data = file.read(SecureFileHandler.MAX_FILE_SIZE + 1)

            if len(file_data) > SecureFileHandler.MAX_FILE_SIZE:
                return jsonify({"error": "File too large"}), 413

            # Quick size check before validation
            if len(file_data) == 0:
                return jsonify({"error": "Empty file not allowed"}), 400

            # Validate and save file securely
            validation_result = secure_file_handler.validate_and_save_file(
                file_data, file.filename
            )

            if not validation_result.is_valid:
                return jsonify({
                    "error": validation_result.error_message,
                    "details": "File validation failed"
                }), 400

            # Get file info
            file_size = len(file_data)
            secure_path = validation_result.file_path

            # Determine file type from saved content/extension using the same
            # detector as the translation pipeline.
            from src.utils.file_detector import detect_file_type
            file_type = detect_file_type(str(secure_path))

            readable_probe = _probe_uploaded_readable_text(secure_path, file_type)
            if not readable_probe["readable"]:
                _delete_failed_upload(secure_path)
                current_app.logger.warning(
                    "Unreadable upload rejected: %s -> %s, type=%s, IP=%s",
                    file.filename,
                    secure_path.name,
                    file_type,
                    client_ip,
                )
                return jsonify({
                    "error": "Uploaded file has no readable text",
                    "message": readable_probe["message"],
                    "details": {
                        "file_type": file_type,
                        "readable_characters": readable_probe["readable_characters"],
                    },
                }), 400

            # Detect source language from file content
            detected_language, confidence = _detect_uploaded_file_language(
                file_data,
                file.filename,
                secure_path,
            )

            # Extract cover for EPUB files
            thumbnail_filename = None
            if file_type == "epub":
                try:
                    from src.core.epub.cover_extractor import EPUBCoverExtractor

                    # Create thumbnails directory
                    thumbnails_dir = Path(output_dir) / 'thumbnails'
                    thumbnails_dir.mkdir(exist_ok=True)

                    # Extract and save thumbnail
                    thumbnail_filename = EPUBCoverExtractor.extract_cover(
                        str(secure_path),
                        thumbnails_dir
                    )

                    if thumbnail_filename:
                        current_app.logger.info(f"Extracted EPUB cover: {thumbnail_filename}")
                except Exception as e:
                    current_app.logger.warning(f"Failed to extract EPUB cover: {e}")
                    # Continue without thumbnail (graceful degradation)

            # Return success response
            response_data = {
                "success": True,
                # A managed filename is the client capability. Do not expose
                # the host's absolute checkout or home-directory layout.
                "file_path": secure_path.name,
                "filename": file.filename,
                "secure_filename": secure_path.name,
                "file_type": file_type,
                "readable": True,
                "readable_characters": readable_probe["readable_characters"],
                "size": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2)
            }

            inferred_profile = _infer_uploaded_book_profile(file.filename, secure_path)
            if inferred_profile:
                response_data["profile_id"] = inferred_profile.get("profile_id")
                response_data["profile_name"] = inferred_profile.get("profile_name")
                response_data["book_profile"] = inferred_profile

            # Add thumbnail if available
            if thumbnail_filename:
                response_data["thumbnail"] = thumbnail_filename

            # Add detected language if available
            if detected_language:
                response_data["detected_language"] = detected_language
                response_data["language_confidence"] = round(confidence, 2)
                current_app.logger.info(
                    f"Language detected: {detected_language} "
                    f"(confidence: {confidence:.2f}) for {file.filename}"
                )

            # Add warnings if any
            if validation_result.warnings:
                response_data["warnings"] = validation_result.warnings

            # Log successful upload
            current_app.logger.info(f"Secure file upload successful: {file.filename} -> {secure_path.name}, Size: {file_size} bytes, IP: {client_ip}")

            return jsonify(response_data), 200

        except SecurityError as e:
            current_app.logger.warning(f"Security violation in file upload: {str(e)}, IP: {client_ip}, Filename: {file.filename}")
            return jsonify({
                "error": "Security validation failed",
                "details": str(e)
            }), 403

        except Exception as e:
            current_app.logger.error(f"File upload error: {str(e)}, IP: {client_ip}, Filename: {file.filename}")
            return jsonify({
                "error": "Upload failed due to server error",
                "details": "Please try again or contact support"
            }), 500

    @bp.route('/api/security/cleanup', methods=['POST'])
    def cleanup_old_files():
        """Clean up old uploaded files (admin endpoint)"""
        try:
            # Get max age from request (default 24 hours)
            max_age_hours = request.json.get('max_age_hours', 24) if request.json else 24

            # Validate input
            if not isinstance(max_age_hours, (int, float)) or max_age_hours < 1:
                return jsonify({"error": "Invalid max_age_hours parameter"}), 400

            # Perform cleanup
            secure_file_handler.cleanup_old_files(max_age_hours)

            return jsonify({
                "success": True,
                "message": f"Cleanup completed for files older than {max_age_hours} hours"
            })

        except Exception as e:
            current_app.logger.error(f"Cleanup error: {str(e)}")
            return jsonify({"error": "Cleanup failed"}), 500

    @bp.route('/api/security/info', methods=['GET'])
    def get_security_info():
        """Get security configuration and limits"""
        client_ip = get_client_ip(request)

        return jsonify({
            "file_limits": {
                "max_size_mb": SecureFileHandler.MAX_FILE_SIZE // (1024 * 1024),
                "allowed_extensions": list(SecureFileHandler.ALLOWED_EXTENSIONS),
                "allowed_mime_types": list(SecureFileHandler.ALLOWED_MIME_TYPES)
            },
            "rate_limit": {
                "remaining_requests": rate_limiter.get_remaining_requests(client_ip),
                "window_seconds": rate_limiter._window_seconds,
                "max_requests": rate_limiter._max_requests
            },
            "upload_directory": str(secure_file_handler.upload_dir)
        })

    @bp.route('/api/uploads/verify', methods=['POST'])
    def verify_uploaded_files():
        """Verify which uploaded files still exist on the server"""
        try:
            data = request.json
            if not data or 'file_paths' not in data:
                return jsonify({"error": "No file paths provided"}), 400

            file_paths = data['file_paths']
            if not isinstance(file_paths, list):
                return jsonify({"error": "Invalid file paths list"}), 400

            existing_files = []
            missing_files = []

            for file_path_str in file_paths:
                try:
                    file_path = PathValidator.resolve_managed_file(
                        file_path_str,
                        [secure_file_handler.upload_dir],
                    )
                except (ValueError, FileNotFoundError):
                    file_path = None
                if file_path is not None:
                    existing_files.append(file_path_str)
                else:
                    missing_files.append(file_path_str)

            return jsonify({
                "existing": existing_files,
                "missing": missing_files
            })

        except Exception as e:
            current_app.logger.error(f"Error verifying uploaded files: {str(e)}")
            return jsonify({"error": "Verification failed", "details": str(e)}), 500

    @bp.route('/api/detect-language', methods=['POST'])
    def detect_language():
        """Detect language from an already uploaded file"""
        try:
            data = request.json
            if not data or 'file_path' not in data:
                return jsonify({"error": "No file path provided"}), 400

            file_path_str = data['file_path']
            try:
                file_path = PathValidator.resolve_managed_file(
                    file_path_str,
                    [secure_file_handler.upload_dir],
                )
            except ValueError:
                return jsonify({"error": "Access denied"}), 403
            except FileNotFoundError:
                return jsonify({"error": "File not found"}), 404

            # Read file and detect language
            with open(file_path, 'rb') as f:
                file_data = f.read(SecureFileHandler.MAX_FILE_SIZE + 1)
            if len(file_data) > SecureFileHandler.MAX_FILE_SIZE:
                return jsonify({"error": "File too large"}), 413

            detected_language, confidence = LanguageDetector.detect_language_from_file(
                file_data, file_path.name
            )

            if detected_language:
                current_app.logger.info(
                    f"Language detected: {detected_language} "
                    f"(confidence: {confidence:.2f}) for {file_path.name}"
                )
                return jsonify({
                    "success": True,
                    "detected_language": detected_language,
                    "language_confidence": round(confidence, 2)
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Could not detect language"
                }), 200

        except Exception as e:
            current_app.logger.error(f"Language detection error: {str(e)}")
            return jsonify({"error": "Language detection failed"}), 500

    @bp.route('/api/thumbnails/<path:filename>', methods=['GET'])
    def serve_thumbnail(filename):
        """Serve EPUB cover thumbnail with security validation"""
        try:
            from werkzeug.utils import secure_filename
            from flask import send_file

            # Security: prevent path traversal
            safe_filename = secure_filename(filename)
            if safe_filename != filename or '..' in filename:
                return jsonify({"error": "Invalid filename"}), 400

            thumbnails_dir = Path(output_dir) / 'thumbnails'
            thumbnail_path = thumbnails_dir / safe_filename

            try:
                resolved = PathValidator.resolve_managed_file(
                    thumbnail_path,
                    [thumbnails_dir],
                )
            except ValueError:
                return jsonify({"error": "Access denied"}), 403
            except FileNotFoundError:
                return jsonify({"error": "Thumbnail not found"}), 404

            # Serve with caching headers
            return send_file(
                resolved,
                mimetype='image/jpeg',
                as_attachment=False,
                max_age=3600  # Cache 1 hour
            )

        except Exception as e:
            current_app.logger.error(f"Error serving thumbnail: {e}")
            return jsonify({"error": "Failed to serve thumbnail"}), 500

    return bp
