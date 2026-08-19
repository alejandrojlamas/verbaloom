/**
 * File Manager - File list management and operations
 *
 * Handles file list display, selection management, batch operations
 * (download/delete), and individual file actions.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { FileActions } from './file-actions.js';
import { t } from '../i18n/i18n.js';

export const FileManager = {
    /**
     * Initialize file manager
     */
    initialize() {
        this.setupEventListeners();
        this.refreshFileList();
    },

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Listen for file list changes
        window.addEventListener('fileListChanged', () => {
            this.refreshFileList();
        });

        // Re-render the list when the UI locale changes: rows are built with
        // t() at render time, so a full refresh applies the new translations
        // (including the "Download N" / "Delete N" selection buttons).
        window.addEventListener('localeChanged', () => {
            this.refreshFileList();
        });

        // Select all checkbox
        const selectAllCheckbox = DomHelpers.getElement('selectAllFiles');
        if (selectAllCheckbox) {
            selectAllCheckbox.addEventListener('change', () => {
                this.toggleSelectAll();
            });
        }

        // Batch download button
        const batchDownloadBtn = DomHelpers.getElement('batchDownloadBtn');
        if (batchDownloadBtn) {
            batchDownloadBtn.addEventListener('click', () => {
                this.downloadSelectedFiles();
            });
        }

        // Batch delete button
        const batchDeleteBtn = DomHelpers.getElement('batchDeleteBtn');
        if (batchDeleteBtn) {
            batchDeleteBtn.addEventListener('click', () => {
                this.deleteSelectedFiles();
            });
        }
    },

    /**
     * Refresh file list from server
     */
    async refreshFileList() {
        const loadingDiv = DomHelpers.getElement('fileListLoading');
        const containerDiv = DomHelpers.getElement('fileManagementContainer');
        const tableBody = DomHelpers.getElement('fileTableBody');
        const emptyDiv = DomHelpers.getElement('fileListEmpty');

        if (!tableBody) return;

        // Show loading, hide container (use inline style to override)
        if (loadingDiv) loadingDiv.style.display = 'block';
        if (containerDiv) containerDiv.style.display = 'none';

        try {
            const data = await ApiClient.getFileList();

            // Hide loading, show container (use inline style to override)
            if (loadingDiv) loadingDiv.style.display = 'none';
            if (containerDiv) containerDiv.style.display = 'block';

            // Clear existing table rows
            tableBody.innerHTML = '';

            // Clear selected files
            StateManager.setState('files.selected', new Set());
            StateManager.setState('files.quality.activeFilename', null);
            this.hideFileQualityPanel();

            // Reset "Select All" checkbox
            const selectAllCheckbox = DomHelpers.getElement('selectAllFiles');
            if (selectAllCheckbox) {
                selectAllCheckbox.checked = false;
            }

            this.updateFileSelectionButtons();

            if (data.files.length === 0) {
                if (emptyDiv) emptyDiv.style.display = 'block';
                const fileTable = containerDiv.querySelector('.file-table');
                if (fileTable) {
                    fileTable.style.display = 'none';
                }
            } else {
                if (emptyDiv) emptyDiv.style.display = 'none';
                const fileTable = containerDiv.querySelector('.file-table');
                if (fileTable) {
                    fileTable.style.display = 'table';
                }

                // Populate table with files
                data.files.forEach(file => {
                    const row = this.createFileRow(file);
                    tableBody.appendChild(row);
                });
            }

            // Update totals
            DomHelpers.setText('totalFileCount', data.total_files);
            DomHelpers.setText('totalFileSize', `${data.total_size_mb} MB`);

            // Store in state
            StateManager.setState('files.managed', data.files);

        } catch (error) {
            if (loadingDiv) loadingDiv.style.display = 'none';
            MessageLogger.showMessage(t('files:load_failed', { error: error.message }), 'error');
        }
    },

    /**
     * Create file row element
     * @param {Object} file - File data object
     * @returns {HTMLElement} Table row element
     */
    createFileRow(file) {
        const row = document.createElement('tr');

        const modifiedDate = new Date(file.modified_date);
        const formattedDate = modifiedDate.toLocaleString();

        const isAudioFile = file.file_type === 'opus' || file.file_type === 'mp3';
        const fileIconClass = file.file_type === 'epub' ? 'book' :
                        file.file_type === 'srt' ? 'movie' :
                        file.file_type === 'pdf' ? 'picture_as_pdf' :
                        file.file_type === 'txt' ? 'description' :
                        isAudioFile ? 'headphones' : 'attach_file';

        const supportsTTS = ['epub', 'txt', 'srt'].includes(file.file_type);
        const safeFilename = DomHelpers.escapeHtml(file.filename);
        const tooltipInfo = `${file.file_type.toUpperCase()} • ${file.size_mb} MB • ${formattedDate}`;

        row.innerHTML = `
            <td style="width: 36px; padding: 0.5rem;">
                <input type="checkbox" class="file-checkbox" data-filename="${safeFilename}">
            </td>
            <td style="max-width: 0;">
                <span class="clickable-filename" data-filename="${safeFilename}" data-action="open" title="${tooltipInfo}">
                    <span class="material-symbols-outlined file-icon-cell">${fileIconClass}</span>
                    <span class="filename-text">${safeFilename}</span>
                </span>
            </td>
            <td class="file-row-actions">
                <div class="file-action-group file-action-group--compact"></div>
            </td>
        `;

        const checkbox = row.querySelector('.file-checkbox');
        if (checkbox) {
            checkbox.addEventListener('change', () => this.toggleFileSelection(file.filename));
        }

        const openLink = row.querySelector('.clickable-filename');
        if (openLink) {
            openLink.addEventListener('click', () => FileActions.open(file.filename));
        }

        const actionsHost = row.querySelector('.file-action-group');

        if (supportsTTS) {
            const audiobookBtn = document.createElement('button');
            audiobookBtn.type = 'button';
            audiobookBtn.className = 'file-action-btn audiobook';
            audiobookBtn.title = t('translation:audiobook_btn_title');
            audiobookBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">headphones</span>';
            audiobookBtn.addEventListener('click', () => window.createAudiobook(file.filename, file.file_path));
            actionsHost.appendChild(audiobookBtn);
        }

        const qualityBtn = document.createElement('button');
        qualityBtn.type = 'button';
        qualityBtn.className = 'file-action-btn quality';
        qualityBtn.title = t('files:quality_action');
        qualityBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">fact_check</span>';
        qualityBtn.addEventListener('click', () => this.showFileQuality(file.filename));
        actionsHost.appendChild(qualityBtn);

        const refreshAfterDelete = () => this.refreshFileList();
        ['open', 'reveal', 'download', 'delete'].forEach(action => {
            actionsHost.appendChild(FileActions.createActionButton({
                action,
                filename: file.filename,
                variant: 'compact',
                onAfter: action === 'delete' ? refreshAfterDelete : undefined
            }));
        });

        return row;
    },

    hideFileQualityPanel() {
        const panel = DomHelpers.getElement('fileQualityPanel');
        if (panel) {
            panel.classList.add('hidden');
        }
    },

    showQualityLoading(filename) {
        const panel = DomHelpers.getElement('fileQualityPanel');
        const content = DomHelpers.getElement('fileQualityContent');
        if (!panel || !content) return;

        panel.classList.remove('hidden');
        content.innerHTML = `
            <div class="file-quality-state">
                <span class="material-symbols-outlined">hourglass_top</span>
                <div>
                    <h3>${DomHelpers.escapeHtml(t('files:quality_loading'))}</h3>
                    <p>${DomHelpers.escapeHtml(filename)}</p>
                </div>
            </div>
        `;
    },

    async showFileQuality(filename) {
        StateManager.setState('files.quality.activeFilename', filename);
        this.showQualityLoading(filename);

        try {
            const report = await ApiClient.getFileQuality(filename);
            this.renderFileQualityReport(report);
        } catch (error) {
            const content = DomHelpers.getElement('fileQualityContent');
            if (content) {
                content.innerHTML = `
                    <div class="file-quality-state file-quality-state--error">
                        <span class="material-symbols-outlined">error</span>
                        <div>
                            <h3>${DomHelpers.escapeHtml(t('files:quality_error'))}</h3>
                            <p>${DomHelpers.escapeHtml(error.message)}</p>
                        </div>
                    </div>
                `;
            }
        }
    },

    renderFileQualityReport(report) {
        const panel = DomHelpers.getElement('fileQualityPanel');
        const content = DomHelpers.getElement('fileQualityContent');
        if (!panel || !content) return;

        panel.classList.remove('hidden');

        const status = report.status || 'warn';
        const statusLabel = t(`files:quality_status_${status}`);
        const text = report.text || {};
        const mexican = report.mexican_spanish || {};
        const epub = report.epub;
        const suggestions = Array.isArray(report.glossary_suggestions)
            ? report.glossary_suggestions
            : [];
        const warnings = Array.isArray(report.warnings) ? report.warnings : [];
        const errors = Array.isArray(report.errors) ? report.errors : [];
        const messages = [...errors, ...warnings];

        content.innerHTML = `
            <div class="file-quality-header">
                <div>
                    <h3>${DomHelpers.escapeHtml(report.filename || '')}</h3>
                    <p>${DomHelpers.escapeHtml(t('files:quality_ready'))}</p>
                </div>
                <span class="file-quality-status file-quality-status--${DomHelpers.escapeHtml(status)}">
                    ${DomHelpers.escapeHtml(statusLabel)}
                </span>
            </div>

            <div class="file-quality-metrics">
                ${this.renderQualityMetric('article', t('files:quality_metric_words'), text.words || 0)}
                ${this.renderQualityMetric('segment', t('files:quality_metric_paragraphs'), text.paragraphs || 0)}
                ${this.renderQualityMetric('language', t('files:quality_metric_mexican_issues'), mexican.total || 0)}
                ${this.renderQualityMetric('data_object', t('files:quality_metric_mojibake'), report.mojibake_score || 0)}
            </div>

            ${this.renderQualityMessages(messages)}
            ${this.renderMexicanSpanishSection(mexican, suggestions)}
            ${this.renderEpubSection(epub)}

            <div class="file-quality-footer">
                <button type="button" class="btn btn-primary" id="fileQualityDownloadBtn">
                    <span class="material-symbols-outlined">download</span>
                    <span>${DomHelpers.escapeHtml(t('files:quality_download'))}</span>
                </button>
            </div>
        `;

        const downloadBtn = DomHelpers.getElement('fileQualityDownloadBtn');
        if (downloadBtn && report.filename) {
            downloadBtn.addEventListener('click', () => FileActions.download(report.filename));
        }
    },

    renderQualityMetric(icon, label, value) {
        return `
            <div class="file-quality-metric">
                <span class="material-symbols-outlined">${DomHelpers.escapeHtml(icon)}</span>
                <div>
                    <strong>${DomHelpers.escapeHtml(String(value))}</strong>
                    <span>${DomHelpers.escapeHtml(label)}</span>
                </div>
            </div>
        `;
    },

    renderQualityMessages(messages) {
        if (!messages.length) return '';
        const items = messages
            .slice(0, 6)
            .map(message => `<li>${DomHelpers.escapeHtml(message)}</li>`)
            .join('');
        return `
            <div class="file-quality-block">
                <h4>${DomHelpers.escapeHtml(t('files:quality_section_findings'))}</h4>
                <ul class="file-quality-list">${items}</ul>
            </div>
        `;
    },

    renderMexicanSpanishSection(mexican, suggestions) {
        if (!suggestions.length) {
            return `
                <div class="file-quality-block">
                    <h4>${DomHelpers.escapeHtml(t('files:quality_section_mexican'))}</h4>
                    <p class="file-quality-muted">${DomHelpers.escapeHtml(t('files:quality_no_issues'))}</p>
                </div>
            `;
        }

        const rows = suggestions
            .slice(0, 8)
            .map(item => {
                const examples = Array.isArray(item.examples) && item.examples.length
                    ? `<p><span>${DomHelpers.escapeHtml(t('files:quality_examples'))}</span> ${DomHelpers.escapeHtml(item.examples.join(', '))}</p>`
                    : '';
                return `
                    <div class="file-quality-suggestion">
                        <div>
                            <strong>${DomHelpers.escapeHtml(item.label || item.code)}</strong>
                            <span>${DomHelpers.escapeHtml(String(item.count || 0))}</span>
                        </div>
                        <p><span>${DomHelpers.escapeHtml(t('files:quality_suggestion_policy'))}</span> ${DomHelpers.escapeHtml(item.suggestion || '')}</p>
                        ${examples}
                    </div>
                `;
            })
            .join('');

        return `
            <div class="file-quality-block">
                <h4>${DomHelpers.escapeHtml(t('files:quality_section_mexican'))}</h4>
                <div class="file-quality-suggestions">${rows}</div>
            </div>
        `;
    },

    renderEpubSection(epub) {
        if (!epub) return '';
        const navStatus = epub.has_nav ? t('files:quality_epub_valid') : t('files:quality_epub_missing');
        const opfStatus = epub.has_opf ? t('files:quality_epub_valid') : t('files:quality_epub_missing');
        return `
            <div class="file-quality-block">
                <h4>${DomHelpers.escapeHtml(t('files:quality_section_epub'))}</h4>
                <div class="file-quality-epub-grid">
                    ${this.renderQualityMetric('menu_book', t('files:quality_epub_docs'), epub.content_documents || 0)}
                    ${this.renderQualityMetric('format_list_numbered', t('files:quality_epub_spine'), epub.spine_items || 0)}
                    ${this.renderQualityMetric('account_tree', t('files:quality_epub_nav'), navStatus)}
                    ${this.renderQualityMetric('inventory_2', 'OPF', opfStatus)}
                </div>
                <dl class="file-quality-details">
                    <div><dt>${DomHelpers.escapeHtml(t('files:quality_epub_title'))}</dt><dd>${DomHelpers.escapeHtml(epub.title || t('files:quality_epub_missing'))}</dd></div>
                    <div><dt>${DomHelpers.escapeHtml(t('files:quality_epub_language'))}</dt><dd>${DomHelpers.escapeHtml(epub.language || t('files:quality_epub_missing'))}</dd></div>
                </dl>
            </div>
        `;
    },

    /**
     * Toggle file selection
     * @param {string} filename - Filename to toggle
     */
    toggleFileSelection(filename) {
        const selectedFiles = StateManager.getState('files.selected');

        if (selectedFiles.has(filename)) {
            selectedFiles.delete(filename);
        } else {
            selectedFiles.add(filename);
        }

        StateManager.setState('files.selected', selectedFiles);
        this.updateFileSelectionButtons();
    },

    /**
     * Select all files
     */
    selectAllFiles() {
        const checkboxes = DomHelpers.getElements('.file-checkbox');
        const selectedFiles = new Set();

        checkboxes.forEach(checkbox => {
            checkbox.checked = true;
            const filename = checkbox.getAttribute('data-filename');
            selectedFiles.add(filename);
        });

        StateManager.setState('files.selected', selectedFiles);
        this.updateFileSelectionButtons();
    },

    /**
     * Deselect all files
     */
    deselectAllFiles() {
        const checkboxes = DomHelpers.getElements('.file-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.checked = false;
        });

        StateManager.setState('files.selected', new Set());
        this.updateFileSelectionButtons();
    },

    /**
     * Toggle select all
     */
    toggleSelectAll() {
        const checkboxes = DomHelpers.getElements('.file-checkbox');
        const selectAllFiles = DomHelpers.getElement('selectAllFiles');

        // Use the Select All checkbox state
        const isChecked = selectAllFiles.checked;

        if (isChecked) {
            this.selectAllFiles();
        } else {
            this.deselectAllFiles();
        }
    },

    /**
     * Update file selection button states
     */
    updateFileSelectionButtons() {
        const selectedFiles = StateManager.getState('files.selected');
        const hasSelection = selectedFiles.size > 0;

        // Update button states
        DomHelpers.setDisabled('batchDownloadBtn', !hasSelection);
        DomHelpers.setDisabled('batchDeleteBtn', !hasSelection);

        // Update "Select All" checkbox state based on actual selection
        const checkboxes = DomHelpers.getElements('.file-checkbox');
        const selectAllCheckbox = DomHelpers.getElement('selectAllFiles');
        if (selectAllCheckbox && checkboxes.length > 0) {
            const allChecked = Array.from(checkboxes).every(cb => cb.checked);
            selectAllCheckbox.checked = allChecked;
        }

        // Update button text with count
        const downloadBtn = DomHelpers.getElement('batchDownloadBtn');
        const deleteBtn = DomHelpers.getElement('batchDeleteBtn');
        if (hasSelection) {
            if (downloadBtn) downloadBtn.innerHTML = `<span class="material-symbols-outlined">download</span> ${t('files:download_selected_with_count', { count: selectedFiles.size })}`;
            if (deleteBtn) deleteBtn.innerHTML = `<span class="material-symbols-outlined">delete</span> ${t('files:delete_selected_with_count', { count: selectedFiles.size })}`;
        } else {
            if (downloadBtn) downloadBtn.innerHTML = `<span class="material-symbols-outlined">download</span> ${t('files:download_selected')}`;
            if (deleteBtn) deleteBtn.innerHTML = `<span class="material-symbols-outlined">delete</span> ${t('files:delete_selected')}`;
        }
    },

    async deleteSingleFile(filename) {
        await FileActions.delete(filename, { onDeleted: () => this.refreshFileList() });
    },

    /**
     * Download selected files as ZIP
     */
    async downloadSelectedFiles() {
        const selectedFiles = StateManager.getState('files.selected');

        if (selectedFiles.size === 0) {
            MessageLogger.showMessage(t('files:no_selection_download'), 'error');
            return;
        }

        try {
            const response = await fetch(`${ApiClient.getBaseUrl()}/api/files/batch/download`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filenames: Array.from(selectedFiles)
                })
            });

            if (response.ok) {
                // Download the zip file
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `translated_files_${new Date().getTime()}.zip`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);

                MessageLogger.showMessage(t('files:downloaded_as_zip', { count: selectedFiles.size }), 'success');
            } else {
                const data = await response.json();
                MessageLogger.showMessage(data.error || t('files:download_failed_default'), 'error');
            }
        } catch (error) {
            MessageLogger.showMessage(t('files:download_error', { error: error.message }), 'error');
        }
    },

    /**
     * Delete selected files
     */
    async deleteSelectedFiles() {
        const selectedFiles = StateManager.getState('files.selected');

        if (selectedFiles.size === 0) {
            MessageLogger.showMessage(t('files:no_selection_delete'), 'error');
            return;
        }

        if (!confirm(t('files:confirm_delete_selected', { count: selectedFiles.size }))) {
            return;
        }

        try {
            const response = await fetch(`${ApiClient.getBaseUrl()}/api/files/batch/delete`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filenames: Array.from(selectedFiles)
                })
            });

            const data = await response.json();

            if (response.ok) {
                const message = data.failed.length > 0
                    ? t('files:deleted_summary_with_failed', { count: data.total_deleted, failed: data.failed.length })
                    : t('files:deleted_summary', { count: data.total_deleted });
                MessageLogger.showMessage(message, data.failed.length > 0 ? 'info' : 'success');
                this.refreshFileList();
            } else {
                MessageLogger.showMessage(data.error || t('files:delete_failed_default'), 'error');
            }
        } catch (error) {
            MessageLogger.showMessage(t('files:delete_error', { error: error.message }), 'error');
        }
    },

};

// Selection toggle stays here (state lives in FileManager); the per-file
// actions (open/reveal/download) are exposed globally by FileActions itself.
window.toggleFileSelection = (filename) => FileManager.toggleFileSelection(filename);
window.deleteSingleFile = (filename) => FileManager.deleteSingleFile(filename);
