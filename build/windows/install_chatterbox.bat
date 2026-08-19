@echo off
REM ============================================================================
REM Robust Chatterbox TTS installer
REM Includes PyTorch with CUDA and all required dependencies
REM ============================================================================

echo ========================================
echo Installing Chatterbox TTS
echo ========================================
echo.

REM Verify that Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or is not available on PATH.
    echo Install Python 3.8+ from https://www.python.org/downloads/
    echo Select "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python is installed:
python --version
echo.

REM Resolve the script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Check whether the virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating the virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        echo Make sure the venv module is installed.
        echo.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment detected.
)
echo.

REM Activate the virtual environment
echo [INFO] Activating the virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate the virtual environment.
    echo.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.
echo.

REM Update pip, setuptools, and wheel
echo [INFO] Updating pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [WARNING] The pip update failed; continuing...
)
echo.

REM Detect CUDA
echo [INFO] Detecting CUDA...
set CUDA_AVAILABLE=0
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set CUDA_AVAILABLE=1
    echo [OK] NVIDIA GPU with CUDA detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo.
) else (
    echo [WARNING] No NVIDIA/CUDA GPU detected.
    echo Installation will use CPU mode, which is slower.
    echo.
)

REM Install base dependencies from requirements.txt
echo [INFO] Installing base dependencies...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install base dependencies.
        echo.
        pause
        exit /b 1
    )
    echo [OK] Base dependencies installed.
) else (
    echo [WARNING] requirements.txt was not found; skipping this step.
)
echo.

REM Remove existing PyTorch packages to prevent conflicts
echo [INFO] Removing existing PyTorch packages...
python -m pip uninstall -y torch torchaudio torchvision 2>nul
echo.

REM Install the CUDA or CPU PyTorch build as available
if "%CUDA_AVAILABLE%"=="1" (
    echo [INFO] Installing PyTorch with CUDA 12.1 support...
    echo This step may take several minutes...
    python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo [ERROR] Failed to install PyTorch with CUDA.
        echo Trying the CPU build...
        python -m pip install torch torchaudio
        if errorlevel 1 (
            echo [ERROR] Failed to install PyTorch.
            echo.
            pause
            exit /b 1
        )
    ) else (
        echo [OK] PyTorch with CUDA installed.
    )
) else (
    echo [INFO] Installing PyTorch in CPU mode...
    echo This step may take several minutes...
    python -m pip install torch torchaudio
    if errorlevel 1 (
        echo [ERROR] Failed to install PyTorch.
        echo.
        pause
        exit /b 1
    )
    echo [OK] PyTorch CPU installed.
)
echo.

REM Verify the PyTorch installation
echo [INFO] Verifying the PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo [ERROR] PyTorch is not installed correctly.
    echo.
    pause
    exit /b 1
)
echo.

REM Verify CUDA inside PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None" 2>nul
echo.

REM Install Chatterbox TTS
echo [INFO] Installing Chatterbox TTS...
python -m pip install chatterbox-tts
if errorlevel 1 (
    echo [ERROR] Failed to install Chatterbox TTS.
    echo.
    echo Trying an alternative --no-deps installation, then resolving dependencies...
    python -m pip install --no-deps chatterbox-tts
    python -m pip install gruut-ipa typing_extensions transformers soundfile phonemizer pysbd
    if errorlevel 1 (
        echo [ERROR] Alternative installation failed.
        echo.
        pause
        exit /b 1
    )
)
echo [OK] Chatterbox TTS installed.
echo.

REM Install ffmpeg if needed for Edge TTS
echo [INFO] Checking ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffmpeg is not installed or is not available on PATH.
    echo ffmpeg is recommended for audio conversion.
    echo Download it from: https://ffmpeg.org/download.html
    echo.
) else (
    echo [OK] ffmpeg is installed.
    echo.
)

REM Create a verification script
echo [INFO] Creating the verification script...
echo import sys > test_chatterbox_install.py
echo import torch >> test_chatterbox_install.py
echo print("=" * 60) >> test_chatterbox_install.py
echo print("CHATTERBOX TTS INSTALLATION TEST") >> test_chatterbox_install.py
echo print("=" * 60) >> test_chatterbox_install.py
echo print(f"Python version: {sys.version}") >> test_chatterbox_install.py
echo print(f"PyTorch version: {torch.__version__}") >> test_chatterbox_install.py
echo print(f"CUDA available: {torch.cuda.is_available()}") >> test_chatterbox_install.py
echo if torch.cuda.is_available(): >> test_chatterbox_install.py
echo     print(f"GPU count: {torch.cuda.device_count()}") >> test_chatterbox_install.py
echo     for i in range(torch.cuda.device_count()): >> test_chatterbox_install.py
echo         print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") >> test_chatterbox_install.py
echo         print(f"    Total VRAM: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB") >> test_chatterbox_install.py
echo else: >> test_chatterbox_install.py
echo     print("CPU-only mode") >> test_chatterbox_install.py
echo print() >> test_chatterbox_install.py
echo try: >> test_chatterbox_install.py
echo     from chatterbox import ChatterboxTTS >> test_chatterbox_install.py
echo     print("[OK] Chatterbox TTS imported successfully") >> test_chatterbox_install.py
echo     print() >> test_chatterbox_install.py
echo     print("Installation completed successfully!") >> test_chatterbox_install.py
echo     print("You can now use Chatterbox TTS in VerbaLoom.") >> test_chatterbox_install.py
echo except Exception as e: >> test_chatterbox_install.py
echo     print(f"[ERROR] Could not import Chatterbox TTS: {e}") >> test_chatterbox_install.py
echo     sys.exit(1) >> test_chatterbox_install.py
echo print("=" * 60) >> test_chatterbox_install.py

echo [OK] Verification script created.
echo.

REM Run the verification
echo [INFO] Testing the installation...
python test_chatterbox_install.py
if errorlevel 1 (
    echo.
    echo [ERROR] The installation test failed.
    echo Review the error messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To use Chatterbox TTS:
echo 1. Start VerbaLoom with start.bat
echo 2. Select "Chatterbox TTS" as the TTS provider in the web interface
echo 3. Optionally upload an audio sample for voice cloning
echo.
echo To test Chatterbox TTS manually:
echo   venv\Scripts\activate
echo   python test_chatterbox_install.py
echo.

if "%CUDA_AVAILABLE%"=="0" (
    echo [NOTE] You are using CPU mode.
    echo For better performance, use an NVIDIA GPU with CUDA.
    echo.
)

echo.
pause
