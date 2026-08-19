@echo off
cd /d "%~dp0.."
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python -m pytest tests/ -v -rsw -W always --tb=no %*
) else (
    python -m pytest tests/ -v -rsw -W always --tb=no %*
)
pause
