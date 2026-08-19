@echo off
echo ============================================
echo   Test de Variance - Meme prompt, meme texte
echo ============================================
echo.

cd /d "%~dp0\..\.."

python -m tools.prompt_optimizer.test_variance --config tools/prompt_optimizer/prompt_optimizer_config.yaml --runs 5

echo.
pause
