@echo off
:: Setup script for Windows
:: To add dependencies, edit environment.yml, then re-run this script.

set ENV_NAME=maze-project

where conda >nul 2>&1
if errorlevel 1 (
    echo Error: conda not found. Install Miniconda or Anaconda first.
    echo   https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

conda env list | findstr /B "%ENV_NAME% " >nul 2>&1
if errorlevel 1 (
    echo Creating environment '%ENV_NAME%'...
    conda env create --file "%~dp0environment.yml"
    echo Environment '%ENV_NAME%' created.
) else (
    echo Environment '%ENV_NAME%' already exists. Updating...
    conda env update --name %ENV_NAME% --file "%~dp0environment.yml" --prune
    echo Environment '%ENV_NAME%' updated.
)

echo.
echo Done! Activate the environment with:
echo   conda activate %ENV_NAME%
