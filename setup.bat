@echo off
rem setup.bat — Windows Command Prompt equivalent of setup.sh
rem
rem Uses conda when available; otherwise falls back to a plain Python venv (.venv).
rem Force the venv path even when conda is present:
rem     set FORCE_VENV=1 && setup.bat
rem
rem Usage: double-click, or run `setup.bat` from Command Prompt.
setlocal EnableDelayedExpansion

set "ENV_NAME=iccp-oct"
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv"
set "VIMBA_URL=https://www.alliedvision.com/en/support/software-downloads/vimba-x-sdk/vimba-x"

echo ==================================================
echo   ICCP 2026 environment setup (Windows)
echo   Project : %PROJECT_DIR%
echo ==================================================

rem ── 1. Locate conda (skipped if FORCE_VENV=1) ─────────────────────────────
set "CONDA_EXE="
if "%FORCE_VENV%"=="1" (
    echo FORCE_VENV=1 - skipping conda, using a Python venv.
) else (
    where conda >nul 2>nul && for /f "delims=" %%i in ('where conda') do if not defined CONDA_EXE set "CONDA_EXE=%%i"
    if not defined CONDA_EXE (
        for %%p in (
            "%ProgramData%\Anaconda3\Scripts\conda.exe"
            "%USERPROFILE%\Anaconda3\Scripts\conda.exe"
            "%USERPROFILE%\miniconda3\Scripts\conda.exe"
            "%LocalAppData%\miniconda3\Scripts\conda.exe"
        ) do if not defined CONDA_EXE if exist %%p set "CONDA_EXE=%%~p"
    )
)

rem ── 2. Create the environment (conda if available, else venv) ─────────────
echo.
if defined CONDA_EXE (
    set "USE_CONDA=1"
    echo conda   : !CONDA_EXE!
    "!CONDA_EXE!" env list | findstr /b /c:"%ENV_NAME% " >nul 2>nul
    if !errorlevel! equ 0 (
        echo Environment '%ENV_NAME%' already exists - updating...
        "!CONDA_EXE!" env update -n %ENV_NAME% -f "%PROJECT_DIR%environment.yml" --prune
    ) else (
        echo Creating environment '%ENV_NAME%'...
        "!CONDA_EXE!" env create -f "%PROJECT_DIR%environment.yml"
    )
    if !errorlevel! neq 0 goto :fail
    for /f "delims=" %%i in ('"!CONDA_EXE!" run -n %ENV_NAME% python -c "import sys; print(sys.executable)"') do set "PYTHON=%%i"
    for %%i in ("!PYTHON!") do set "PYDIR=%%~dpi"
    set "PIP=!PYDIR!Scripts\pip.exe"
) else (
    set "USE_CONDA=0"
    echo conda   : not found - using a Python venv (.venv) instead.
    where python >nul 2>nul
    if !errorlevel! neq 0 (
        echo ERROR: neither conda nor python found.
        echo        Install Python 3.12+ from python.org or Anaconda, then re-run.
        goto :fail
    )
    echo Creating venv at: %VENV_DIR%
    python -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 goto :fail
    set "PYTHON=%VENV_DIR%\Scripts\python.exe"
    set "PIP=%VENV_DIR%\Scripts\pip.exe"
    "!PYTHON!" -m pip install --upgrade pip
    echo.
    echo Installing Python dependencies (mirrors environment.yml)...
    "!PIP!" install numpy==2.4.6 scipy==1.18.0 matplotlib==3.11.0 Pillow==12.2.0 pyserial==3.5 pylablib==1.4.5 pyftdi opencv-python ipykernel
    if !errorlevel! neq 0 goto :fail
)
echo Python  : !PYTHON!

rem ── 3. Install vmbpy from the Vimba X SDK wheel (Allied Vision only) ───────
echo.
echo Looking for vmbpy (Allied Vision Vimba X SDK)...
set "VMBPY_WHEEL="
for /f "delims=" %%i in ('dir /s /b "%ProgramFiles%\Allied Vision\vmbpy-*.whl" 2^>nul') do if not defined VMBPY_WHEEL set "VMBPY_WHEEL=%%i"
if defined VMBPY_WHEEL (
    "!PIP!" install "!VMBPY_WHEEL!"
    echo vmbpy installed from: !VMBPY_WHEEL!
) else (
    echo WARNING: Vimba X SDK ^(vmbpy wheel^) not found.
    echo   Only needed for Allied Vision cameras - skip this if you use an IDS camera.
    echo   Download Vimba X from:
    echo     %VIMBA_URL%
    echo   Installer : VimbaX_Setup-2026-1-Win64.exe
    echo   Wheel     : C:\Program Files\Allied Vision\Vimba X\api\python\vmbpy-*.whl
    echo   Then install it with:
    echo     "!PIP!" install ^<path-to-vmbpy-*.whl^>
)

rem ── 4. Register Jupyter kernel ─────────────────────────────────────────────
echo.
echo Registering Jupyter kernel...
"!PYTHON!" -m ipykernel install --user --name %ENV_NAME% --display-name "Python (%ENV_NAME%)"

rem ── 5. Summary ──────────────────────────────────────────────────────────────
echo.
echo ==================================================
echo   Setup complete!
echo.
if "!USE_CONDA!"=="1" (
    echo   Activate :  conda activate %ENV_NAME%
    echo   VSCode   :  Ctrl+Shift+P - Python: Select Interpreter - Python ^(%ENV_NAME%^)
) else (
    echo   Activate :  .venv\Scripts\activate
    echo   VSCode   :  Ctrl+Shift+P - Python: Select Interpreter - .\.venv
)
echo.
echo   Platform notes (Windows):
echo     Stage  : requires Thorlabs Kinesis / APT driver installed
echo     Camera : Allied Vision - Vimba X SDK (VimbaX_Setup-2026-1-Win64.exe)
echo              IDS - IDS Software Suite 4.97 + pip install pyueye
echo ==================================================
exit /b 0

:fail
echo.
echo Setup failed - see the error above.
exit /b 1
