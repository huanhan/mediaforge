@echo off
setlocal
cd /d "%~dp0"
"%~dp0python\python.exe" -I "%~dp0launch.py" --check
set "H3_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %H3_EXIT_CODE%
