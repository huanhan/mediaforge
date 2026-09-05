@echo off
setlocal
cd /d "%~dp0"
"%~dp0python\python.exe" -I "%~dp0launch.py" %*
set "H3_EXIT_CODE=%ERRORLEVEL%"
if not "%H3_EXIT_CODE%"=="0" pause
exit /b %H3_EXIT_CODE%
