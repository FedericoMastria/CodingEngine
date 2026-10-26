@echo off
rem CodingEngine `ai` wrapper. Add this folder to PATH to run `ai "task" +file` from any console.
rem Prefers uv's managed CPython (>=3.14); falls back to `python` on PATH. Exit code is propagated.
rem PREFER the native launcher: `uv tool install --editable .` installs ai.exe on PATH.
rem cmd.exe re-parses this wrapper's command line: PowerShell strips quotes from arguments without
rem spaces, so an unquoted > | & in an argument becomes a redirection/pipe BEFORE Python starts.
setlocal
where uv >nul 2>&1
if %errorlevel%==0 (
    uv run --no-project --quiet --python ">=3.14" "%~dp0ai_router.py" %*
) else (
    python "%~dp0ai_router.py" %*
)
exit /b %errorlevel%
