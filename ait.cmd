@echo off
rem `ait <command>`: run a command and record its failure for `ai --fix` (spec W04).
rem An explicit opt-in runner: python / npm / pytest themselves are never aliased.
rem PREFER the native launcher: `uv tool install --editable .` installs ait.exe on PATH.
rem cmd.exe re-parses this wrapper's command line: PowerShell strips quotes from arguments without
rem spaces, so an unquoted > | & in an argument becomes a redirection/pipe BEFORE Python starts.
setlocal
where uv >nul 2>&1
if %errorlevel%==0 (
    uv run --no-project --quiet --python ">=3.12" "%~dp0ai_router.py" --ait %*
) else (
    python "%~dp0ai_router.py" --ait %*
)
exit /b %errorlevel%
