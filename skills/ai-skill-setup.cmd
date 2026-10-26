@echo off
rem Double-click setup: distributes ai-orchestrator, Ponytail and Superpowers skills to every agent.
rem Extra arguments are forwarded, e.g.:  ai-skill-setup.cmd --project C:\MyRepo --dry-run
setlocal
where uv >nul 2>&1
if %errorlevel%==0 (
    uv run --no-project --quiet --python ">=3.12" "%~dp0install_skills.py" %*
) else (
    python "%~dp0install_skills.py" %*
)
set "rc=%errorlevel%"
echo.
pause
exit /b %rc%
