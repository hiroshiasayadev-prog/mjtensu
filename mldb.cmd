@echo off
setlocal DisableDelayedExpansion
set "MLDB_REPO_ROOT=%~dp0"
pushd "%MLDB_REPO_ROOT%" >nul || exit /b 1

rem Load repository-local defaults. Explicit process environment wins.
if exist "%MLDB_REPO_ROOT%.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%MLDB_REPO_ROOT%.env") do (
    if not "%%A"=="" if not defined %%A set "%%A=%%B"
  )
)

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m mldb_v2.src.cli %*
) else (
  python -m mldb_v2.src.cli %*
)
set "MLDB_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %MLDB_EXIT_CODE%
