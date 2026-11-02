@echo off
rem am2Strata from a release: runs from this folder and keeps models and settings in %%AM2STRATA_DATA%%.
rem   am2strata                        start the server (http://127.0.0.1:8000)
rem   am2strata pull ^<repo^> [^<quant^>]  download a model from Hugging Face; no quant lists what the repository has
setlocal
set "HERE=%~dp0"
if "%AM2STRATA_DATA%"=="" set "AM2STRATA_DATA=%LOCALAPPDATA%\am2strata"
if not exist "%AM2STRATA_DATA%" mkdir "%AM2STRATA_DATA%"
if /i "%~1"=="pull" goto pull
if /i "%~1"=="serve" shift
if not exist "%AM2STRATA_DATA%\strata-*.json" (
    echo No model yet. See QUICKSTART.md for a small one to start with.
    exit /b 1
)
"%HERE%bin\am2strata.exe" --models-dir "%AM2STRATA_DATA%" %1 %2 %3 %4 %5 %6
exit /b %ERRORLEVEL%
:pull
if "%~2"=="" (
    echo usage: am2strata pull ^<repo^> [^<quant^>]
    exit /b 2
)
if "%~3"=="" (
    "%HERE%bin\am2strata-run.exe" pull "%~2" --models-dir "%AM2STRATA_DATA%" --data-dir "%AM2STRATA_DATA%\data"
) else (
    "%HERE%bin\am2strata-run.exe" pull "%~2" --quant "%~3" --models-dir "%AM2STRATA_DATA%" --data-dir "%AM2STRATA_DATA%\data"
)
exit /b %ERRORLEVEL%
