@echo off
rem am2Strata from a release: runs from this folder and keeps models and settings in %%AM2STRATA_DATA%%.
rem   am2strata                        start the server (http://127.0.0.1:8000)
rem   am2strata pull ^<repo^> [^<quant^>]  download a model from Hugging Face; no quant lists what the repository has
rem   am2strata agent list^|start^|setup^|reset^|show ^<agent^>  coding agents on this server or another (--url)
rem   am2strata agentrun ^<agent^> [options] [agent args]  run one, as unsloth start ^<agent^> does
rem   am2strata doctor                 what this PC is ready for, and what is missing
rem   am2strata tls^|workshop^|computer^|keys ^<action^> ...  HTTPS, the Workshop, the Computer tab and API keys on the running server
rem   am2strata --version
setlocal
set "HERE=%~dp0"
if "%AM2STRATA_DATA%"=="" set "AM2STRATA_DATA=%LOCALAPPDATA%\am2strata"
if not exist "%AM2STRATA_DATA%" mkdir "%AM2STRATA_DATA%"
if /i "%~1"=="pull" goto pull
if /i "%~1"=="agent" goto agent
if /i "%~1"=="agentrun" goto agentrun
if /i "%~1"=="doctor" goto doctor
if /i "%~1"=="tls" goto run
if /i "%~1"=="workshop" goto run
if /i "%~1"=="computer" goto run
if /i "%~1"=="keys" goto run
if /i "%~1"=="--version" goto version
if /i "%~1"=="-V" goto version
if /i "%~1"=="version" goto version
if /i "%~1"=="serve" shift
if not exist "%AM2STRATA_DATA%\strata-*.json" (
    echo No model yet. See QUICKSTART.md for a small one to start with.
    exit /b 1
)
"%HERE%bin\am2strata.exe" --models-dir "%AM2STRATA_DATA%" --data-dir "%AM2STRATA_DATA%\data" %1 %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%
:doctor
"%HERE%bin\am2strata-run.exe" doctor --data-dir "%AM2STRATA_DATA%\data" %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%
:version
"%HERE%bin\am2strata.exe" --version
exit /b %ERRORLEVEL%
:run
"%HERE%bin\am2strata-run.exe" %1 --models-dir "%AM2STRATA_DATA%" %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%
:agent
"%HERE%bin\am2strata-run.exe" agent --models-dir "%AM2STRATA_DATA%" %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%
:agentrun
"%HERE%bin\am2strata-run.exe" agentrun --models-dir "%AM2STRATA_DATA%" %2 %3 %4 %5 %6 %7 %8 %9
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
