@echo off
REM Detached training driver for the OSCD change pipeline. Launched with
REM `start /b` from the shell so a killed tool call cannot kill the run.
REM Uses `cd /d "%~dp0"` so it works no matter which directory it is started
REM from, and never depends on the caller's quoting surviving the spawn.
cd /d "%~dp0"
echo START %DATE% %TIME% > results\train.log
".venv\Scripts\python.exe" -m forest_change.train data\raw results >> results\train.log 2>&1
echo TRAIN_EXIT=%ERRORLEVEL% %DATE% %TIME% >> results\train.log
