@echo off
REM Wrapper script to run inpainting with proper environment setup

set SCRIPT_DIR=%~dp0
set PLUGIN_DIR=%SCRIPT_DIR%..\..\
set PYTHON_EXE=%PLUGIN_DIR%Python3118\python.exe
set SCRIPT_PATH=%SCRIPT_DIR%ObjectRemovalPipeline.py

REM Set up Python path to include both library directories
set PYTHONPATH=%PLUGIN_DIR%Libraries\ObjectRemoval;%PLUGIN_DIR%Libraries\DepthMap
set PYTHONDONTWRITEBYTECODE=1

REM Run the Python script with all arguments passed through
"%PYTHON_EXE%" "%SCRIPT_PATH%" %*
