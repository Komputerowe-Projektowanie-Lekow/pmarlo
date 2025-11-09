@echo off
REM Launcher script for pmarlo webapp with openmm-torch support
REM This ensures the app runs in the correct conda environment

echo Activating ommtorch environment...
call micromamba activate ommtorch

if errorlevel 1 (
    echo Failed to activate ommtorch environment
    echo Make sure micromamba is installed and the ommtorch environment exists
    pause
    exit /b 1
)

echo Starting Streamlit app...
streamlit run pmarlo_webapp\app\app.py

if errorlevel 1 (
    echo Streamlit failed to start
    echo Make sure streamlit is installed in the ommtorch environment:
    echo   micromamba install -c conda-forge streamlit
    pause
    exit /b 1
)
