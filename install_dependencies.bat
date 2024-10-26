@echo off
echo Checking Python version...
python --version > tmp.txt
set /p PYTHON_VERSION=<tmp.txt
del tmp.txt

echo Installing in Python %PYTHON_VERSION%

REM Create and activate virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing/Updating PyTorch...
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Installing/Updating dependencies...
pip install --upgrade -r requirements.txt

echo Creating necessary directories...
mkdir "C:\Users\danie\OneDrive - SCP Ozona SL\Escritorio\RAG\knowledge" 2>nul
mkdir "C:\Users\danie\OneDrive - SCP Ozona SL\Escritorio\RAG\db" 2>nul

echo Installation complete! 
echo To run the application:
echo 1. Make sure you're in the virtual environment (run 'venv\Scripts\activate')
echo 2. Run 'python main.py'
pause