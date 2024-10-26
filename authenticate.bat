@echo off
echo Installing Hugging Face CLI...
pip install --upgrade huggingface_hub

echo.
echo Initiating Hugging Face login...
huggingface-cli login