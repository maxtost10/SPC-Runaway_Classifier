@echo off
REM Navigate to your project directory
cd C:\Users\Max Tost

REM Use PowerShell to activate the virtual environment and run Jupyter Notebook
powershell -Command ". .\.venv\Scripts\Activate; jupyter notebook"