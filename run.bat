@echo off
REM Reproduces the headline results on Windows (PowerShell or cmd).
REM Equivalent to run.sh — runs the same three steps in sequence.
REM See README.md "How to run" for full details.

echo == 1/3: inspecting the data ==
python src\load_data.py data\twcs_sample.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo == 2/3: running one example through the live agent ==
python src\agent.py "@AppleSupport my battery is draining so fast since the update, please help"
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo == 3/3: running the full evaluation harness ==
python eval\run_eval.py
if %errorlevel% neq 0 exit /b %errorlevel%
python eval\judge_agreement.py
if %errorlevel% neq 0 exit /b %errorlevel%
