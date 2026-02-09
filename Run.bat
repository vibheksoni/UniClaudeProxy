@echo off
echo Starting UniClaudeProxy...
python -m uvicorn app.main:app --host 127.0.0.1 --port 9223
pause
