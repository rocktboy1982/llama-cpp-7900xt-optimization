@echo off
rem UNTESTED on Windows. Adapted from docs/windows-original-guide.md and docs/llama-cpp-guide.md section 2.4.
rem Edit the three paths below, create the API key file, then run from cmd or PowerShell (not Git Bash).
setlocal
set HIP_VISIBLE_DEVICES=0
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.2
set PATH=%ROCM_PATH%\bin;%PATH%
set SERVER=C:\llama.cpp\build-hip\bin\llama-server.exe
set MODEL=C:\models\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
set /p API_KEY=<C:\llama.cpp\api-key.txt
rem Context: 131072 fits when nothing else uses the GPU; use 122880 if the desktop renders on the 7900 XT.
set CTX=131072

"%SERVER%" ^
    -m "%MODEL%" ^
    -c %CTX% -ngl 999 --parallel 1 ^
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 ^
    --batch-size 2048 --ubatch-size 1024 ^
    --spec-type ngram-simple,draft-mtp --spec-draft-n-max 4 ^
    --jinja --reasoning-format deepseek ^
    --chat-template-kwargs "{\"reasoning_effort\":\"medium\"}" ^
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 ^
    --host 0.0.0.0 --port 11437 --api-key "%API_KEY%"
