@echo off
title 企业知识库问答系统
cd /d "%~dp0"

REM ========== 1. 依赖检查（首次运行自动安装） ==========
python -c "import fastapi, uvicorn, jieba, httpx, yaml, numpy, pypdf" 2>nul
if errorlevel 1 (
    echo [首次运行] 正在安装依赖，请稍候...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo 依赖安装失败，请检查 Python 环境后重试。
        pause
        exit /b 1
    )
)

REM ========== 2. 检查服务是否已在运行（避免端口冲突） ==========
netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo 服务已在运行，正在为您打开浏览器...
    start http://127.0.0.1:8000
    timeout /t 3 >nul
    exit /b 0
)

REM ========== 3. 启动服务并自动打开浏览器 ==========
echo 正在启动企业知识库问答系统...
echo 首次启动需加载示例文档，请等待浏览器自动打开（约 5-10 秒）
echo.
echo 关闭本窗口即可停止服务
echo.
start "" /b cmd /c "timeout /t 8 >nul & start http://127.0.0.1:8000"
python run.py

echo.
echo 服务已停止。
pause
