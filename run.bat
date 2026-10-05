@echo off
echo 正在安装依赖（第一次需要几分钟，以后直接启动）...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo 依赖安装失败：请先安装 Python 3.10-3.12 并勾选 Add to PATH
    pause
    exit /b 1
)
echo 启动实时字幕...
python main.py
pause
