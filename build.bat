@echo off
echo 正在安装依赖...
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 (
    echo 依赖安装失败
    pause
    exit /b 1
)
echo 正在打包...
pyinstaller sherpa-live-sub.spec
if errorlevel 1 (
    echo 打包失败
    pause
    exit /b 1
)
echo.
echo 构建完成：dist\实时字幕.exe
pause
