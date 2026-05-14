@echo off
chcp 65001 >nul
echo ==========================================
echo  彩票图片识别功能 - 依赖安装脚本
echo ==========================================
echo.

echo [1/4] 正在安装 Pillow（图像处理）...
pip install Pillow -i https://mirrors.aliyun.com/pypi/simple/ -q
if errorlevel 1 (
    echo [错误] Pillow 安装失败
    pause
    exit /b 1
)

echo [2/4] 正在安装 Shapely（几何计算）...
pip install Shapely -i https://mirrors.aliyun.com/pypi/simple/ -q
if errorlevel 1 (
    echo [错误] Shapely 安装失败
    pause
    exit /b 1
)

echo [3/4] 正在安装 OpenCV（计算机视觉）...
pip install opencv-python -i https://mirrors.aliyun.com/pypi/simple/ -q
if errorlevel 1 (
    echo [错误] OpenCV 安装失败
    pause
    exit /b 1
)

echo [4/4] 正在安装 PaddlePaddle CPU 版（深度学习框架）...
echo 注意：此步骤可能需要几分钟，请耐心等待...
pip install paddlepaddle -i https://mirrors.aliyun.com/pypi/simple/ -q
if errorlevel 1 (
    echo [错误] PaddlePaddle 安装失败，尝试使用百度镜像...
    pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple -q
    if errorlevel 1 (
        echo [警告] PaddlePaddle 安装失败，图片识别功能可能无法使用
        echo         请手动安装: pip install paddlepaddle
    )
)

echo.
echo ==========================================
echo  依赖安装完成！
echo ==========================================
echo.
echo 安装 PaddleOCR（可选，如需图片识别功能）:
echo   pip install paddleocr -i https://mirrors.aliyun.com/pypi/simple/
echo.
echo 注意：首次运行 OCR 功能时，会自动下载模型文件（约 10MB）
echo.
pause
