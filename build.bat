@echo off
chcp 65001 >nul
rem ===== Сборка DustHeal.exe для Windows 10/11 x64 =====
rem Требуется Python 3.10-3.13 (64-bit) с https://www.python.org (галочка "Add python.exe to PATH")
cd /d "%~dp0"
where python >nul 2>nul || (echo Python не найден. Установите Python 3.12 x64 с python.org & pause & exit /b 1)
if not exist .venv (python -m venv .venv || (pause & exit /b 1))
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller || (echo Ошибка установки зависимостей & pause & exit /b 1)
pyinstaller --noconfirm --clean --windowed --name DustHeal --icon assets\icon.ico ^
  --add-data "assets;assets" ^
  --exclude-module tkinter --exclude-module matplotlib ^
  app.py || (echo Ошибка сборки & pause & exit /b 1)
rem программный OpenGL (20 МБ) приложению не нужен
if exist dist\DustHeal\_internal\PySide6\opengl32sw.dll del dist\DustHeal\_internal\PySide6\opengl32sw.dll
echo.
echo Готово: dist\DustHeal\DustHeal.exe
echo Папку dist\DustHeal можно скопировать куда угодно (портативная версия).
pause
