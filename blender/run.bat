@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  Blender 3D Generator — Windows launcher
REM  Place this file next to generate_3d.py, depth.png, texture.png
REM
REM  Usage:
REM    run.bat [depth.png] [texture.png] [output.glb] [resolution] [depth_scale]
REM
REM  Defaults:
REM    depth.png   texture.png   output.glb   512   0.5
REM ─────────────────────────────────────────────────────────────────────────────

setlocal

REM ── Auto-detect Blender ──────────────────────────────────────────────────────
set BLENDER=
for %%P in (blender.exe) do set BLENDER=%%~$PATH:P

if "%BLENDER%"=="" (
    REM Try common install paths
    for %%D in (
        "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
        "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
        "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"
        "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"
        "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"
        "%LOCALAPPDATA%\Programs\Blender Foundation\Blender 4.3\blender.exe"
        "%LOCALAPPDATA%\Programs\Blender Foundation\Blender 4.2\blender.exe"
    ) do (
        if exist %%D set BLENDER=%%~D
    )
)

if "%BLENDER%"=="" (
    echo [ERROR] Blender が見つかりません。
    echo         インストールしてください: https://www.blender.org/download/
    echo         または blender.exe を PATH に追加してください。
    pause
    exit /b 1
)

echo [INFO] Blender: %BLENDER%

REM ── Arguments (with defaults) ────────────────────────────────────────────────
set DEPTH=%~1
if "%DEPTH%"=="" set DEPTH=depth.png

set TEXTURE=%~2
if "%TEXTURE%"=="" set TEXTURE=texture.png

set OUTPUT=%~3
if "%OUTPUT%"=="" set OUTPUT=output.glb

set RESOLUTION=%~4
if "%RESOLUTION%"=="" set RESOLUTION=512

set DEPTH_SCALE=%~5
if "%DEPTH_SCALE%"=="" set DEPTH_SCALE=0.5

REM ── Run ──────────────────────────────────────────────────────────────────────
echo [INFO] 生成開始...
echo        depth=%DEPTH%  texture=%TEXTURE%  output=%OUTPUT%
echo        resolution=%RESOLUTION%  depth_scale=%DEPTH_SCALE%

"%BLENDER%" --background --python "%~dp0generate_3d.py" -- ^
    --depth "%~dp0%DEPTH%" ^
    --texture "%~dp0%TEXTURE%" ^
    --output "%~dp0%OUTPUT%" ^
    --resolution %RESOLUTION% ^
    --depth_scale %DEPTH_SCALE%

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 生成に失敗しました。上のログを確認してください。
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [OK] 完了: %OUTPUT%
echo      gltf.report で確認: https://gltf.report/
pause
