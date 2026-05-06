@echo off
setlocal

REM Auto-detect Blender installation
set BLENDER=
for %%P in (blender.exe) do set BLENDER=%%~$PATH:P

if "%BLENDER%"=="" (
    for %%D in (
        "C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
        "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
        "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
        "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"
        "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"
        "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"
    ) do (
        if exist %%D set BLENDER=%%~D
    )
)

if "%BLENDER%"=="" (
    echo ERROR: Blender not found.
    echo Please install Blender: https://www.blender.org/download/
    pause
    exit /b 1
)

echo Blender: %BLENDER%
echo Generating 3D model...

set SCRIPT=%~dp0generate_3d.py
set DEPTH=%~dp0depth.png
set TEXTURE=%~dp0texture.png
set OUTPUT=%~dp0output.glb

"%BLENDER%" --background --python "%SCRIPT%" -- --depth "%DEPTH%" --texture "%TEXTURE%" --output "%OUTPUT%" --resolution 512 --depth_scale 0.5

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Generation failed. Check the log above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Done: output.glb
echo Preview at: https://gltf.report/
pause
