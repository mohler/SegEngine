@echo off
SET MAVEN_EXE=C:\tools\apache-maven\bin\mvn.cmd
SET MAIN_CLASS=org.wormguides.segmentation.SegmentationTest

echo ====================================================
echo   CYTOSHOW SEGMENTATION ENGINE - V100 LAUNCHER
echo ====================================================

:: 1. Verify Maven exists at your known path
if not exist "%MAVEN_EXE%" (
    echo [ERROR] Maven not found at %MAVEN_EXE%
    pause
    exit /b
)

:: 2. Compile with the 'No-Processor' fix for Java 8
echo [INFO] Compiling (Bypassing Java 11 Annotation Processors)...
call "%MAVEN_EXE%" compiler:compile -Dmaven.compiler.proc=none
if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed.
    pause
    exit /b
)

:: 3. Execute the Segmentation Test
echo [INFO] Executing %MAIN_CLASS% on Tesla V100...
echo [INFO] Ensure resnet101_v100.pt is in C:\models\
call "%MAVEN_EXE%" exec:java -Dexec.mainClass="%MAIN_CLASS%"

echo ====================================================
echo   PROCESS COMPLETE
echo ====================================================
pause
