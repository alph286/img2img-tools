@echo off
echo ===================================
echo  Launcher img2img-tools
echo ===================================
echo.

REM Verifica se Python è installato
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRORE: Python non è installato o non è nel PATH di sistema.
    echo Scarica e installa Python da https://www.python.org/downloads/
    echo Assicurati di selezionare "Aggiungi Python al PATH" durante l'installazione.
    echo.
    pause
    exit /b 1
)

REM Esegui il launcher Python
echo Avvio del launcher...
python launcher.py

REM Se il launcher termina con un errore, mostra un messaggio
if %errorlevel% neq 0 (
    echo.
    echo Si è verificato un errore durante l'esecuzione del launcher.
    echo.
    pause
)

exit /b 0