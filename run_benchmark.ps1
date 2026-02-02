$REPO_URL = "https://github.com/MattiaManneschi/N-gram-Histograms.git"
$REPO_DIR = "N-gram-Histograms"
$NTHREADS = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors


$NGRAM_SIZE = 2
$TEST_MODE = "THREAD"
$WORK_DIR = ""


function Print-Header
{
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "        N-GRAM HISTOGRAM BENCHMARK RUNNER (Windows)                   " -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Print-Step
{
    param([string]$message)
    Write-Host "[STEP] " -ForegroundColor Blue -NoNewline
    Write-Host $message
}

function Print-Success
{
    param([string]$message)
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host $message
}

function Print-Warning
{
    param([string]$message)
    Write-Host "[WARNING] " -ForegroundColor Yellow -NoNewline
    Write-Host $message
}

function Print-Error
{
    param([string]$message)
    Write-Host "[ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $message
}

function Get-Timestamp
{
    return Get-Date -Format "yyyyMMdd_HHmmss"
}

function Wait-ForExit
{
    Write-Host ""
    Write-Host "Premi un tasto per chiudere..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}





function Setup-WorkingDirectory
{
    Print-Step "Rilevamento directory di lavoro..."


    if ((Test-Path "src") -and (Test-Path "Makefile") -and (Test-Path "data\Texts"))
    {
        Print-Success "Gia' dentro il repository"
        $script:WORK_DIR = (Get-Location).Path
    }

    elseif ((Test-Path "$REPO_DIR\src") -and (Test-Path "$REPO_DIR\Makefile"))
    {
        Print-Success "Repository trovato in .\$REPO_DIR"
        Set-Location $REPO_DIR
        $script:WORK_DIR = (Get-Location).Path
    }
    else
    {

        Print-Step "Repository non trovato, clonazione..."
        git clone $REPO_URL
        Set-Location $REPO_DIR
        $script:WORK_DIR = (Get-Location).Path
        Print-Success "Repository clonato"
    }

    Write-Host "Working directory: $WORK_DIR"
}





function Install-Dependencies
{
    Print-Step "Controllo dipendenze..."

    $hasWinget = Get-Command winget -ErrorAction SilentlyContinue
    $hasChoco = Get-Command choco -ErrorAction SilentlyContinue


    if (-not (Get-Command git -ErrorAction SilentlyContinue))
    {
        Print-Step "Installazione Git..."
        if ($hasWinget)
        {
            winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
        }
        elseif ($hasChoco)
        {
            choco install git -y
        }
        else
        {
            Print-Error "Git non trovato. Installa Git manualmente da https://git-scm.com/"
            Wait-ForExit
            exit 1
        }
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    }
    Print-Success "Git OK"


    if (-not (Get-Command python -ErrorAction SilentlyContinue))
    {
        Print-Step "Installazione Python..."
        if ($hasWinget)
        {
            winget install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements
        }
        elseif ($hasChoco)
        {
            choco install python -y
        }
        else
        {
            Print-Error "Python non trovato. Installa Python manualmente da https://python.org/"
            Wait-ForExit
            exit 1
        }
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    }
    Print-Success "Python OK"


    if (-not (Get-Command g++ -ErrorAction SilentlyContinue))
    {
        Print-Step "Installazione MinGW (g++)..."
        if ($hasWinget)
        {
            winget install --id MSYS2.MSYS2 -e --source winget --accept-package-agreements --accept-source-agreements
            Print-Warning "MSYS2 installato. Esegui 'pacman -S mingw-w64-x86_64-gcc' in MSYS2 terminal"
        }
        elseif ($hasChoco)
        {
            choco install mingw -y
        }
        else
        {
            Print-Error "g++ non trovato. Installa MinGW manualmente"
            Wait-ForExit
            exit 1
        }
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    }
    Print-Success "g++ OK"


    if (-not (Get-Command make -ErrorAction SilentlyContinue))
    {
        if (-not (Get-Command mingw32-make -ErrorAction SilentlyContinue))
        {
            Print-Warning "Make non trovato. Compilazione manuale."
        }
    }


    Print-Step "Installazione pacchetti Python..."
    python -m pip install --upgrade pip 2> $null
    python -m pip install matplotlib pandas numpy 2> $null
    Print-Success "Pacchetti Python OK"
}





function Compile-Project
{
    Print-Step "Compilazione progetto..."


    if (-not (Test-Path "bin"))
    {
        New-Item -ItemType Directory -Path "bin" | Out-Null
    }


    $makeCmd = $null
    if (Get-Command make -ErrorAction SilentlyContinue)
    {
        $makeCmd = "make"
    }
    elseif (Get-Command mingw32-make -ErrorAction SilentlyContinue)
    {
        $makeCmd = "mingw32-make"
    }

    if ($makeCmd)
    {
        & $makeCmd clean 2> $null
        & $makeCmd all
    }
    else
    {
        Print-Step "Compilazione manuale con g++..."
        $srcFiles = Get-ChildItem -Path "src" -Filter "*.cpp" | ForEach-Object { $_.FullName }
        $srcFilesStr = $srcFiles -join " "

        $compileCmd = "g++ -std=c++17 -Wall -O3 -fopenmp $srcFilesStr -o bin/ngram_analyzer_par.exe"
        Invoke-Expression $compileCmd
    }

    Print-Success "Compilazione completata"
}





function Select-NgramSize
{
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "  Seleziona la dimensione degli N-grammi:" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1) 2-grammi (bigrammi)"
    Write-Host "  2) 3-grammi (trigrammi)"
    Write-Host ""
    $choice = Read-Host "Scelta [1-2]"

    switch ($choice)
    {
        "1" {
            $script:NGRAM_SIZE = 2
        }
        "2" {
            $script:NGRAM_SIZE = 3
        }
        default {
            $script:NGRAM_SIZE = 2
        }
    }

    Print-Success "Selezionato: $NGRAM_SIZE-grammi"
}

function Select-TestMode
{
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "  Seleziona il tipo di test:" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1) Thread Scaling  (workload fisso, thread variabili)"
    Write-Host "  2) Workload Scaling (thread fissi, workload variabile)"
    Write-Host "  3) Entrambi"
    Write-Host ""
    $choice = Read-Host "Scelta [1-3]"

    switch ($choice)
    {
        "1" {
            $script:TEST_MODE = "THREAD"
        }
        "2" {
            $script:TEST_MODE = "WORKLOAD"
        }
        "3" {
            $script:TEST_MODE = "BOTH"
        }
        default {
            $script:TEST_MODE = "THREAD"
        }
    }

    Print-Success "Selezionato: $TEST_MODE"
}

function Select-Threads
{
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "  Numero di thread (rilevati: $NTHREADS core):" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
    $input_threads = Read-Host "Numero thread [$NTHREADS]"

    if ($input_threads)
    {
        $script:NTHREADS = [int]$input_threads
    }

    Print-Success "Selezionato: $NTHREADS thread"
}





function Run-Test
{
    param([string]$mode)

    $timestamp = Get-Timestamp
    $outputDir = "results_${NGRAM_SIZE}gram\$($mode.ToLower() )_$timestamp"

    Print-Step "Esecuzione test $mode per $NGRAM_SIZE-grammi..."
    Print-Step "Output: $outputDir"

    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null


    $exePath = "bin\ngram_analyzer_par.exe"
    if (-not (Test-Path $exePath))
    {
        $exePath = "bin\ngram_analyzer_par"
    }

    & $exePath "data\Texts" $NGRAM_SIZE $NTHREADS $mode


    python plot_results.py $NGRAM_SIZE $NTHREADS


    if (Test-Path "results")
    {
        Copy-Item -Path "results\*" -Destination $outputDir -Recurse -Force -ErrorAction SilentlyContinue
    }


    $cpuInfo = (Get-CimInstance Win32_Processor).Name
    $osInfo = (Get-CimInstance Win32_OperatingSystem).Caption

    $testInfo = @"
======================================================================
  N-GRAM BENCHMARK TEST INFO
======================================================================

Data/Ora:       $( Get-Date )
N-gram size:    $NGRAM_SIZE
Test mode:      $mode
Threads:        $NTHREADS
Sistema:        $osInfo
CPU:            $cpuInfo
Core totali:    $( (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors )

======================================================================
"@

    $testInfo | Out-File -FilePath "$outputDir\test_info.txt" -Encoding UTF8

    Print-Success "Risultati salvati in: $outputDir"
}





function Main
{
    Print-Header

    $startDir = Get-Location

    try
    {

        Setup-WorkingDirectory
        Install-Dependencies
        Compile-Project


        Select-NgramSize
        Select-TestMode
        Select-Threads


        Write-Host ""
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host "  AVVIO BENCHMARK" -ForegroundColor Cyan
        Write-Host "======================================================================" -ForegroundColor Cyan
        Write-Host ""

        switch ($TEST_MODE)
        {
            "THREAD"   {
                Run-Test "THREAD"
            }
            "WORKLOAD" {
                Run-Test "WORKLOAD"
            }
            "BOTH"     {
                Run-Test "THREAD"; Run-Test "WORKLOAD"
            }
        }


        Write-Host ""
        Write-Host "======================================================================" -ForegroundColor Green
        Write-Host "  BENCHMARK COMPLETATO!" -ForegroundColor Green
        Write-Host "======================================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Risultati in: results_${NGRAM_SIZE}gram\"
        Get-ChildItem -Path "results_${NGRAM_SIZE}gram" -ErrorAction SilentlyContinue | Format-Table Name, LastWriteTime

    }
    catch
    {
        Print-Error "Errore: $_"
    }
    finally
    {
        Set-Location $startDir

        Wait-ForExit
    }
}


Main