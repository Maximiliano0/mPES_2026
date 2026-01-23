<#
.SYNOPSIS
    Vigila las optimizaciones bayesianas activas en Windows.
    Cuando cada una termina, hace git add + commit + push a la rama actual.

.DESCRIPTION
    Equivalente Windows de watch_and_push.sh.
    Recibe el nombre del paquete y uno o mas PIDs como argumentos.
    Cada 30 segundos comprueba si cada PID sigue vivo.
    Cuando un PID termina, ejecuta git add -A, commit y push.

.PARAMETER Package
    Nombre del paquete (pes_ql, pes_dql, pes_dqn, pes_ac, pes_trf).

.PARAMETER Pids
    Uno o mas PIDs de procesos a vigilar.

.EXAMPLE
    Start-Process powershell -ArgumentList "-File utils\watch_and_push.ps1 pes_dqn 5972" -WindowStyle Hidden
#>
param(
    [Parameter(Mandatory)][string]$Package,
    [string]$LogFile = '',
    [Parameter(Mandatory, ValueFromRemainingArguments)][int[]]$Pids
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

# ── Self-logging (for detached execution without -RedirectStandard*) ─
if ($LogFile) {
    Start-Transcript -Path $LogFile -Append -Force | Out-Null
}

# ── Paths ────────────────────────────────────────────────────────
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Notify     = Join-Path $ProjectDir "utils\notify.py"
$Python     = Join-Path $ProjectDir "win_mpes_env\Scripts\python.exe"

# ── Resolve package ─────────────────────────────────────────────
$validPackages = @{
    'pes_ql'='pes_ql';   'bayesian'='pes_ql'; 'bay'='pes_ql'; '1'='pes_ql'
    'pes_dql'='pes_dql'; 'dql'='pes_dql';     'ql'='pes_dql'; '2'='pes_dql'
    'pes_dqn'='pes_dqn'; 'dqn'='pes_dqn';                     '3'='pes_dqn'
    'pes_ac'='pes_ac';   'ac'='pes_ac';  'a2c'='pes_ac'; 'actor-critic'='pes_ac'; '4'='pes_ac'
    'pes_trf'='pes_trf'; 'transformer'='pes_trf'; 'tr'='pes_trf';   '5'='pes_trf'
}

$PkgName = $validPackages[$Package]
if (-not $PkgName) {
    Write-Error "Paquete desconocido: '$Package'"
    exit 1
}

# ── Git branch ──────────────────────────────────────────────────
Push-Location $ProjectDir
$Branch = git branch --show-current
Pop-Location

Write-Host "[watch_and_push] Paquete: $PkgName"
Write-Host "[watch_and_push] Vigilando PIDs: $($Pids -join ', ')"
Write-Host "[watch_and_push] Proyecto: $ProjectDir"
Write-Host "[watch_and_push] Rama: $Branch"

# ── Commit + push function ──────────────────────────────────────
function Invoke-CommitPush {
    param([int]$Pid, [string]$Label)

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm"
    Write-Host "[watch_and_push] [$ts] PID $Pid ($Label) termino. Realizando commit + push..."

    Push-Location $ProjectDir
    git add -A
    $commitResult = git commit -m "auto: Optimizacion $Label completada ($ts)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[watch_and_push] Sin cambios para commit (PID $Pid)"
        Pop-Location
        return
    }

    $pushResult = git push origin $Branch 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[watch_and_push] [$ts] Push completado para $Label"
        $body = "La optimizacion '$Label' (PID $Pid) termino correctamente.`nCommit y push a rama '$Branch' completados ($ts).`nProyecto: $ProjectDir"
        if (Test-Path $Notify) {
            & $Python $Notify "[$PkgName] Optimizacion completada - push realizado" $body 2>$null
        }
    } else {
        Write-Host "[watch_and_push] [$ts] ERROR en push para $Label"
        Write-Host $pushResult
        $body = "Error al hacer push de la optimizacion '$Label' (PID $Pid).`nRama: $Branch`nTimestamp: $ts`nProyecto: $ProjectDir"
        if (Test-Path $Notify) {
            & $Python $Notify "[$PkgName] ERROR en git push" $body 2>$null
        }
    }
    Pop-Location
}

# ── Label each PID ──────────────────────────────────────────────
$PidLabels = @{}
$Done = @{}
foreach ($p in $Pids) {
    try {
        $proc = Get-Process -Id $p -ErrorAction Stop
        $PidLabels[$p] = "$PkgName (PID $p)"
    } catch {
        $PidLabels[$p] = "PID_$p"
    }
    $Done[$p] = $false
    Write-Host "[watch_and_push] PID $p -> $($PidLabels[$p])"
}

# ── Watch loop ──────────────────────────────────────────────────
while ($true) {
    $allDone = $true
    foreach ($p in $Pids) {
        if ($Done[$p]) { continue }

        $alive = $false
        try {
            $proc = Get-Process -Id $p -ErrorAction Stop
            $alive = -not $proc.HasExited
        } catch {
            $alive = $false
        }

        if (-not $alive) {
            Invoke-CommitPush -Pid $p -Label $PidLabels[$p]
            $Done[$p] = $true
        } else {
            $allDone = $false
        }
    }

    if ($allDone) {
        Write-Host "[watch_and_push] Todas las optimizaciones terminaron. Saliendo."
        break
    }

    Start-Sleep -Seconds 30
}

# ── Detener transcript ──────────────────────────────────────────
if ($LogFile) {
    Stop-Transcript | Out-Null
}
