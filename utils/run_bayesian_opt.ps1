<#
.SYNOPSIS
    Lanzar optimizacion Bayesiana en Windows.

.DESCRIPTION
    Equivalente Windows de run_bayesian_opt.sh.
    Lanza la optimizacion en segundo plano, configura la PC para no
    suspenderse y arranca el watcher que hace commit+push al terminar.

    Los procesos se ejecutan con Start-Process -WindowStyle Hidden,
    por lo que NO es necesario mantener ninguna terminal abierta.

.PARAMETER Package
    Alias o nombre del paquete destino.
    Valores: bayesian|bay|1, qlv2|ql|2, dqn|3, ac|a2c|4, transformer|tr|5

.PARAMETER NTrials
    Numero de trials de optimizacion (por defecto 30).

.PARAMETER ResumeDate
    (Opcional) Fecha YYYY-MM-DD para reanudar una corrida previa.

.EXAMPLE
    .\utils\run_bayesian_opt.ps1 dqn 110
    .\utils\run_bayesian_opt.ps1 ac 100
    .\utils\run_bayesian_opt.ps1 bayesian 100 2026-02-12
#>
param(
    [Parameter(Mandatory, Position = 0)][string]$Package,
    [Parameter(Position = 1)][int]$NTrials = 30,
    [Parameter(Position = 2)][string]$ResumeDate = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Rutas base ───────────────────────────────────────────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$Python     = Join-Path $ProjectDir 'win_mpes_env\Scripts\python.exe'
$Watcher    = Join-Path $ScriptDir  'watch_and_push.ps1'

if (-not (Test-Path $Python)) {
    Write-Error "Python no encontrado: $Python"
    exit 1
}

# ── Resolver paquete ────────────────────────────────────────────
$pkgMap = @{
    'pes_bline'='pes_bline'; 'bayesian'='pes_bline'; 'bay'='pes_bline'; '1'='pes_bline'
    'pes_qlv2'='pes_qlv2';  'qlv2'='pes_qlv2';      'ql'='pes_qlv2';  '2'='pes_qlv2'
    'pes_dqn'='pes_dqn';    'dqn'='pes_dqn';                           '3'='pes_dqn'
    'pes_ac'='pes_ac';      'ac'='pes_ac';  'a2c'='pes_ac'; 'actor-critic'='pes_ac'; '4'='pes_ac'
    'pes_trf'='pes_trf';    'transformer'='pes_trf'; 'tr'='pes_trf';   '5'='pes_trf'
}

$PkgName = $pkgMap[$Package]
if (-not $PkgName) {
    Write-Error "Paquete desconocido: '$Package'. Opciones: bayesian, qlv2, dqn, ac, transformer"
    exit 1
}

# ── Resolver modulo de optimizacion ─────────────────────────────
$modMap = @{
    'pes_bline' = 'pes_bline.ext.optimize_rl'
    'pes_qlv2'  = 'pes_qlv2.ext.optimize_rl'
    'pes_dqn'   = 'pes_dqn.ext.optimize_dqn'
    'pes_ac'    = 'pes_ac.ext.optimize_ac'
    'pes_trf'   = 'pes_trf.ext.optimize_tr'
}
$OptModule = $modMap[$PkgName]

# ── Preparar directorio de logs ─────────────────────────────────
$LogDir = Join-Path $ProjectDir "$PkgName\inputs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$LogSuffix = ''
if ($ResumeDate) { $LogSuffix = "_resume_$ResumeDate" }
$LogFile    = Join-Path $LogDir "bayesian_opt${LogSuffix}.log"
$ErrFile    = Join-Path $LogDir "bayesian_opt${LogSuffix}_err.log"
$WatcherLog = Join-Path $LogDir "watcher${LogSuffix}.log"
$WatcherErr = Join-Path $LogDir "watcher${LogSuffix}_err.log"

# ── Construir argumentos ────────────────────────────────────────
# Use utils/run_module.py wrapper so that stdout/stderr are redirected at the
# Python level (line-buffered) instead of via Start-Process -Redirect*, which
# holds an exclusive file lock that prevents monitoring the log in real-time.
$RunModule = Join-Path $ScriptDir 'run_module.py'
$pyArgs = @($RunModule, $OptModule, $LogFile, $ErrFile, "$NTrials")
if ($ResumeDate) { $pyArgs += @('--resume', $ResumeDate) }

# ── Evitar suspension / hibernacion / apagado de pantalla ───────
Write-Host "`n  Configurando energia: sin suspension ni hibernacion..."
powercfg /change standby-timeout-ac 0   2>$null
powercfg /change standby-timeout-dc 0   2>$null
powercfg /change hibernate-timeout-ac 0 2>$null
powercfg /change hibernate-timeout-dc 0 2>$null
powercfg /change monitor-timeout-ac 0   2>$null
powercfg /change monitor-timeout-dc 0   2>$null

# ── Configurar entorno para el proceso hijo ─────────────────────
$env:VIRTUAL_ENV       = Join-Path $ProjectDir 'win_mpes_env'
$env:PYTHONIOENCODING  = 'utf-8'
$env:TF_ENABLE_ONEDNN_OPTS = '0'

# ── Lanzar optimizacion en segundo plano ────────────────────────
# No -RedirectStandard* here — the run_module.py wrapper handles
# file-level stdout/stderr redirection (line-buffered, real-time readable).
$optProc = Start-Process -FilePath $Python `
    -ArgumentList $pyArgs `
    -WorkingDirectory $ProjectDir `
    -PassThru -WindowStyle Hidden

$OptPid = $optProc.Id
Write-Host "  Optimizacion lanzada   PID=$OptPid  trials=$NTrials"
Write-Host "  Log: $PkgName\inputs\bayesian_opt${LogSuffix}.log"

# ── Lanzar watcher (commit+push al terminar) ────────────────────
$watcherPid = $null
if (Test-Path $Watcher) {
    $watchProc = Start-Process -FilePath powershell -ArgumentList @('-ExecutionPolicy', 'Bypass', '-File', $Watcher, $PkgName, "$OptPid") -WorkingDirectory $ProjectDir -RedirectStandardOutput $WatcherLog -RedirectStandardError $WatcherErr -PassThru -WindowStyle Hidden
    $watcherPid = $watchProc.Id
    Write-Host "  Watcher lanzado        PID=$watcherPid"
} else {
    Write-Host "  AVISO: watch_and_push.ps1 no encontrado - sin watcher"
}

# ── Resumen ─────────────────────────────────────────────────────
Write-Host "`n  =========================================="
Write-Host "  Optimizacion lanzada"
Write-Host "    Paquete:     $PkgName"
Write-Host "    Modulo:      $OptModule"
Write-Host "    Trials:      $NTrials"
Write-Host "    PID:         $OptPid"
if ($watcherPid) {
    Write-Host "    Watcher PID: $watcherPid"
}
Write-Host "    Log:         $PkgName\inputs\bayesian_opt${LogSuffix}.log"
Write-Host "    Watcher log: $PkgName\inputs\watcher${LogSuffix}.log"
Write-Host "  =========================================="

Write-Host "`n  Comandos utiles:"
Write-Host "    Progreso:    Select-String 'Trial' $LogFile | Select-Object -Last 10"
Write-Host "    Tiempo real: Get-Content $LogFile -Wait -Tail 20"
Write-Host "    Vivo?:       Get-Process -Id $OptPid -ErrorAction SilentlyContinue"
Write-Host "    Errores:     Get-Content $ErrFile -Tail 20"
Write-Host ""

Write-Host "  Los procesos corren en segundo plano. Puede cerrar esta terminal."
Write-Host ""
