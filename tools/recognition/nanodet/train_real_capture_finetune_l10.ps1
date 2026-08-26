param(
    [string]$RemoteHost = "bugrat@192.168.11.22",
    [string]$RemoteRoot = "/srv/bugrat/data-lv/mjtensu",
    [int]$LayoutCount = 10,
    [int]$Seed = 42,
    [int]$RealRepeat = 20,
    [int]$BaseReplayImages = 512
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($LayoutCount -ne 10) {
    throw "This config and run name are fixed to layouts 1-10. LayoutCount must be 10."
}

function Assert-ExternalCommand([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "py"
}

$Builder = Join-Path $RepositoryRoot "tools\recognition\build_nanodet_capture_finetune_dataset.py"
$DatasetDirectory = Join-Path $RepositoryRoot ".local\recognition\nanodet_capture_finetune_dataset"
$CaptureComposites = Join-Path $RepositoryRoot ".local\recognition\capture_dataset\composites"
$ConfigName = "e1_nanodet_plus_m_320_real_capture_ft10_l10.yml"
$ConfigPath = Join-Path $PSScriptRoot "configs\$ConfigName"

foreach ($RequiredPath in @($Builder, $CaptureComposites, $ConfigPath)) {
    if (-not (Test-Path $RequiredPath)) {
        throw "Required path does not exist: $RequiredPath"
    }
}

Write-Host "=== Building real-capture fine-tune dataset ==="
& $Python $Builder `
    --layout-count $LayoutCount `
    --seed $Seed `
    --real-repeat $RealRepeat `
    --base-replay-images $BaseReplayImages
Assert-ExternalCommand "Fine-tune dataset build"

$RemoteDatasetParent = "$RemoteRoot/.local/recognition"
$RemoteCaptureRoot = "$RemoteRoot/.local/recognition/capture_dataset"
$RemoteConfigRoot = "$RemoteRoot/nanodet/experiment-configs"
$RemoteNanoDetRoot = "$RemoteRoot/nanodet/nanodet"
$RemoteCheckpoint = "$RemoteRoot/.local/recognition/nanodet_runs/E1_plus_m_320_composite_augmented_amp40_seed42/model_best/nanodet_model_best.pth"
$RemoteConfig = "$RemoteConfigRoot/$ConfigName"

Write-Host "=== Preparing remote directories and checking source model ==="
$PrepareCommand = @"
set -eu
mkdir -p '$RemoteDatasetParent' '$RemoteCaptureRoot' '$RemoteConfigRoot'
test -f '$RemoteCheckpoint'
"@
& ssh $RemoteHost $PrepareCommand
Assert-ExternalCommand "Remote preparation"

Write-Host "=== Copying fine-tune annotations ==="
& scp -r $DatasetDirectory "${RemoteHost}:$RemoteDatasetParent/"
Assert-ExternalCommand "Fine-tune dataset scp"

Write-Host "=== Copying captured composite images ==="
& scp -r $CaptureComposites "${RemoteHost}:$RemoteCaptureRoot/"
Assert-ExternalCommand "Capture composites scp"

Write-Host "=== Copying NanoDet config ==="
& scp $ConfigPath "${RemoteHost}:$RemoteConfigRoot/"
Assert-ExternalCommand "Fine-tune config scp"

Write-Host "=== Starting NanoDet real-capture fine-tune ==="
$TrainCommand = @"
set -eu
cd '$RemoteNanoDetRoot'
.venv/bin/python tools/train.py '$RemoteConfig' --seed $Seed
"@
& ssh $RemoteHost $TrainCommand
Assert-ExternalCommand "Remote fine-tune"

Write-Host ""
Write-Host "Fine-tune completed."
Write-Host "Remote run: $RemoteRoot/.local/recognition/nanodet_runs/E1_plus_m_320_real_capture_ft10_l10_seed42"
