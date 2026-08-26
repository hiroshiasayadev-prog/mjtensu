param(
    [string]$RemoteHost = "bugrat@192.168.11.22",
    [string]$RemoteRoot = "/srv/bugrat/data-lv/mjtensu",
    [double]$OperatingThreshold = 0.30
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-ExternalCommand([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Evaluation(
    [string]$Label,
    [string]$Model,
    [string]$Annotations,
    [string]$OutputDirectory,
    [int]$OverlayLimit
) {
    Write-Host "=== Evaluating $Label ==="
    if (Test-Path $OutputDirectory) {
        Remove-Item -Recurse -Force $OutputDirectory
    }
    & $script:Python $script:EvaluateScript `
        --model $Model `
        --annotations $Annotations `
        --image-root $script:RepositoryRoot `
        --output-directory $OutputDirectory `
        --candidate-threshold 0.001 `
        --operating-threshold $OperatingThreshold `
        --overlay-limit $OverlayLimit
    Assert-ExternalCommand "$Label evaluation"

    & $script:Python $script:SweepScript `
        --annotations $Annotations `
        --predictions (Join-Path $OutputDirectory "predictions.json") `
        --output-directory $OutputDirectory
    Assert-ExternalCommand "$Label threshold sweep"
}

function Get-ThresholdResult([object]$Sweep, [double]$Threshold) {
    return $Sweep.results | Where-Object {
        [math]::Abs([double]$_.threshold - $Threshold) -lt 0.000001
    } | Select-Object -First 1
}

function Write-Comparison(
    [string]$Title,
    [string]$OldDirectory,
    [string]$NewDirectory
) {
    $OldSweep = Get-Content (Join-Path $OldDirectory "threshold_sweep.json") -Raw | ConvertFrom-Json
    $NewSweep = Get-Content (Join-Path $NewDirectory "threshold_sweep.json") -Raw | ConvertFrom-Json
    $OldAtOperating = Get-ThresholdResult $OldSweep $OperatingThreshold
    $NewAtOperating = Get-ThresholdResult $NewSweep $OperatingThreshold

    Write-Host ""
    Write-Host "=== $Title ==="
    Write-Host ("images={0}, ground_truths={1}" -f $NewSweep.image_count, $NewSweep.ground_truth_count)
    Write-Host ("threshold={0:N2}" -f $OperatingThreshold)
    Write-Host ("old: precision={0:N4} recall={1:N4} F1={2:N4} FP={3} FN={4} clean={5}" -f `
        $OldAtOperating.precision, $OldAtOperating.recall, $OldAtOperating.f1, `
        $OldAtOperating.false_positive_count, $OldAtOperating.false_negative_count, `
        $OldAtOperating.images_with_no_errors)
    Write-Host ("new: precision={0:N4} recall={1:N4} F1={2:N4} FP={3} FN={4} clean={5}" -f `
        $NewAtOperating.precision, $NewAtOperating.recall, $NewAtOperating.f1, `
        $NewAtOperating.false_positive_count, $NewAtOperating.false_negative_count, `
        $NewAtOperating.images_with_no_errors)
    Write-Host ("old best F1: threshold={0:N2} F1={1:N4} precision={2:N4} recall={3:N4}" -f `
        $OldSweep.best_f1.threshold, $OldSweep.best_f1.f1, `
        $OldSweep.best_f1.precision, $OldSweep.best_f1.recall)
    Write-Host ("new best F1: threshold={0:N2} F1={1:N4} precision={2:N4} recall={3:N4}" -f `
        $NewSweep.best_f1.threshold, $NewSweep.best_f1.f1, `
        $NewSweep.best_f1.precision, $NewSweep.best_f1.recall)
    Write-Host "new overlays: $NewDirectory\overlays"
    Write-Host "old overlays: $OldDirectory\overlays"
}

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "py"
}

$RunName = "E1_plus_m_320_real_capture_ft10_l10_seed42"
$ConfigName = "e1_nanodet_plus_m_320_real_capture_ft10_l10.yml"
$RemoteNanoDetRoot = "$RemoteRoot/nanodet/nanodet"
$RemoteConfig = "$RemoteRoot/nanodet/experiment-configs/$ConfigName"
$RemoteCheckpoint = "$RemoteRoot/.local/recognition/nanodet_runs/$RunName/model_best/model_best.ckpt"
$RemoteOnnx = "$RemoteRoot/.local/recognition/nanodet_runs/$RunName/model_best/nanodet-plus-m-320-real-capture-ft10-l10.onnx"

$LocalRunDirectory = Join-Path $RepositoryRoot ".local\recognition\nanodet_runs\$RunName"
$LocalOnnx = Join-Path $LocalRunDirectory "model_best\nanodet-plus-m-320-real-capture-ft10-l10.onnx"
$OldModel = Join-Path $RepositoryRoot "tools\recognition\pwa_capture_dataset\public\models\nanodet-plus-m-320-composite-augmented.onnx"
$RealAnnotations = Join-Path $RepositoryRoot ".local\recognition\nanodet_capture_finetune_dataset\annotations\instances_real_val.json"
$CompositeAnnotations = Join-Path $RepositoryRoot ".local\recognition\nanodet_composite_augmented_dataset\annotations\instances_composite_val.json"
$EvaluationRoot = Join-Path $RepositoryRoot ".local\recognition\real_capture_finetune_eval_l10"
$EvaluateScript = Join-Path $PSScriptRoot "evaluate_composite_onnx.py"
$SweepScript = Join-Path $PSScriptRoot "sweep_composite_thresholds.py"

foreach ($RequiredPath in @(
    $OldModel,
    $RealAnnotations,
    $CompositeAnnotations,
    $EvaluateScript,
    $SweepScript
)) {
    if (-not (Test-Path $RequiredPath)) {
        throw "Required path does not exist: $RequiredPath"
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $LocalOnnx) | Out-Null

Write-Host "=== Exporting fine-tuned best checkpoint to ONNX on the server ==="
$ExportCommand = @"
set -eu
test -f '$RemoteCheckpoint'
test -f '$RemoteConfig'
cd '$RemoteNanoDetRoot'
.venv/bin/python tools/export_onnx.py \
  --cfg_path '$RemoteConfig' \
  --model_path '$RemoteCheckpoint' \
  --out_path '$RemoteOnnx' \
  --input_shape 320,320
"@
& ssh $RemoteHost $ExportCommand
Assert-ExternalCommand "Remote ONNX export"

Write-Host "=== Copying fine-tuned ONNX model to Windows ==="
& scp "${RemoteHost}:$RemoteOnnx" $LocalOnnx
Assert-ExternalCommand "Fine-tuned ONNX scp"

$RealOld = Join-Path $EvaluationRoot "real_val\old_model"
$RealNew = Join-Path $EvaluationRoot "real_val\new_model"
$CompositeOld = Join-Path $EvaluationRoot "composite_val\old_model"
$CompositeNew = Join-Path $EvaluationRoot "composite_val\new_model"

Invoke-Evaluation "old model / held-out real captures" $OldModel $RealAnnotations $RealOld 8
Invoke-Evaluation "new model / held-out real captures" $LocalOnnx $RealAnnotations $RealNew 8
Invoke-Evaluation "old model / composite regression validation" $OldModel $CompositeAnnotations $CompositeOld 30
Invoke-Evaluation "new model / composite regression validation" $LocalOnnx $CompositeAnnotations $CompositeNew 30

Write-Comparison "Held-out real capture comparison" $RealOld $RealNew
Write-Comparison "Composite regression comparison" $CompositeOld $CompositeNew

Write-Host ""
Write-Host "Fine-tuned ONNX: $LocalOnnx"
Write-Host "Evaluation root: $EvaluationRoot"
