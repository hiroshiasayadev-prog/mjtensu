param(
    [string]$RemoteHost = "bugrat@192.168.11.22",
    [string]$RemoteRoot = "/srv/bugrat/data-lv/mjtensu",
    [double]$OperatingThreshold = 0.50
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

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

$RunName = "E1_plus_m_320_composite_augmented_amp40_seed42"
$ConfigName = "e1_nanodet_plus_m_320_composite_augmented_amp40.yml"
$RemoteNanoDetRoot = "$RemoteRoot/nanodet/nanodet"
$RemoteConfig = "$RemoteRoot/nanodet/experiment-configs/$ConfigName"
$RemoteCheckpoint = "$RemoteRoot/.local/recognition/nanodet_runs/$RunName/model_best/model_best.ckpt"
$RemoteOnnx = "$RemoteRoot/.local/recognition/nanodet_runs/$RunName/model_best/nanodet-plus-m-320-composite-augmented.onnx"

$LocalRunDirectory = Join-Path $RepositoryRoot ".local\recognition\nanodet_runs\$RunName"
$LocalOnnx = Join-Path $LocalRunDirectory "model_best\nanodet-plus-m-320-composite-augmented.onnx"
$Annotations = Join-Path $RepositoryRoot ".local\recognition\nanodet_composite_augmented_dataset\annotations\instances_composite_val.json"
$OldModel = Join-Path $RepositoryRoot "tools\recognition\pwa_detector_probe\public\models\nanodet-plus-m-320.onnx"
$EvaluationRoot = Join-Path $RepositoryRoot ".local\recognition\composite_augmented_heldout_eval"
$NewOutput = Join-Path $EvaluationRoot "new_model"
$OldOutput = Join-Path $EvaluationRoot "old_model"
$EvaluateScript = Join-Path $PSScriptRoot "evaluate_composite_onnx.py"
$SweepScript = Join-Path $PSScriptRoot "sweep_composite_thresholds.py"

foreach ($RequiredPath in @($Annotations, $OldModel, $EvaluateScript, $SweepScript)) {
    if (-not (Test-Path $RequiredPath)) {
        throw "Required path does not exist: $RequiredPath"
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $LocalOnnx) | Out-Null

$ExportCommand = @"
cd '$RemoteNanoDetRoot' && .venv/bin/python tools/export_onnx.py \
  --cfg_path '$RemoteConfig' \
  --model_path '$RemoteCheckpoint' \
  --out_path '$RemoteOnnx' \
  --input_shape 320,320
"@

Write-Host "=== Exporting the new ONNX model on the server ==="
& ssh $RemoteHost $ExportCommand
Assert-ExternalCommand "Remote ONNX export"

Write-Host "=== Copying the new ONNX model to Windows ==="
$RemoteSource = "${RemoteHost}:$RemoteOnnx"
& scp $RemoteSource $LocalOnnx
Assert-ExternalCommand "ONNX scp"

foreach ($OutputDirectory in @($NewOutput, $OldOutput)) {
    if (Test-Path $OutputDirectory) {
        Remove-Item -Recurse -Force $OutputDirectory
    }
}

Write-Host "=== Evaluating the new model on held-out composite validation ==="
& $Python $EvaluateScript `
    --model $LocalOnnx `
    --annotations $Annotations `
    --image-root $RepositoryRoot `
    --output-directory $NewOutput `
    --candidate-threshold 0.001 `
    --operating-threshold $OperatingThreshold `
    --overlay-limit 71
Assert-ExternalCommand "New-model evaluation"

Write-Host "=== Evaluating the old model on the same held-out validation ==="
& $Python $EvaluateScript `
    --model $OldModel `
    --annotations $Annotations `
    --image-root $RepositoryRoot `
    --output-directory $OldOutput `
    --candidate-threshold 0.001 `
    --operating-threshold $OperatingThreshold `
    --overlay-limit 71
Assert-ExternalCommand "Old-model evaluation"

Write-Host "=== Sweeping thresholds for the new model ==="
& $Python $SweepScript `
    --annotations $Annotations `
    --predictions (Join-Path $NewOutput "predictions.json") `
    --output-directory $NewOutput
Assert-ExternalCommand "New-model threshold sweep"

Write-Host "=== Sweeping thresholds for the old model ==="
& $Python $SweepScript `
    --annotations $Annotations `
    --predictions (Join-Path $OldOutput "predictions.json") `
    --output-directory $OldOutput
Assert-ExternalCommand "Old-model threshold sweep"

$NewSweep = Get-Content (Join-Path $NewOutput "threshold_sweep.json") -Raw | ConvertFrom-Json
$OldSweep = Get-Content (Join-Path $OldOutput "threshold_sweep.json") -Raw | ConvertFrom-Json
$NewAtOperating = $NewSweep.results | Where-Object { [math]::Abs([double]$_.threshold - $OperatingThreshold) -lt 0.000001 }
$OldAtOperating = $OldSweep.results | Where-Object { [math]::Abs([double]$_.threshold - $OperatingThreshold) -lt 0.000001 }

Write-Host ""
Write-Host "=== Held-out composite comparison ==="
Write-Host ("images={0}, ground_truths={1}" -f $NewSweep.image_count, $NewSweep.ground_truth_count)
Write-Host ("threshold={0:N2}" -f $OperatingThreshold)
Write-Host ("old: precision={0:N4} recall={1:N4} F1={2:N4} FP={3} FN={4} FP/image={5:N3} clean={6}" -f `
    $OldAtOperating.precision, $OldAtOperating.recall, $OldAtOperating.f1, `
    $OldAtOperating.false_positive_count, $OldAtOperating.false_negative_count, `
    $OldAtOperating.false_positives_per_image, $OldAtOperating.images_with_no_errors)
Write-Host ("new: precision={0:N4} recall={1:N4} F1={2:N4} FP={3} FN={4} FP/image={5:N3} clean={6}" -f `
    $NewAtOperating.precision, $NewAtOperating.recall, $NewAtOperating.f1, `
    $NewAtOperating.false_positive_count, $NewAtOperating.false_negative_count, `
    $NewAtOperating.false_positives_per_image, $NewAtOperating.images_with_no_errors)
Write-Host ("old best F1: threshold={0:N2} F1={1:N4}" -f $OldSweep.best_f1.threshold, $OldSweep.best_f1.f1)
Write-Host ("new best F1: threshold={0:N2} F1={1:N4}" -f $NewSweep.best_f1.threshold, $NewSweep.best_f1.f1)
Write-Host ""
Write-Host "New overlays: $NewOutput\overlays"
Write-Host "Old overlays: $OldOutput\overlays"
