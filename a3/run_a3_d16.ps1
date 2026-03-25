$proj = "C:\Users\minhc\workspace\csc490\Compos3D\a3"
$logDir = Join-Path $proj "log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Start-Job -ScriptBlock {
    param($proj, $logDir)

    Set-Location $proj
    Write-Output "PWD: $(Get-Location)"

    modal run part3/nanochat_modal.py::stage_pretrain_phase2 *>&1 |
        Out-File -FilePath (Join-Path $logDir "p3_d16_phase2.log") -Encoding utf8
} -ArgumentList $proj, $logDir

Start-Job -ScriptBlock {
    param($proj, $logDir)

    Set-Location $proj
    Write-Output "PWD: $(Get-Location)"

    modal run part3/nanochat_modal.py::stage_pretrain_baseline *>&1 |
        Out-File -FilePath (Join-Path $logDir "p3_d16_baseline.log") -Encoding utf8
} -ArgumentList $proj, $logDir

Get-Job | Wait-Job
Receive-Job -Keep