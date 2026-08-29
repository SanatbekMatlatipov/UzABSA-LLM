# =============================================================================
# UzABSA-LLM: status board for the rerun_v2 pipeline (2-GPU lane layout)
# =============================================================================
# Read-only. Safe to run at any time; it never touches the running jobs.
#
#   .\scripts\rerun_status.ps1          one snapshot
#   .\scripts\rerun_status.ps1 -Watch   refresh every 30s until Ctrl+C
#
# After the 2026-08-29 crash and restart, work runs in three sequential lane
# scripts (lane_a/b/c.log) rather than one log per model — each lane runs
# several stages back-to-back on a single GPU to respect the 2-GPU power cap.
# This board greps whichever lane log currently mentions a given model, so it
# stays correct as a lane moves from one stage to the next.
# =============================================================================
param([switch]$Watch, [int]$Every = 30)

$Root = Split-Path $PSScriptRoot -Parent
$Lanes = 'lane_a', 'lane_b', 'lane_c'

function Get-LastMatch($file, $pattern) {
    if (-not (Test-Path $file)) { return $null }
    $t = (Get-Content $file -Raw -ErrorAction SilentlyContinue) -replace "`r", "`n"
    if (-not $t) { return $null }
    $m = [regex]::Matches($t, $pattern)
    if ($m.Count) { return $m[$m.Count - 1].Value }
    return $null
}

function Get-LaneTail($file, $n = 1) {
    # tqdm's \r progress ticks are each their own "line" to Get-Content, so a
    # tail of even a few hundred lines can be entirely progress-bar noise once
    # training has run a while. Search the whole raw text instead.
    if (-not (Test-Path $file)) { return @() }
    $t = (Get-Content $file -Raw -ErrorAction SilentlyContinue) -replace "`r", "`n"
    if (-not $t) { return @() }
    $m = [regex]::Matches($t, '=== [A-Za-z0-9].*? ===')
    if ($m.Count -eq 0) { return @() }
    $start = [math]::Max(0, $m.Count - $n)
    return ($m | Select-Object -Skip $start) | ForEach-Object { $_.Value }
}

function Show-Status {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host " UzABSA rerun_v2   $(Get-Date -Format 'HH:mm:ss')   (2-GPU cap since the 08-29 restart)"
    Write-Host ("=" * 70)

    Write-Host "`n[Lanes] each runs its stages back-to-back on ONE gpu"
    foreach ($lane in $Lanes) {
        $log = "$Root\outputs\rerun_v2\logs\$lane.log"
        if (-not (Test-Path $log)) {
            "    {0,-8} not started" -f $lane
            continue
        }
        $body = Get-Content $log -Raw -ErrorAction SilentlyContinue
        if (-not $body) {
            # 0-byte log: this lane's wait-for-a-free-GPU loop hasn't claimed one
            # yet (by design under the 2-GPU cap), not a stalled job.
            "    {0,-8} waiting for a free GPU (2-GPU cap)" -f $lane
            continue
        }
        $lastModified = (Get-Item $log).LastWriteTime
        $ageMin = [math]::Round(((Get-Date) - $lastModified).TotalMinutes, 1)
        $stage = Get-LaneTail $log 1
        # Training counts to 1000 steps, eval/zero-shot count to 609 examples.
        # Match both and take whichever appears last, or the bar stays pinned at
        # the finished training run's 1000/1000 for the rest of the lane.
        $prog = Get-LastMatch $log '\d+/(?:1000|609) \[[\d:]+<[\d:]+'
        $done = $body -match "LANE [A-Z] DONE"
        $failed = $body -match "FAILED|Traceback"

        $stageLabel = if ($stage) { ($stage -replace '=== ', '' -replace ' ===', '') } else { "starting..." }
        $flag = ""
        if ($done) { $flag = " [DONE]" }
        elseif ($failed -and -not $prog) { $flag = " [CHECK LOG - error seen]" }
        elseif ($ageMin -gt 5) { $flag = " [STALE - no log output in $ageMin min]" }

        "    {0,-8} {1,-30} {2}{3}" -f $lane, $stageLabel, $(if ($prog) { $prog } else { "" }), $flag
    }

    Write-Host "`n[Completed models] merged + evaluated"
    foreach ($m in 'qwen2.5-7b', 'llama3.1-8b', 'deepseek-7b') {
        $merged = Test-Path "$Root\outputs\rerun_v2\uzabsa_${m}_v2\merged_model\config.json"
        $ev = Get-ChildItem "$Root\outputs\rerun_v2\uzabsa_${m}_v2\eval_results*.json" -ErrorAction SilentlyContinue
        $zs = Get-ChildItem "$Root\outputs\rerun_v2\zeroshot\$m\eval_results*.json" -ErrorAction SilentlyContinue
        $s = if ($ev) { "trained + evaluated" } elseif ($merged) { "merged, eval pending" } else { "not yet merged" }
        $zsLabel = if ($zs) { "zero-shot done" } else { "zero-shot pending" }
        "    {0,-14} {1,-22} {2}" -f $m, $s, $zsLabel
    }

    Write-Host "`n[GPUs]"
    $rows = nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
    $totalW = 0
    foreach ($r in $rows) {
        "    GPU $r"
        if ($r -match '([\d.]+)\s*W\s*$') { $totalW += [double]$Matches[1] }
    }
    "    ---"
    "    total GPU power draw: {0:N0} W" -f $totalW

    Write-Host ""
}

if ($Watch) {
    while ($true) { Clear-Host; Show-Status; Start-Sleep -Seconds $Every }
} else {
    Show-Status
}
