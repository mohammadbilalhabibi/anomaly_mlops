<#
.SYNOPSIS
  Deploy the anomaly_mlops demo to Azure Container Apps (Consumption plan, scale to zero).

.DESCRIPTION
  Run from the project folder after `az login`:
    powershell -ExecutionPolicy Bypass -File scripts\deploy_azure.ps1             # first deploy or update
    powershell -ExecutionPolicy Bypass -File scripts\deploy_azure.ps1 -SkipBudget # without the budget step

  Prerequisites:
    - The image is built and pushed by GitHub Actions (.github/workflows/ci.yml) to
      ghcr.io/mohammadbilalhabibi/anomaly_mlops; the package must be public so Azure can pull it
      without credentials. The image contains no models or MVTec images.
    - Models and samples are in a PRIVATE Hugging Face repo (scripts/upload_assets_hf.py).
    - .env contains HF_TOKEN=<read-only token for that repo>. It is sent to Azure as a secret and
      never printed.

  Steps:
    1. Show the subscription that will be used.
    2. Register the resource providers Container Apps needs (once per subscription).
    3. Create the resource group in the allowed region.
    4. Create the Container Apps environment (Consumption only, no Log Analytics workspace).
    5. Create or update the app from deploy\containerapp.yaml (image, size, env vars, probes,
       scale 0..1, external HTTPS ingress on port 8000).
    6. Set the HF_TOKEN secret from .env (the value is never printed).
    7. Replace the running replica so it uses the secret.
    8. Print the public URL.
    9. Create a monthly budget for this resource group with e-mail alerts (skip with -SkipBudget).
#>
param(
    [string]$Location = "francecentral",
    [string]$ResourceGroup = "anomaly-rg",
    [string]$Environment = "anomaly-env",
    [string]$AppName = "anomaly-inspector",
    [string]$Image = "ghcr.io/mohammadbilalhabibi/anomaly_mlops:latest",
    [string]$HfAssetsRepo = "MohammadBilalHabibi12345/anomaly-mlops-assets",
    # Measured locally (Windows, all 15 INT8 models): 0.97 GB after loading, 2.2 GB peak
    # working set while scoring one image per category. 1.5 vCPU / 3 GiB is the smallest
    # Consumption size with headroom; check Azure's WorkingSetBytes metric after deploying.
    [string]$Cpu = "1.5",
    [string]$Memory = "3Gi",
    # ONNX Runtime otherwise sizes its thread pool from the host's cores, not the 1.5 vCPU quota.
    [int]$OrtThreads = 2,
    [int]$BudgetUsd = 10,
    [switch]$SkipBudget
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$az = (Get-Command az -ErrorAction SilentlyContinue).Source
if (-not $az) { $az = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" }

function Invoke-Az {
    $out = & $az @args
    if ($LASTEXITCODE -ne 0) { throw "az $($args[0..2] -join ' ') ... failed (exit code $LASTEXITCODE)" }
    return $out
}

function Find-Az {
    # Existence checks: "not found" is an expected answer here, not an error.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $out = & $az @args 2>$null } finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { return $null }
    return $out
}

# Read the token first so a missing value stops the script before anything is created.
$line = Get-Content (Join-Path $root ".env") -ErrorAction SilentlyContinue |
    Where-Object { $_ -match '^\s*HF_TOKEN\s*=' } | Select-Object -First 1
$hfToken = if ($line) { ($line -split '=', 2)[1].Trim().Trim('"').Trim("'") } else { "" }
if (-not $hfToken) { throw "HF_TOKEN is missing in .env (read-only token for $HfAssetsRepo)" }
Remove-Variable line

# 1. Which subscription? (az login must have been run)
Write-Host "[1/9] Subscription:" -ForegroundColor Cyan
$sub = Invoke-Az account show --query "{name:name, id:id, user:user.name}" -o json | ConvertFrom-Json
Write-Host "      $($sub.name) ($($sub.id))"

# 2. Resource providers (no cost; takes a minute the first time)
Write-Host "[2/9] Registering resource providers..." -ForegroundColor Cyan
foreach ($provider in @("Microsoft.App", "Microsoft.OperationalInsights")) {
    Invoke-Az provider register --namespace $provider --wait | Out-Null
    Write-Host "      $provider registered"
}

# 3. Resource group (the Azure for Students subscription only allows francecentral, norwayeast,
#    polandcentral, belgiumcentral and switzerlandnorth)
Write-Host "[3/9] Resource group $ResourceGroup in $Location..." -ForegroundColor Cyan
Invoke-Az group create --name $ResourceGroup --location $Location -o none

# 4. Environment: Consumption only (pay per use, free monthly grant), no Log Analytics
#    workspace (avoids log ingestion costs; `az containerapp logs show` still streams logs)
Write-Host "[4/9] Container Apps environment $Environment..." -ForegroundColor Cyan
$envId = Find-Az containerapp env show -n $Environment -g $ResourceGroup --query id -o tsv
if (-not $envId) {
    Invoke-Az containerapp env create --name $Environment --resource-group $ResourceGroup `
        --location $Location --logs-destination none -o none
    $envId = Invoke-Az containerapp env show -n $Environment -g $ResourceGroup --query id -o tsv
}
Write-Host "      $envId"

# 5. App from the YAML template (placeholders filled in; no secret value in the file)
Write-Host "[5/9] Creating/updating the app $AppName ($Cpu vCPU, $Memory)..." -ForegroundColor Cyan
$yaml = Get-Content (Join-Path $root "deploy\containerapp.yaml") -Raw
$values = @{
    "__LOCATION__" = $Location; "__ENVIRONMENT_ID__" = $envId; "__IMAGE__" = $Image
    "__CPU__" = $Cpu; "__MEMORY__" = $Memory; "__HF_ASSETS_REPO__" = $HfAssetsRepo
    "__ORT_THREADS__" = "$OrtThreads"
}
foreach ($key in $values.Keys) { $yaml = $yaml.Replace($key, $values[$key]) }
$tmp = Join-Path $env:TEMP "anomaly-containerapp.yaml"
[System.IO.File]::WriteAllText($tmp, $yaml)
try {
    $exists = Find-Az containerapp show -n $AppName -g $ResourceGroup --query name -o tsv
    if ($exists) {
        Invoke-Az containerapp update --name $AppName --resource-group $ResourceGroup --yaml $tmp -o none
    } else {
        Invoke-Az containerapp create --name $AppName --resource-group $ResourceGroup --yaml $tmp -o none
    }
} finally {
    Remove-Item $tmp -ErrorAction SilentlyContinue
}

# 6. Secret from .env (never echoed)
Write-Host "[6/9] Setting the HF_TOKEN secret from .env..." -ForegroundColor Cyan
Invoke-Az containerapp secret set --name $AppName --resource-group $ResourceGroup --secrets "hf-token=$hfToken" -o none
Remove-Variable hfToken

# 7. Replace the running replica so it reads the current secret (changing an environment
#    variable creates a new replica; "revision restart" is not supported in express environments).
Write-Host "[7/9] Restarting the replica (DEPLOYED_AT env var)..." -ForegroundColor Cyan
$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
Invoke-Az containerapp update --name $AppName --resource-group $ResourceGroup --set-env-vars "DEPLOYED_AT=$stamp" -o none

# 8. Public URL
$fqdn = Invoke-Az containerapp show -n $AppName -g $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
Write-Host "[8/9] App URL: https://$fqdn  (the first request after idle wakes the app)" -ForegroundColor Green

# 9. Budget for this resource group with e-mail alerts at 50 %, 80 % (actual) and 100 % (forecast)
if ($SkipBudget) {
    Write-Host "[9/9] Budget skipped." -ForegroundColor DarkGray
} else {
    Write-Host "[9/9] Budget of $BudgetUsd USD/month for $ResourceGroup, alerts to $($sub.user)..." -ForegroundColor Cyan
    $start = (Get-Date -Day 1).ToString("yyyy-MM-01T00:00:00Z")
    $end = (Get-Date -Day 1).AddYears(1).ToString("yyyy-MM-01T00:00:00Z")
    $alert = { param($t, $kind) @{ enabled = $true; operator = "GreaterThan"; threshold = $t; thresholdType = $kind; contactEmails = @($sub.user) } }
    $body = @{
        properties = @{
            category = "Cost"; amount = $BudgetUsd; timeGrain = "Monthly"
            timePeriod = @{ startDate = $start; endDate = $end }
            notifications = @{
                actual50 = (& $alert 50 "Actual"); actual80 = (& $alert 80 "Actual")
                forecast100 = (& $alert 100 "Forecasted")
            }
        }
    } | ConvertTo-Json -Depth 6
    $bodyFile = Join-Path $env:TEMP "anomaly-budget.json"
    [System.IO.File]::WriteAllText($bodyFile, $body)
    $url = "https://management.azure.com/subscriptions/$($sub.id)/resourceGroups/$ResourceGroup/providers/Microsoft.Consumption/budgets/anomaly-monthly-$($BudgetUsd)usd?api-version=2023-05-01"
    & $az rest --method put --url $url --body "@$bodyFile" -o none
    if ($LASTEXITCODE -eq 0) { Write-Host "      budget created" -ForegroundColor Green }
    else { Write-Host "      Azure rejected the budget; create it in the portal." -ForegroundColor Yellow }
    Remove-Item $bodyFile -ErrorAction SilentlyContinue
}
