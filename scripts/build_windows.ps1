param(
    [ValidateSet("staging", "canary", "stable")]
    [string]$Channel = "staging",
    [string]$Version = "",
    [string]$TemplatePackVersion = "unversioned",
    [string]$Python = "python",
    [string]$DistRoot = "dist",
    [switch]$SkipInstall,
    [switch]$SkipTests,
    [switch]$SkipSbom
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "==> $Name"
    & $Command
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = & $Python -c "import pathlib, tomllib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'))['project']['version'])"
}

$artifactName = "rok-resource-assistant-$Version-$Channel-windows"
$distPath = Join-Path $DistRoot $artifactName
$zipPath = Join-Path $DistRoot "$artifactName.zip"
$checksumPath = "$zipPath.sha256"
$sbomPath = Join-Path $DistRoot "$artifactName.cyclonedx.json"

if (-not $SkipInstall) {
    Invoke-Step "Install locked dependencies" {
        & $Python -m pip install --upgrade pip
        & $Python -m pip install -r requirements-lock.txt
    }
}

Invoke-Step "Validate packaging inputs" {
    & $Python scripts/validate_packaging.py --channel $Channel --template-pack-version $TemplatePackVersion
}

if (-not $SkipTests) {
    Invoke-Step "Run tests" {
        $env:QT_QPA_PLATFORM = "offscreen"
        & $Python -m pytest tests -q
    }
}

Invoke-Step "Build PyInstaller distribution" {
    & $Python -m PyInstaller --noconfirm packaging/pyinstaller/rok_resource_assistant.spec --distpath $DistRoot --workpath build/pyinstaller
}

Invoke-Step "Write release metadata" {
    $metadata = [ordered]@{
        application = "rok-resource-assistant"
        version = $Version
        channel = $Channel
        template_pack_version = $TemplatePackVersion
        built_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        source_commit = (& git rev-parse --short HEAD 2>$null)
    }
    $metadata | ConvertTo-Json | Set-Content -Path (Join-Path $distPath "release.json") -Encoding UTF8
}

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath
}

Invoke-Step "Create zip artifact" {
    Compress-Archive -Path (Join-Path $distPath "*") -DestinationPath $zipPath -Force
}

Invoke-Step "Generate SHA256 checksum" {
    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath
    "$($hash.Hash.ToLowerInvariant())  $(Split-Path -Leaf $zipPath)" | Set-Content -Path $checksumPath -Encoding ASCII
}

if (-not $SkipSbom) {
    Invoke-Step "Generate SBOM" {
        & $Python -m cyclonedx_py requirements requirements-lock.txt --of JSON --output-file $sbomPath
    }
}

Write-Host "Built artifact: $zipPath"
Write-Host "Checksum: $checksumPath"
if (-not $SkipSbom) {
    Write-Host "SBOM: $sbomPath"
}
