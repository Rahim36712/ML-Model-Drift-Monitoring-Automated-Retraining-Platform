param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$demoPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $demoPython)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $supportedPython = $null
        foreach ($candidateVersion in @('3.12', '3.11')) {
            & py "-$candidateVersion" -c "import sys; sys.exit(0)" 2>$null
            if ($LASTEXITCODE -eq 0) { $supportedPython = $candidateVersion; break }
        }
        if (-not $supportedPython) { throw "Install Python 3.11 or 3.12; another default Python version may be installed." }
        & py "-$supportedPython" -m venv .venv
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { & python -m venv .venv }
    else { throw "Install Python 3.11 or 3.12 from python.org, then run this script again." }
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}
& $demoPython -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), 'Python 3.11 or 3.12 is required'"
if ($LASTEXITCODE -ne 0) { throw "Use a Python 3.11 or 3.12 virtual environment." }
& $demoPython -m pip install --timeout 120 --retries 5 -r requirements.lock
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Check the network and retry; no services were started." }
if ($NoBrowser) { & $demoPython -m scripts.run_demo --no-browser }
else { & $demoPython -m scripts.run_demo }
if ($LASTEXITCODE -ne 0) { throw "Platform stopped with an error. Inspect data/demo/logs." }
