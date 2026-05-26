param(
    [string]$Remote = "lrh@100.118.41.2",
    [string]$RemoteProject = "~/桌面/xunlian",
    [string]$LocalBackupDir = "C:\训练\machine_backups"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $LocalBackupDir | Out-Null

Write-Host "[fetch] remote: $Remote"
Write-Host "[fetch] remote project: $RemoteProject"
Write-Host "[fetch] local backup dir: $LocalBackupDir"

$latest = ssh $Remote "cd $RemoteProject/migration_backups && ls -t xunlian_migration_*.tar.gz | head -1"
if (-not $latest) {
    throw "No migration archive found on remote machine. Run scripts/remote_make_migration_backup.sh first."
}

$latest = $latest.Trim()
$remoteArchive = "$RemoteProject/migration_backups/$latest"
$remoteSha = "$remoteArchive.sha256"
$remoteManifest = $remoteArchive -replace '\.tar\.gz$', '_manifest.txt'

Write-Host "[fetch] archive: $latest"

scp "${Remote}:$remoteArchive" $LocalBackupDir
scp "${Remote}:$remoteSha" $LocalBackupDir
scp "${Remote}:$remoteManifest" $LocalBackupDir

$localArchive = Join-Path $LocalBackupDir $latest
$localSha = "$localArchive.sha256"

if (Test-Path $localSha) {
    $expected = (Get-Content $localSha | Select-Object -First 1).Split(" ")[0].Trim()
    $actual = (Get-FileHash -Algorithm SHA256 $localArchive).Hash.ToLower()
    if ($expected.ToLower() -ne $actual) {
        throw "SHA256 mismatch. Expected $expected but got $actual"
    }
    Write-Host "[fetch] checksum ok: $actual"
} else {
    Write-Warning "[fetch] checksum file was not downloaded"
}

Write-Host "[fetch] done: $localArchive"

