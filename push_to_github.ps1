# Script para fazer push do projeto para o GitHub
# Execute este script no PowerShell dentro do diretório do projeto

Write-Host "=== Configurando Git para o GitHub ===" -ForegroundColor Cyan

# Verificar se estamos no diretório correto
$currentDir = Get-Location
Write-Host "Diretório atual: $currentDir" -ForegroundColor Yellow

# Verificar se há arquivos Python no diretório
if (-not (Test-Path "main.py")) {
    Write-Host "ERRO: Não foi possível encontrar main.py no diretório atual." -ForegroundColor Red
    Write-Host "Certifique-se de executar este script no diretório do projeto." -ForegroundColor Red
    exit 1
}

# Inicializar git se não existir
if (-not (Test-Path ".git")) {
    Write-Host "`nInicializando repositório Git..." -ForegroundColor Green
    git init
} else {
    Write-Host "`nRepositório Git já existe." -ForegroundColor Yellow
}

# Adicionar remote (remove se já existir)
Write-Host "`nConfigurando remote do GitHub..." -ForegroundColor Green
git remote remove origin 2>$null
git remote add origin https://github.com/rodrigogus94/atividade-visao-computacional.git

# Verificar remote
Write-Host "`nRemotes configurados:" -ForegroundColor Cyan
git remote -v

# Adicionar arquivos
Write-Host "`nAdicionando arquivos ao staging..." -ForegroundColor Green
git add .

# Verificar status
Write-Host "`nStatus do repositório:" -ForegroundColor Cyan
git status

# Fazer commit
Write-Host "`nFazendo commit..." -ForegroundColor Green
$commitMessage = "Initial commit: Projeto de classificação de imagens AI Art vs Human Art com SVM e Regressão Logística"
git commit -m $commitMessage

# Renomear branch para main
Write-Host "`nConfigurando branch main..." -ForegroundColor Green
git branch -M main

# Fazer push
Write-Host "`nFazendo push para o GitHub..." -ForegroundColor Green
Write-Host "Você pode precisar inserir suas credenciais do GitHub." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Push realizado com sucesso! ===" -ForegroundColor Green
    Write-Host "Acesse: https://github.com/rodrigogus94/atividade-visao-computacional" -ForegroundColor Cyan
} else {
    Write-Host "`n=== Erro ao fazer push ===" -ForegroundColor Red
    Write-Host "Verifique suas credenciais do GitHub e tente novamente." -ForegroundColor Yellow
    Write-Host "Você pode precisar configurar um Personal Access Token." -ForegroundColor Yellow
}
