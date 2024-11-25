# Nome do serviço definido no docker-compose.yml
$serviceName = "naca-airfoil"

Write-Host "Stopping and removing existing containers..." -ForegroundColor Yellow
# Parar os contêineres associados ao docker-compose.yml
docker-compose down --volumes --remove-orphans

Write-Host "Removing unused Docker volumes..." -ForegroundColor Yellow
# Remover volumes não utilizados
docker volume prune -f

Write-Host "Building the Docker image..." -ForegroundColor Green
# Construir a imagem a partir do docker-compose.yml
docker-compose build

Write-Host "Deploying the application in detached mode..." -ForegroundColor Green
# Subir o contêiner em modo detached
docker-compose up -d

Write-Host "Deployment completed. Access the application at http://localhost:8501" -ForegroundColor Cyan
