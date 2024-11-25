# Use uma imagem base oficial do Windows com Python
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Instale o Python
# Baixe o instalador do Python
ADD https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe C:\\python-installer.exe

# Instale o Python silenciosamente
RUN C:\\python-installer.exe /quiet InstallAllUsers=1 PrependPath=1

# Defina o diretório de trabalho dentro do contêiner
WORKDIR C:\\app

# Instale dependências necessárias
# Instalar o Visual C++ Build Tools se necessário para compilar pacotes Python
RUN powershell -Command \
    $ErrorActionPreference = 'Stop'; \
    Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "C:\\vc_redist.x64.exe"; \
    Start-Process -FilePath "C:\\vc_redist.x64.exe" -ArgumentList "/install", "/quiet", "/norestart" -NoNewWindow -Wait; \
    Remove-Item -Force C:\\vc_redist.x64.exe

# Copie o arquivo requirements.txt para o contêiner
COPY requirements.txt C:\\app\\requirements.txt

# Instale as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie o código da aplicação para o contêiner
COPY . C:\\app

# Defina a variável de ambiente para evitar buffering de saída
ENV PYTHONUNBUFFERED=1

# Exponha a porta padrão do Streamlit
EXPOSE 8501

# Comando para executar a aplicação Streamlit
CMD ["streamlit", "run", "naca_airfoil_generator.py", "--server.port=8501", "--server.address=0.0.0.0"]
