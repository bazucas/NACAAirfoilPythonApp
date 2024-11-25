# Use uma imagem base oficial do Python
FROM python:3.13-slim

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo requirements.txt para o contêiner
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie o código da aplicação para o contêiner
COPY . .

# Exponha a porta padrão do Streamlit
EXPOSE 8501

# Comando para executar a aplicação Streamlit
CMD ["streamlit", "run", "naca_airfoil_generator.py", "--server.port=8501", "--server.address=0.0.0.0"]
