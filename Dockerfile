# ============================================================================
# 🌾 DOCKERFILE - SISTEMA DE PREVISÃO DE RENDIMENTO DE COLHEITA
# ============================================================================
# Imagem base oficial do Python otimizada
FROM python:3.12-slim

# Metadados da imagem
LABEL maintainer="Análise Profissional de Dados"
LABEL description="Sistema inteligente de previsão de rendimento de colheita"
LABEL version="2.0"

# Variáveis de ambiente para otimização
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Criando usuário não-root para segurança
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Instalando dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Definindo diretório de trabalho
WORKDIR /app

# Copiando arquivos de dependências primeiro (para cache layers)
COPY requirements.txt .

# Instalando dependências Python
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copiando código da aplicação
COPY crop_yield_predictor.py .
COPY app_streamlit.py .
COPY crop_yield_data.csv .

# Criando modelo pré-treinado (opcional - para performance)
RUN python crop_yield_predictor.py

# Mudando ownership para usuário não-root
RUN chown -R appuser:appuser /app

# Mudando para usuário não-root
USER appuser

# Expondo porta do Streamlit
EXPOSE 8501

# Health check para monitoramento
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando padrão para executar a aplicação
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"] 