#!/bin/bash
# ============================================================================
# 🚀 SCRIPT DE DEPLOY - SISTEMA DE PREVISÃO DE RENDIMENTO DE COLHEITA
# ============================================================================
# Script profissional para automatizar o deployment da aplicação
# Autor: Análise Profissional de Dados
# Versão: 2.0
# ============================================================================

set -e  # Para execução em caso de erro

# Cores para output mais bonito
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configurações
APP_NAME="crop-yield-predictor"
IMAGE_NAME="crop-yield-predictor:latest"
CONTAINER_NAME="crop_yield_app"
PORT="8501"

# Funções auxiliares
print_header() {
    echo -e "${PURPLE}"
    echo "============================================================================"
    echo "🌾 DEPLOY - SISTEMA DE PREVISÃO DE RENDIMENTO DE COLHEITA"
    echo "============================================================================"
    echo -e "${NC}"
}

print_step() {
    echo -e "${BLUE}🔹 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_dependencies() {
    print_step "Verificando dependências..."
    
    # Verificar Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker não está instalado!"
        echo "Instale o Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Verificar Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_warning "Docker Compose não encontrado, tentando 'docker compose'..."
        if ! docker compose version &> /dev/null; then
            print_error "Docker Compose não está disponível!"
            exit 1
        else
            COMPOSE_CMD="docker compose"
        fi
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    print_success "Dependências verificadas"
}

check_files() {
    print_step "Verificando arquivos necessários..."
    
    required_files=("Dockerfile" "docker-compose.yml" "requirements.txt" "crop_yield_predictor.py" "app_streamlit.py" "crop_yield_data.csv")
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "Arquivo obrigatório não encontrado: $file"
            exit 1
        fi
    done
    
    print_success "Todos os arquivos necessários encontrados"
}

cleanup_containers() {
    print_step "Limpando containers existentes..."
    
    # Parar containers existentes
    if docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
        print_step "Parando container existente..."
        docker stop $CONTAINER_NAME || true
    fi
    
    # Remover containers existentes
    if docker ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
        print_step "Removendo container existente..."
        docker rm $CONTAINER_NAME || true
    fi
    
    print_success "Limpeza concluída"
}

build_image() {
    print_step "Construindo imagem Docker..."
    
    echo "Iniciando build da imagem: $IMAGE_NAME"
    
    if docker build -t $IMAGE_NAME .; then
        print_success "Imagem construída com sucesso!"
    else
        print_error "Falha na construção da imagem!"
        exit 1
    fi
}

run_container() {
    print_step "Executando container..."
    
    if $COMPOSE_CMD up -d; then
        print_success "Container iniciado com sucesso!"
    else
        print_error "Falha ao iniciar container!"
        exit 1
    fi
}

check_health() {
    print_step "Verificando saúde da aplicação..."
    
    echo "Aguardando aplicação inicializar..."
    sleep 10
    
    max_attempts=12
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:$PORT/_stcore/health &> /dev/null; then
            print_success "Aplicação está saudável e respondendo!"
            return 0
        fi
        
        echo "Tentativa $attempt/$max_attempts - aguardando..."
        sleep 5
        ((attempt++))
    done
    
    print_warning "Aplicação pode estar demorando para inicializar"
    print_step "Verificando logs..."
    docker logs $CONTAINER_NAME --tail 20
}

show_info() {
    echo ""
    echo -e "${PURPLE}============================================================================${NC}"
    echo -e "${GREEN}🎉 DEPLOY CONCLUÍDO COM SUCESSO!${NC}"
    echo -e "${PURPLE}============================================================================${NC}"
    echo ""
    echo -e "${BLUE}📱 Aplicação disponível em:${NC}"
    echo -e "   🌐 Local: ${GREEN}http://localhost:$PORT${NC}"
    echo -e "   🌐 Rede: ${GREEN}http://$(hostname -I | awk '{print $1}'):$PORT${NC}"
    echo ""
    echo -e "${BLUE}🛠️  Comandos úteis:${NC}"
    echo -e "   📊 Ver logs:        ${YELLOW}docker logs $CONTAINER_NAME -f${NC}"
    echo -e "   🔍 Status:          ${YELLOW}docker ps${NC}"
    echo -e "   ⏹️  Parar:           ${YELLOW}$COMPOSE_CMD down${NC}"
    echo -e "   🔄 Reiniciar:       ${YELLOW}$COMPOSE_CMD restart${NC}"
    echo -e "   📈 Monitoramento:   ${YELLOW}docker stats${NC}"
    echo ""
    echo -e "${BLUE}🏗️  Para rebuild:${NC}"
    echo -e "   ${YELLOW}$COMPOSE_CMD up --build -d${NC}"
    echo ""
    echo -e "${PURPLE}============================================================================${NC}"
}

# Função principal
main() {
    print_header
    
    # Verificações preliminares
    check_dependencies
    check_files
    
    # Deploy
    cleanup_containers
    
    # Escolher método de deploy
    if [[ "$1" == "--docker-only" ]]; then
        build_image
        docker run -d --name $CONTAINER_NAME -p $PORT:$PORT $IMAGE_NAME
    else
        run_container
    fi
    
    # Verificações pós-deploy
    check_health
    show_info
}

# Função de ajuda
show_help() {
    echo "🌾 Script de Deploy - Sistema de Previsão de Rendimento de Colheita"
    echo ""
    echo "Uso:"
    echo "  ./deploy.sh              # Deploy com Docker Compose (recomendado)"
    echo "  ./deploy.sh --docker-only # Deploy apenas com Docker"
    echo "  ./deploy.sh --help       # Mostrar esta ajuda"
    echo ""
    echo "Pré-requisitos:"
    echo "  - Docker instalado"
    echo "  - Docker Compose instalado (para método padrão)"
    echo "  - Arquivos do projeto na pasta atual"
}

# Verificar argumentos
case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac 