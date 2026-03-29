#!/bin/bash
# Setup rápido do Traffic Monitor

set -e

echo "=== Traffic Monitor - Setup ==="
echo ""

# Verificar Python 3
if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python 3 não encontrado"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $PYTHON_VERSION"

# Criar venv
if [ ! -d "venv" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv venv
fi

# Ativar
source venv/bin/activate

# Instalar dependências
echo "Instalando dependências (pode demorar na primeira vez)..."
pip install --upgrade pip
pip install -r requirements.txt

# Criar config se não existe
if [ ! -f "config.yaml" ]; then
    cp config.example.yaml config.yaml
    echo ""
    echo "ATENÇÃO: Edite config.yaml com as configurações da sua câmera!"
    echo "  vim config.yaml"
fi

# Criar diretórios
mkdir -p logs clips

echo ""
echo "=== Setup completo! ==="
echo ""
echo "Próximos passos:"
echo "  1. Edite config.yaml com a URL RTSP da sua câmera"
echo "  2. Rode a calibração: python calibrate.py --rtsp rtsp://..."
echo "  3. Inicie: python main.py --debug"
echo ""
