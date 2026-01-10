# -*- coding: utf-8 -*-
"""
Script para verificar se GPU/CPU está sendo usado corretamente
"""

import sys
from pathlib import Path

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

print("="*60)
print("VERIFICAÇÃO DE GPU/CPU")
print("="*60)

# 1. Verificar PyTorch
print("\n1. VERIFICANDO PYTORCH:")
try:
    import torch
    print(f"   [OK] PyTorch instalado - Versao: {torch.__version__}")
except ImportError:
    print("   [ERRO] PyTorch nao esta instalado!")
    print("   Execute: pip install torch torchvision")
    sys.exit(1)

# 2. Verificar CUDA disponível
print("\n2. VERIFICANDO CUDA:")
cuda_available = torch.cuda.is_available()
print(f"   CUDA disponivel: {'[OK] Sim' if cuda_available else '[ERRO] Nao'}")

if cuda_available:
    print(f"   Versao CUDA: {torch.version.cuda}")
    print(f"   Versao cuDNN: {torch.backends.cudnn.version()}")
    print(f"   Numero de GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"     Nome: {torch.cuda.get_device_name(i)}")
        print(f"     Capacidade: {torch.cuda.get_device_capability(i)}")
        print(f"     Memoria Total: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
        print(f"     Memoria Alocada: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
        print(f"     Memoria Reservada: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
else:
    print("   [AVISO] GPU nao disponivel - sera usado CPU")
    print("   Possiveis razoes:")
    print("     - PyTorch foi instalado sem suporte CUDA")
    print("     - GPU nao compativel ou drivers nao instalados")
    print("     - CUDA nao esta instalado no sistema")

# 3. Verificar configuração do projeto
print("\n3. CONFIGURACAO DO PROJETO:")
try:
    # Ler config diretamente sem importar módulos pesados
    config_path = ROOT_DIR / 'src' / 'config.py'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'USE_GPU = True' in content:
                use_gpu_config = True
            elif 'USE_GPU = False' in content:
                use_gpu_config = False
            else:
                use_gpu_config = True  # Padrão
        print(f"   USE_GPU em config.py: {use_gpu_config}")
        
        if use_gpu_config and not cuda_available:
            print("   [AVISO] USE_GPU=True mas GPU nao disponivel!")
            print("   O projeto usara CPU mesmo com USE_GPU=True")
        elif use_gpu_config and cuda_available:
            print("   [OK] Configuracao correta - usara GPU")
        elif not use_gpu_config:
            print("   [INFO] USE_GPU=False - forcando uso de CPU")
    else:
        print("   [AVISO] config.py nao encontrado")
except Exception as e:
    print(f"   [ERRO] Erro ao ler config: {e}")

# 4. Simular setup_device (sem importar utils)
print("\n4. SIMULANDO setup_device():")
try:
    # Ler config
    try:
        config_path = ROOT_DIR / 'src' / 'config.py'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                use_gpu_config = 'USE_GPU = True' in content and 'USE_GPU = False' not in content
        else:
            use_gpu_config = True
    except:
        use_gpu_config = True
    
    if cuda_available and use_gpu_config:
        device_gpu = torch.device('cuda')
        print(f"   Com use_gpu=True:")
        print(f"     Dispositivo: {device_gpu}")
        print(f"     Tipo: {device_gpu.type}")
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
    else:
        device_gpu = torch.device('cpu')
        print(f"   Com use_gpu=True (mas GPU nao disponivel ou config=False):")
        print(f"     Dispositivo: {device_gpu}")
    
    device_cpu = torch.device('cpu')
    print(f"\n   Com use_gpu=False:")
    print(f"     Dispositivo: {device_cpu}")
except Exception as e:
    print(f"   [ERRO] Erro ao simular setup_device: {e}")

# 5. Teste prático - criar tensor
print("\n5. TESTE PRATICO - CRIANDO TENSOR:")
try:
    # Determinar dispositivo baseado em disponibilidade e config
    try:
        config_path = ROOT_DIR / 'src' / 'config.py'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                use_gpu_config = 'USE_GPU = True' in content and 'USE_GPU = False' not in content
        else:
            use_gpu_config = True
    except:
        use_gpu_config = True
    
    # Escolher dispositivo
    if use_gpu_config and cuda_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"   Dispositivo configurado: {device}")
    
    # Criar tensor de teste
    test_tensor = torch.randn(100, 100).to(device)
    print(f"   [OK] Tensor criado com sucesso")
    print(f"   Localizacao do tensor: {test_tensor.device}")
    
    # Operação de teste
    result = torch.matmul(test_tensor, test_tensor.t())
    print(f"   [OK] Operacao executada com sucesso")
    print(f"   Localizacao do resultado: {result.device}")
    
    if test_tensor.device.type == 'cuda':
        print("\n   [OK] CONFIRMADO: Usando GPU para calculos!")
        # Mostrar uso de memória GPU
        print(f"   Memoria GPU usada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    else:
        print("\n   [INFO] Usando CPU para calculos")
    
    # Limpar
    del test_tensor, result
    if cuda_available:
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"   [ERRO] Erro no teste pratico: {e}")

# 6. Verificar uso durante treinamento (simulado)
print("\n6. VERIFICACAO PARA TREINAMENTO:")
try:
    # Determinar dispositivo
    try:
        config_path = ROOT_DIR / 'src' / 'config.py'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                use_gpu_config = 'USE_GPU = True' in content and 'USE_GPU = False' not in content
        else:
            use_gpu_config = True
    except:
        use_gpu_config = True
    
    if use_gpu_config and cuda_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if device.type == 'cuda':
        print("   [OK] Durante o treinamento, os modelos usarao GPU")
        print("   [OK] Dados (images, labels) serao movidos para GPU automaticamente")
        print(f"   GPU que sera usada: {torch.cuda.get_device_name(device.index or 0)}")
        
        # Verificar memória disponível
        total_mem = torch.cuda.get_device_properties(device.index or 0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device.index or 0) / (1024**3)
        cached = torch.cuda.memory_reserved(device.index or 0) / (1024**3)
        free = total_mem - cached
        
        print(f"\n   Memoria GPU disponivel para treinamento:")
        print(f"     Total: {total_mem:.2f} GB")
        print(f"     Livre: {free:.2f} GB")
        print(f"     Usada: {cached:.2f} GB")
        
        if free < 2.0:
            print("     [AVISO] Pouca memoria GPU disponivel!")
            print("     Considere reduzir BATCH_SIZE em config.py")
    else:
        print("   [INFO] Durante o treinamento, os modelos usarao CPU")
        print("   [AVISO] Treinamento sera mais lento - considere usar GPU se disponivel")
        
except Exception as e:
    print(f"   [ERRO] Erro na verificacao: {e}")

# 7. Resumo
print("\n" + "="*60)
print("RESUMO")
print("="*60)

try:
    # Ler config
    try:
        config_path = ROOT_DIR / 'src' / 'config.py'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                use_gpu_config = 'USE_GPU = True' in content and 'USE_GPU = False' not in content
        else:
            use_gpu_config = True
    except:
        use_gpu_config = True
    
    # Determinar dispositivo que será usado
    if use_gpu_config and cuda_available:
        device = torch.device('cuda')
        print("[OK] STATUS: GPU ESTA SENDO USADA")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Config USE_GPU: {use_gpu_config}")
        print(f"   Memoria Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        device = torch.device('cpu')
        print("[INFO] STATUS: CPU ESTA SENDO USADA")
        print(f"   Config USE_GPU: {use_gpu_config}")
        if use_gpu_config and not cuda_available:
            print("   [AVISO] USE_GPU=True mas GPU nao disponivel")
            print("   Para usar GPU, voce precisa:")
            print("     1. PyTorch ja esta instalado com CUDA")
            print("     2. Verificar se GPU esta funcionando")
            print("     3. Verificar drivers NVIDIA")
    
    print("\nPara alterar o comportamento, edite src/config.py:")
    print("   USE_GPU = True   # Tenta usar GPU")
    print("   USE_GPU = False  # Forca uso de CPU")
    
except Exception as e:
    print(f"[ERRO] Erro ao gerar resumo: {e}")

print("="*60)
