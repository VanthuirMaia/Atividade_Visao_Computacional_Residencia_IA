# -*- coding: utf-8 -*-
"""
Script de Verificação: PyTorch e CUDA
Verifica se PyTorch está instalado corretamente com suporte CUDA
"""

import sys
import os

# Configurar encoding para UTF-8 no Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

print("="*80)
print("VERIFICAÇÃO DE PYTORCH E CUDA")
print("="*80)

# 1. Verificar se PyTorch está instalado
print("\n1. VERIFICAÇÃO DE INSTALAÇÃO DO PYTORCH")
print("-" * 80)
try:
    import torch
    print(f"[OK] PyTorch esta instalado")
    print(f"   Versão: {torch.__version__}")
except ImportError:
    print("[ERRO] PyTorch NAO esta instalado!")
    print("\n   Para instalar PyTorch com CUDA:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# 2. Verificar suporte CUDA
print("\n2. VERIFICAÇÃO DE SUPORTE CUDA")
print("-" * 80)
cuda_available = torch.cuda.is_available()
print(f"CUDA disponível: {cuda_available}")

if cuda_available:
    print("[OK] CUDA esta disponivel!")
    
    # Informações CUDA
    print(f"\n   Informações CUDA:")
    print(f"     Versão CUDA compilada: {torch.version.cuda if torch.version.cuda else 'N/A'}")
    print(f"     cuDNN disponível: {torch.backends.cudnn.is_available()}")
    if torch.backends.cudnn.is_available():
        print(f"     Versão cuDNN: {torch.backends.cudnn.version()}")
    
    # Informações GPU
    gpu_count = torch.cuda.device_count()
    print(f"\n   GPUs detectadas: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\n   GPU {i}:")
        try:
            gpu_name = torch.cuda.get_device_name(i)
            print(f"     Nome: {gpu_name}")
            
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)
            print(f"     Memória total: {total_mem:.2f} GB")
            print(f"     Compute Capability: {props.major}.{props.minor}")
            
            # Memória atual
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"     Memória alocada: {allocated:.2f} GB")
            print(f"     Memória reservada: {reserved:.2f} GB")
        except Exception as e:
            print(f"     Erro ao obter informações: {e}")
    
    # Teste prático
    print(f"\n   Teste prático com GPU:")
    try:
        device = torch.device('cuda:0')
        test_tensor = torch.randn(100, 100).to(device)
        result = test_tensor @ test_tensor.t()
        
        print(f"     [OK] Tensor criado na GPU: {test_tensor.device}")
        print(f"     [OK] Operacao executada na GPU: {result.device}")
        print(f"     [OK] GPU esta funcionando corretamente!")
        
        # Limpar
        del test_tensor, result
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        error_msg = str(e).lower()
        if 'out of memory' in error_msg:
            print(f"     [AVISO] GPU sem memoria disponivel (pode estar ocupada)")
        else:
            print(f"     [ERRO] ERRO ao testar GPU: {e}")
    except Exception as e:
        print(f"     [ERRO] ERRO inesperado: {type(e).__name__}: {e}")
        
else:
    print("[ERRO] CUDA NAO esta disponivel!")
    print("\n   Possíveis razões:")
    print("   1. PyTorch instalado sem suporte CUDA (versão CPU-only)")
    print("   2. Drivers NVIDIA não instalados ou desatualizados")
    print("   3. CUDA toolkit não instalado no sistema")
    print("   4. GPU não compatível")
    print("   5. GPU está ocupada por outro processo")
    
    # Diagnóstico adicional
    print(f"\n   Diagnóstico adicional:")
    print(f"     Versão PyTorch: {torch.__version__}")
    print(f"     Versão CUDA compilada: {torch.version.cuda if torch.version.cuda else 'N/A'}")
    
    if torch.version.cuda:
        print(f"     [AVISO] PyTorch foi compilado COM suporte CUDA, mas CUDA nao esta disponivel")
        print(f"        Isso geralmente significa:")
        print(f"        - Drivers NVIDIA não instalados")
        print(f"        - CUDA runtime não instalado")
        print(f"        - GPU não detectada pelo sistema")
    else:
        print(f"     [ERRO] PyTorch foi compilado SEM suporte CUDA (versao CPU-only)")
        print(f"        Para instalar PyTorch com CUDA:")
        print(f"        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

# 3. Verificar configuração do projeto
print("\n3. VERIFICAÇÃO DA CONFIGURAÇÃO DO PROJETO")
print("-" * 80)
try:
    sys.path.insert(0, '.')
    from src.config import USE_GPU
    print(f"USE_GPU (config.py): {USE_GPU}")
    
    if USE_GPU and not cuda_available:
        print("   [AVISO] ATENCAO: USE_GPU=True mas CUDA nao esta disponivel!")
        print("   O projeto tentara usar GPU mas caira para CPU automaticamente")
    elif USE_GPU and cuda_available:
        print("   [OK] Configuracao correta: USE_GPU=True e CUDA disponivel")
    elif not USE_GPU:
        print("   [INFO] USE_GPU=False - projeto usara CPU intencionalmente")
        
except Exception as e:
    print(f"   [AVISO] Erro ao verificar configuracao: {e}")

# 4. Teste com setup_device
print("\n4. TESTE COM setup_device()")
print("-" * 80)
try:
    from src.utils import setup_device
    
    print("   Testando setup_device(use_gpu=True)...")
    device = setup_device(use_gpu=True)
    print(f"   Dispositivo retornado: {device}")
    print(f"   Tipo: {type(device)}")
    
    if isinstance(device, torch.device):
        if device.type == 'cuda':
            print(f"   [OK] setup_device() retornou GPU corretamente!")
        else:
            print(f"   [AVISO] setup_device() retornou CPU mesmo com CUDA disponivel")
            if cuda_available:
                print(f"   Isso pode indicar problema no teste de GPU")
    else:
        print(f"   [AVISO] setup_device() retornou tipo inesperado: {type(device)}")
        
except Exception as e:
    print(f"   [ERRO] Erro ao testar setup_device(): {e}")
    import traceback
    traceback.print_exc()

# 5. Resumo e recomendações
print("\n5. RESUMO E RECOMENDAÇÕES")
print("-" * 80)

if cuda_available:
    print("[OK] PyTorch esta configurado CORRETAMENTE com suporte CUDA")
    print("[OK] GPU esta disponivel e deve ser usada pelos modelos")
    print("\n   Se os modelos ainda estiverem usando CPU:")
    print("   1. Verifique os logs durante inicializacao do pipeline")
    print("   2. Procure por mensagens de erro no teste de GPU")
    print("   3. Verifique se ha processos ocupando a GPU (nvidia-smi)")
else:
    print("[ERRO] PyTorch NAO esta configurado para usar GPU")
    print("\n   AÇÕES RECOMENDADAS:")
    
    if torch.version.cuda:
        print("   1. Instalar drivers NVIDIA:")
        print("      - Baixe do site da NVIDIA para sua GPU")
        print("      - Reinicie o computador após instalação")
        print("   2. Verificar se GPU está visível:")
        print("      - Execute: nvidia-smi (deve mostrar sua GPU)")
        print("   3. Se nvidia-smi não funcionar:")
        print("      - Drivers não estão instalados corretamente")
    else:
        print("   1. Reinstalar PyTorch com suporte CUDA:")
        print("      pip uninstall torch torchvision")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("   2. Verificar versão CUDA do sistema:")
        print("      nvcc --version")
        print("   3. Instalar versão compatível do PyTorch")

print("\n" + "="*80)
print("VERIFICAÇÃO CONCLUÍDA")
print("="*80)
