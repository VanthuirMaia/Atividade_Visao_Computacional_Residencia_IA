# -*- coding: utf-8 -*-
"""
Teste Direto: Verificar se modelos usam GPU
Testa diretamente sem dependências do projeto
"""

import sys
import os

# Configurar encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

import torch

print("="*80)
print("TESTE DIRETO: VERIFICACAO DE USO DE GPU")
print("="*80)

# 1. Verificar CUDA
print("\n1. VERIFICACAO CUDA")
print("-" * 80)
print(f"CUDA disponivel: {torch.cuda.is_available()}")
print(f"Numero de GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda:0')
    print(f"Dispositivo configurado: {device}")
else:
    print("[ERRO] CUDA nao disponivel!")
    sys.exit(1)

# 2. Teste com modelo simples
print("\n2. TESTE COM MODELO SIMPLES")
print("-" * 80)
try:
    import torch.nn as nn
    
    # Criar modelo simples
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 2)
    )
    
    print("Modelo criado na CPU (padrao)")
    print(f"Dispositivo do modelo antes de mover: {next(model.parameters()).device}")
    
    # Mover para GPU
    model = model.to(device)
    print(f"Modelo movido para: {next(model.parameters()).device}")
    
    if next(model.parameters()).device.type == 'cuda':
        print("[OK] Modelo esta na GPU!")
    else:
        print("[ERRO] Modelo NAO esta na GPU!")
    
    # Teste de forward pass
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(test_input)
        print(f"Input device: {test_input.device}")
        print(f"Output device: {output.device}")
        print(f"Output shape: {output.shape}")
        
        if output.device.type == 'cuda':
            print("[OK] Forward pass executado na GPU!")
        else:
            print("[ERRO] Forward pass NAO executado na GPU!")
    
except Exception as e:
    print(f"[ERRO] Erro no teste: {e}")
    import traceback
    traceback.print_exc()

# 3. Teste com ResNet50
print("\n3. TESTE COM RESNET50")
print("-" * 80)
try:
    from torchvision import models
    
    print("Carregando ResNet50...")
    model = models.resnet50(weights='IMAGENET1K_V2')
    print(f"ResNet50 criado. Dispositivo antes: {next(model.parameters()).device}")
    
    # Mover para GPU
    model = model.to(device)
    print(f"ResNet50 movido para: {next(model.parameters()).device}")
    
    if next(model.parameters()).device.type == 'cuda':
        print("[OK] ResNet50 esta na GPU!")
        
        # Verificar memoria
        mem_allocated = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"Memoria GPU alocada: {mem_allocated:.2f} MB")
    else:
        print("[ERRO] ResNet50 NAO esta na GPU!")
    
    # Teste de forward pass
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(test_input)
        print(f"Input device: {test_input.device}")
        print(f"Output device: {output.device}")
        print(f"Output shape: {output.shape}")
        
        if output.device.type == 'cuda':
            print("[OK] ResNet50 forward pass executado na GPU!")
        else:
            print("[ERRO] ResNet50 forward pass NAO executado na GPU!")
    
    # Limpar
    del model, test_input, output
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"[ERRO] Erro no teste ResNet50: {e}")
    import traceback
    traceback.print_exc()

# 4. Simular setup_device
print("\n4. SIMULACAO DE setup_device()")
print("-" * 80)
try:
    def setup_device_simulado(use_gpu=True):
        if use_gpu:
            if not torch.cuda.is_available():
                print("  [AVISO] CUDA nao disponivel, usando CPU")
                return torch.device('cpu')
            
            device = torch.device('cuda:0')
            
            # Teste
            try:
                test_tensor = torch.randn(10, 10).to(device)
                result = test_tensor @ test_tensor.t()
                if result.device.type != 'cuda':
                    print("  [AVISO] Teste falhou - resultado nao esta na GPU")
                    return torch.device('cpu')
                del test_tensor, result
                torch.cuda.empty_cache()
                print("  [OK] Teste de GPU bem-sucedido")
            except Exception as e:
                print(f"  [AVISO] Erro no teste: {e}")
                print("  [INFO] Continuando com GPU mesmo assim")
            
            return device
        else:
            return torch.device('cpu')
    
    print("Testando setup_device_simulado(use_gpu=True)...")
    device_result = setup_device_simulado(use_gpu=True)
    print(f"Resultado: {device_result}")
    print(f"Tipo: {type(device_result)}")
    
    if device_result.type == 'cuda':
        print("[OK] setup_device() retornaria GPU corretamente!")
    else:
        print("[ERRO] setup_device() retornaria CPU!")
        
except Exception as e:
    print(f"[ERRO] Erro na simulacao: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TESTE CONCLUIDO")
print("="*80)
print("\nCONCLUSAO:")
if torch.cuda.is_available():
    print("[OK] PyTorch esta configurado corretamente")
    print("[OK] GPU esta disponivel e funcionando")
    print("\nSe os modelos ainda estiverem usando CPU durante treinamento:")
    print("1. Verifique os logs de inicializacao do pipeline")
    print("2. Procure por mensagens de erro ou avisos")
    print("3. Verifique se self.device esta sendo usado corretamente")
    print("4. Verifique se model.to(device) esta sendo chamado")
else:
    print("[ERRO] CUDA nao esta disponivel!")
