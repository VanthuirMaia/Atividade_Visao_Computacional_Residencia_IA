# -*- coding: utf-8 -*-
"""
Script de Diagnóstico: Verificação de Uso de GPU
Verifica se os modelos deep learning estão usando GPU corretamente
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

import torch
from src.config import USE_GPU, TRAIN_DIR, TEST_DIR
from src.utils import setup_device


def diagnose_gpu():
    """
    Diagnóstico completo do uso de GPU
    """
    print("="*80)
    print("DIAGNÓSTICO DE GPU - MODELOS DEEP LEARNING")
    print("="*80)
    
    # 1. Verificar PyTorch
    print("\n1. VERIFICAÇÃO PYTORCH")
    print("-" * 80)
    print(f"Versão PyTorch: {torch.__version__}")
    print(f"CUDA compilada: {torch.version.cuda if torch.version.cuda else 'N/A'}")
    print(f"cuDNN disponível: {torch.backends.cudnn.is_available()}")
    if torch.backends.cudnn.is_available():
        print(f"Versão cuDNN: {torch.backends.cudnn.version()}")
    
    # 2. Verificar CUDA
    print("\n2. VERIFICAÇÃO CUDA")
    print("-" * 80)
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Número de GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Nome: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memória total: {props.total_memory / (1024**3):.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Memória atual
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"  Memória alocada: {allocated:.2f} GB")
            print(f"  Memória reservada: {reserved:.2f} GB")
    else:
        print("❌ CUDA não está disponível!")
        print("   Possíveis razões:")
        print("   - PyTorch instalado sem suporte CUDA (versão CPU-only)")
        print("   - Drivers NVIDIA não instalados ou desatualizados")
        print("   - GPU não compatível")
        print("   - CUDA toolkit não instalado")
    
    # 3. Verificar configuração
    print("\n3. CONFIGURAÇÃO DO PROJETO")
    print("-" * 80)
    print(f"USE_GPU (config.py): {USE_GPU}")
    device = setup_device(USE_GPU)
    print(f"Dispositivo configurado: {device}")
    print(f"Tipo do dispositivo: {type(device)}")
    if isinstance(device, torch.device):
        print(f"Dispositivo.type: {device.type}")
        print(f"Dispositivo.index: {device.index}")
    
    # 4. Teste prático: Criar tensor na GPU
    print("\n4. TESTE PRÁTICO")
    print("-" * 80)
    if torch.cuda.is_available() and USE_GPU:
        try:
            print("Criando tensor de teste na GPU...")
            test_tensor = torch.randn(1000, 1000).to(device)
            result = test_tensor @ test_tensor.t()
            print(f"✅ Tensor criado com sucesso na {device}")
            print(f"   Dispositivo do tensor: {test_tensor.device}")
            print(f"   Dispositivo do resultado: {result.device}")
            
            # Verificar memória GPU
            if device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated(device.index or 0) / (1024**2)
                print(f"   Memória GPU alocada: {mem_allocated:.2f} MB")
            
            # Limpar
            del test_tensor, result
            torch.cuda.empty_cache()
            print("   Tensor de teste removido")
            
        except Exception as e:
            print(f"❌ ERRO ao criar tensor na GPU: {e}")
            print(f"   Tentando CPU como fallback...")
            device = torch.device('cpu')
            test_tensor = torch.randn(100, 100).to(device)
            print(f"   ✅ Tensor criado na CPU como fallback")
    else:
        print("ℹ️  GPU não disponível ou não configurada. Usando CPU para teste...")
        device = torch.device('cpu')
        test_tensor = torch.randn(100, 100).to(device)
        print(f"✅ Tensor criado na CPU")
    
    # 5. Teste com modelo simples
    print("\n5. TESTE COM MODELO")
    print("-" * 80)
    try:
        from src.models import SimpleCNN
        
        print(f"Criando modelo SimpleCNN...")
        model = SimpleCNN(num_classes=2)
        
        # Mover para dispositivo
        model = model.to(device)
        model_device = next(model.parameters()).device
        
        print(f"✅ Modelo criado")
        print(f"   Dispositivo do modelo: {model_device}")
        print(f"   Dispositivo esperado: {device}")
        
        if model_device == device:
            print(f"   ✅ Modelo está no dispositivo correto!")
        else:
            print(f"   ⚠️  AVISO: Modelo não está no dispositivo esperado!")
            print(f"      Mover manualmente: model = model.to({device})")
        
        # Teste de forward pass
        print(f"\n   Testando forward pass...")
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(device)
            output = model(test_input)
            print(f"   ✅ Forward pass bem-sucedido!")
            print(f"      Input device: {test_input.device}")
            print(f"      Output shape: {output.shape}")
            
            if test_input.device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated(device.index or 0) / (1024**2)
                print(f"      Memória GPU após forward: {mem_allocated:.2f} MB")
        
        # Limpar
        del model, test_input, output
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ ERRO ao testar modelo: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Verificar ResNet50
    print("\n6. TESTE COM RESNET50")
    print("-" * 80)
    try:
        from torchvision import models
        
        print(f"Criando ResNet50...")
        model = models.resnet50(weights='IMAGENET1K_V2')
        
        # Mover para dispositivo
        model = model.to(device)
        model_device = next(model.parameters()).device
        
        print(f"✅ ResNet50 criado")
        print(f"   Dispositivo do modelo: {model_device}")
        
        if model_device == device:
            print(f"   ✅ ResNet50 está no dispositivo correto!")
        else:
            print(f"   ⚠️  AVISO: ResNet50 não está no dispositivo esperado!")
        
        # Verificar memória
        if device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated(device.index or 0) / (1024**2)
            print(f"   Memória GPU alocada: {mem_allocated:.2f} MB")
        
        # Limpar
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ ERRO ao testar ResNet50: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. Resumo e recomendações
    print("\n7. RESUMO E RECOMENDAÇÕES")
    print("-" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA não está disponível!")
        print("\nAÇÕES RECOMENDADAS:")
        print("1. Verificar se drivers NVIDIA estão instalados: nvidia-smi")
        print("2. Instalar PyTorch com suporte CUDA:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("3. Verificar versão do CUDA instalada: nvcc --version")
        print("4. Verificar se GPU está visível: nvidia-smi")
    elif USE_GPU and device.type != 'cuda':
        print("⚠️  GPU está disponível mas modelo não está usando GPU!")
        print("\nPOSSÍVEIS CAUSAS:")
        print("1. Erro ao mover modelo para GPU (verificar código)")
        print("2. Modelo criado antes de ser movido para dispositivo")
        print("3. Weights carregados na CPU antes de mover para GPU")
        print("\nAÇÕES RECOMENDADAS:")
        print("1. Verificar se modelo.to(device) é chamado após criação")
        print("2. Verificar se create_resnet_model() move modelo para GPU")
        print("3. Verificar logs de verificação de dispositivo durante treinamento")
    elif USE_GPU and device.type == 'cuda':
        print("✅ GPU está disponível e configurada corretamente!")
        print("✅ Modelos devem usar GPU automaticamente")
        print("\nVERIFICAÇÕES ADICIONAIS:")
        print("1. Durante treinamento, verificar mensagens: 'Modelo está em: cuda:0'")
        print("2. Verificar uso de GPU: nvidia-smi (durante treinamento)")
        print("3. Verificar logs que mostram '✅ Modelo está na GPU'")
    else:
        print("ℹ️  GPU não está configurada para uso (USE_GPU=False)")
        print("   Para usar GPU, altere USE_GPU=True em src/config.py")
    
    print("\n" + "="*80)
    print("DIAGNÓSTICO CONCLUÍDO")
    print("="*80)


if __name__ == "__main__":
    diagnose_gpu()
