# -*- coding: utf-8 -*-
"""
Script de diagnóstico para identificar problemas na classificação de imagens
"""

import sys
import io
from pathlib import Path

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# Definir MODELS_DIR diretamente sem importar config (para evitar problemas de dependências)
MODELS_DIR = ROOT_DIR / 'outputs' / 'models'

print("="*80)
print("DIAGNÓSTICO DE CLASSIFICAÇÃO DE IMAGENS")
print("="*80)
print()

# 1. Verificar diretório de modelos
print("1. VERIFICANDO DIRETÓRIO DE MODELOS")
print("-"*80)
print(f"Diretório: {MODELS_DIR}")
print(f"Existe: {MODELS_DIR.exists()}")
print(f"É diretório: {MODELS_DIR.is_dir()}")
print()

# 2. Listar todos os arquivos
print("2. ARQUIVOS ENCONTRADOS NO DIRETÓRIO")
print("-"*80)
if MODELS_DIR.exists():
    files = list(MODELS_DIR.glob('*'))
    print(f"Total de arquivos: {len(files)}")
    for f in sorted(files):
        size = f.stat().st_size if f.exists() else 0
        size_mb = size / (1024 * 1024)
        status = "[OK]" if f.exists() else "[ERRO]"
        print(f"  {status} {f.name:<40} ({size_mb:.2f} MB)")
else:
    print("❌ Diretório não existe!")
print()

# 3. Verificar cada modelo individualmente
print("3. VERIFICAÇÃO DETALHADA DE CADA MODELO")
print("-"*80)

models_to_check = {
    'CNN': {
        'model_file': MODELS_DIR / 'simple_cnn.pth',
        'json_file': MODELS_DIR / 'simple_cnn.json',
        'type': 'deep_learning',
        'requires': ['simple_cnn.pth']
    },
    'SVM': {
        'model_file': MODELS_DIR / 'svm_model.pkl',
        'json_file': MODELS_DIR / 'svm_model.json',
        'scaler_file': MODELS_DIR / 'svm_scaler.pkl',
        'pca_file': MODELS_DIR / 'svm_pca.pkl',
        'type': 'classic',
        'requires': ['svm_model.pkl', 'svm_scaler.pkl']
    },
    'Random Forest': {
        'model_file': MODELS_DIR / 'random_forest_model.pkl',
        'json_file': MODELS_DIR / 'random_forest_model.json',
        'scaler_file': MODELS_DIR / 'random_forest_scaler.pkl',
        'pca_file': MODELS_DIR / 'random_forest_pca.pkl',
        'type': 'classic',
        'requires': ['random_forest_model.pkl']
    },
    'ResNet50': {
        'model_file': MODELS_DIR / 'resnet50_transfer.pth',
        'json_file': MODELS_DIR / 'resnet50_transfer.json',
        'type': 'deep_learning',
        'requires': ['resnet50_transfer.pth']
    }
}

for model_name, info in models_to_check.items():
    print(f"\n[MODELO] {model_name}:")
    
    # Verificar arquivo principal
    model_file = info['model_file']
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  [OK] Modelo principal: {model_file.name} ({size_mb:.2f} MB)")
    else:
        print(f"  [ERRO] Modelo principal NAO encontrado: {model_file.name}")
        continue
    
    # Verificar JSON de metadados
    json_file = info['json_file']
    if json_file.exists():
        print(f"  [OK] Metadados JSON: {json_file.name}")
        try:
            import json
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"     - Nome: {metadata.get('model_name', 'N/A')}")
            print(f"     - Acurácia: {metadata.get('metrics', {}).get('accuracy', 0):.2%}")
            print(f"     - Classes: {metadata.get('class_names', [])}")
        except Exception as e:
            print(f"  [AVISO] Erro ao ler JSON: {e}")
    else:
        print(f"  [AVISO] Metadados JSON nao encontrados: {json_file.name}")
    
    # Verificar arquivos adicionais para modelos clássicos
    if info['type'] == 'classic':
        scaler_file = info.get('scaler_file')
        if scaler_file and scaler_file.exists():
            size_mb = scaler_file.stat().st_size / (1024 * 1024)
            print(f"  [OK] Scaler: {scaler_file.name} ({size_mb:.2f} MB)")
        elif scaler_file:
            print(f"  [AVISO] Scaler nao encontrado (opcional): {scaler_file.name}")
        
        pca_file = info.get('pca_file')
        if pca_file and pca_file.exists():
            size_mb = pca_file.stat().st_size / (1024 * 1024)
            print(f"  [OK] PCA: {pca_file.name} ({size_mb:.2f} MB)")
        elif pca_file:
            print(f"  [INFO] PCA nao encontrado (opcional): {pca_file.name}")

print()
print("="*80)
print("4. TESTE DE CARREGAMENTO DE MODELOS")
print("-"*80)

# Tentar carregar cada modelo
import traceback

for model_name, info in models_to_check.items():
    model_file = info['model_file']
    
    if not model_file.exists():
        print(f"\n[PULADO] {model_name}: arquivo nao existe")
        continue
    
    print(f"\n[TESTE] Carregando {model_name}...")
    
    try:
        if info['type'] == 'deep_learning':
            # Testar carregamento de modelo deep learning
            try:
                import torch
                device = 'cpu'
                checkpoint = torch.load(model_file, map_location=device)
                metadata = checkpoint.get('metadata', {})
                model_state = checkpoint.get('model_state_dict')
                
                if model_state is None:
                    print(f"  [ERRO] 'model_state_dict' nao encontrado no checkpoint!")
                    print(f"     Chaves disponiveis: {list(checkpoint.keys())}")
                else:
                    print(f"  [OK] Checkpoint carregado com sucesso!")
                    print(f"     - Chaves no checkpoint: {list(checkpoint.keys())}")
                    print(f"     - Metadados presentes: {'metadata' in checkpoint}")
                    print(f"     - Estado do modelo presente: {'model_state_dict' in checkpoint}")
            except ImportError:
                print(f"  [ERRO] PyTorch nao esta instalado! Instale com: pip install torch")
                
        else:  # classic
            # Testar carregamento de modelo clássico
            try:
                import joblib
                data = joblib.load(model_file)
                
                if 'model' not in data:
                    print(f"  [ERRO] 'model' nao encontrado no arquivo pickle!")
                    print(f"     Chaves disponiveis: {list(data.keys())}")
                else:
                    model = data['model']
                    print(f"  [OK] Modelo carregado com sucesso!")
                    print(f"     - Tipo: {type(model).__name__}")
                    print(f"     - Chaves no arquivo: {list(data.keys())}")
                    
                    # Verificar scaler
                    scaler_file = info.get('scaler_file')
                    if scaler_file and scaler_file.exists():
                        try:
                            scaler = joblib.load(scaler_file)
                            print(f"  [OK] Scaler carregado: {type(scaler).__name__}")
                        except Exception as e:
                            print(f"  [ERRO] Erro ao carregar scaler: {e}")
                    elif 'scaler_file' in info and scaler_file:
                        print(f"  [AVISO] Scaler nao encontrado: {scaler_file.name}")
            except ImportError:
                print(f"  [ERRO] joblib nao esta instalado! Instale com: pip install joblib")
                
    except Exception as e:
        print(f"  [ERRO] Erro ao carregar {model_name}:")
        print(f"     Tipo: {type(e).__name__}")
        print(f"     Mensagem: {str(e)}")
        print(f"     Traceback completo:")
        traceback.print_exc()
        print()

print()
print("="*80)
print("5. TESTE DE FUNÇÕES DE CLASSIFICAÇÃO")
print("-"*80)

# Verificar se as funções de classificação podem ser importadas
try:
    from classify_image import (
        classify_with_all_models,
        load_classic_model,
        load_deep_learning_model,
        classify_with_classic_model,
        classify_with_deep_learning_model
    )
    print("[OK] Funcoes de classificacao importadas com sucesso")
except Exception as e:
    print(f"[ERRO] Erro ao importar funcoes de classificacao: {e}")
    print("       Isso pode ser esperado se dependencias nao estao instaladas")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("6. RECOMENDACOES")
print("-"*80)

# Contar modelos válidos
valid_models = []
for model_name, info in models_to_check.items():
    model_file = info['model_file']
    if model_file.exists():
        valid_models.append(model_name)

if len(valid_models) == 0:
    print("[ERRO] NENHUM modelo encontrado!")
    print("       Execute: python main.py para treinar os modelos")
elif len(valid_models) < len(models_to_check):
    print(f"[AVISO] Apenas {len(valid_models)}/{len(models_to_check)} modelos encontrados")
    print(f"        Modelos encontrados: {', '.join(valid_models)}")
    missing = [m for m in models_to_check.keys() if m not in valid_models]
    print(f"        Modelos faltando: {', '.join(missing)}")
    print("        Execute o pipeline completo para ter todos os modelos disponiveis")
else:
    print(f"[OK] {len(valid_models)} modelos encontrados e prontos para uso!")
    print(f"     Modelos: {', '.join(valid_models)}")

print()
print("="*80)
