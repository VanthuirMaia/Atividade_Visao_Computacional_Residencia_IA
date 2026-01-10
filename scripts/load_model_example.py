# -*- coding: utf-8 -*-
"""
Exemplo de como carregar modelos salvos com metadados
"""

import sys
from pathlib import Path

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.config import MODELS_DIR
from src.model_saver import load_model_with_metadata
from src.models import SimpleCNN
import torch


def load_classic_model_example():
    """Exemplo de carregamento de modelo clássico (SVM ou Random Forest)"""
    print("="*60)
    print("EXEMPLO: Carregar Modelo Clássico (SVM/Random Forest)")
    print("="*60)
    
    # Caminho do modelo
    model_path = MODELS_DIR / 'svm_model.pkl'
    
    if not model_path.exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        print("   Execute o pipeline clássico primeiro para treinar o modelo.")
        return
    
    try:
        # Carregar modelo e metadados
        model, metadata = load_model_with_metadata(
            model_path=model_path,
            model_type='sklearn'
        )
        
        print(f"\n✅ Modelo carregado com sucesso!")
        print(f"\n📊 Metadados do modelo:")
        print(f"   Nome: {metadata.get('model_name', 'N/A')}")
        print(f"   Data de treinamento: {metadata.get('timestamp', 'N/A')}")
        print(f"   Acurácia: {metadata.get('metrics', {}).get('accuracy', 'N/A'):.4f}")
        print(f"   Precisão: {metadata.get('metrics', {}).get('precision', 'N/A'):.4f}")
        print(f"   Recall: {metadata.get('metrics', {}).get('recall', 'N/A'):.4f}")
        print(f"   F1-Score: {metadata.get('metrics', {}).get('f1_score', 'N/A'):.4f}")
        print(f"   Classes: {metadata.get('class_names', [])}")
        print(f"\n⚙️  Hiperparâmetros:")
        for key, value in metadata.get('hyperparameters', {}).items():
            print(f"   {key}: {value}")
        
        print(f"\n💡 Para usar o modelo:")
        print(f"   predictions = model.predict(X_test)")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")


def load_deep_learning_model_example():
    """Exemplo de carregamento de modelo Deep Learning (CNN ou ResNet)"""
    print("\n" + "="*60)
    print("EXEMPLO: Carregar Modelo Deep Learning (CNN/ResNet)")
    print("="*60)
    
    # Caminho do modelo
    model_path = MODELS_DIR / 'simple_cnn.pth'
    
    if not model_path.exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        print("   Execute o pipeline de Deep Learning primeiro para treinar o modelo.")
        return
    
    try:
        # Carregar modelo e metadados
        # Nota: Para carregar, precisamos recriar a estrutura do modelo
        checkpoint = torch.load(model_path, map_location='cpu')
        metadata = checkpoint.get('metadata', {})
        
        # Recriar modelo com base nos metadados
        num_classes = metadata.get('num_classes', 2)
        hyperparams = metadata.get('hyperparameters', {})
        
        model = SimpleCNN(
            num_classes=num_classes,
            dropout_rate=hyperparams.get('dropout_rate', 0.5),
            hidden_units=hyperparams.get('hidden_units', 512)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"\n✅ Modelo carregado com sucesso!")
        print(f"\n📊 Metadados do modelo:")
        print(f"   Nome: {metadata.get('model_name', 'N/A')}")
        print(f"   Data de treinamento: {metadata.get('timestamp', 'N/A')}")
        print(f"   Acurácia: {metadata.get('metrics', {}).get('accuracy', 'N/A'):.4f}")
        print(f"   Precisão: {metadata.get('metrics', {}).get('precision', 'N/A'):.4f}")
        print(f"   Recall: {metadata.get('metrics', {}).get('recall', 'N/A'):.4f}")
        print(f"   F1-Score: {metadata.get('metrics', {}).get('f1_score', 'N/A'):.4f}")
        print(f"   Classes: {metadata.get('class_names', [])}")
        print(f"\n⚙️  Hiperparâmetros:")
        for key, value in metadata.get('hyperparameters', {}).items():
            print(f"   {key}: {value}")
        
        print(f"\n💡 Para usar o modelo:")
        print(f"   model.eval()")
        print(f"   with torch.no_grad():")
        print(f"       outputs = model(images)")
        print(f"       predictions = torch.argmax(outputs, dim=1)")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")


def list_saved_models():
    """Lista todos os modelos salvos"""
    print("\n" + "="*60)
    print("MODELOS SALVOS")
    print("="*60)
    
    models = list(MODELS_DIR.glob('*.pkl')) + list(MODELS_DIR.glob('*.pth'))
    
    if not models:
        print("❌ Nenhum modelo encontrado em:", MODELS_DIR)
        return
    
    print(f"\n📁 Diretório: {MODELS_DIR}")
    print(f"\n📦 Modelos encontrados ({len(models)}):")
    
    for model_path in sorted(models):
        print(f"\n   {model_path.name}")
        
        # Tentar carregar metadados JSON
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            import json
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"      ✅ Metadados disponíveis")
                print(f"      📊 Acurácia: {metadata.get('metrics', {}).get('accuracy', 'N/A'):.4f}")
                print(f"      📅 Data: {metadata.get('timestamp', 'N/A')[:10]}")
            except:
                print(f"      ⚠️  Erro ao ler metadados")
        else:
            print(f"      ⚠️  Metadados não encontrados (modelo antigo?)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXEMPLOS DE CARREGAMENTO DE MODELOS")
    print("="*60)
    
    # Listar modelos salvos
    list_saved_models()
    
    # Exemplos de carregamento
    load_classic_model_example()
    load_deep_learning_model_example()
    
    print("\n" + "="*60)
    print("FIM DOS EXEMPLOS")
    print("="*60)
