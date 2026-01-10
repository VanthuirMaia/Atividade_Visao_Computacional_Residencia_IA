# -*- coding: utf-8 -*-
"""
Módulo para salvamento e carregamento de modelos com metadados
"""

import json
from datetime import datetime
from pathlib import Path


def save_model_with_metadata(model, model_path, metadata, model_type='pytorch'):
    """
    Salva modelo com metadados completos
    
    Args:
        model: Modelo a ser salvo
        model_path: Caminho para salvar o modelo
        metadata: Dicionário com metadados (métricas, hiperparâmetros, etc.)
        model_type: Tipo do modelo ('pytorch' ou 'sklearn')
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Salvar modelo
    if model_type == 'pytorch':
        import torch
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'metadata': metadata
        }, model_path)
    elif model_type == 'sklearn':
        import joblib
        joblib.dump({
            'model': model,
            'metadata': metadata
        }, model_path)
    
    # Salvar metadados em JSON separado
    metadata_path = model_path.with_suffix('.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Modelo salvo em: {model_path}")
    print(f"✅ Metadados salvos em: {metadata_path}")


def load_model_with_metadata(model_path, model_type='pytorch', model_class=None):
    """
    Carrega modelo com metadados
    
    Args:
        model_path: Caminho do modelo salvo
        model_type: Tipo do modelo ('pytorch' ou 'sklearn')
        model_class: Classe do modelo (necessário para PyTorch)
    
    Returns:
        model: Modelo carregado
        metadata: Dicionário com metadados
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    if model_type == 'pytorch':
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if model_class is None:
            raise ValueError("model_class é necessário para carregar modelos PyTorch")
        
        # Recriar modelo com metadados
        metadata = checkpoint.get('metadata', {})
        model = model_class(**metadata.get('model_params', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, metadata
    
    elif model_type == 'sklearn':
        import joblib
        data = joblib.load(model_path)
        return data['model'], data.get('metadata', {})
    
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")


def create_model_metadata(model_name, metrics, hyperparams, training_info, class_names):
    """
    Cria dicionário de metadados para um modelo
    
    Args:
        model_name: Nome do modelo
        metrics: Dicionário com métricas (accuracy, precision, etc.)
        hyperparams: Dicionário com hiperparâmetros
        training_info: Dicionário com informações de treinamento
        class_names: Lista de nomes das classes
    
    Returns:
        metadata: Dicionário com metadados completos
    """
    return {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': float(metrics.get('accuracy', 0)),
            'precision': float(metrics.get('precision', 0)),
            'recall': float(metrics.get('recall', 0)),
            'f1_score': float(metrics.get('f1_score', 0))
        },
        'hyperparameters': hyperparams,
        'training_info': training_info,
        'class_names': class_names,
        'num_classes': len(class_names),
        'version': '1.0'
    }
