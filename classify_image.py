# -*- coding: utf-8 -*-
"""
Script para classificar uma imagem nova como AI Art ou Human Art
Usa TODOS os modelos treinados por padrão e retorna uma resposta final.

Uso:
    # Selecionar imagem usando interface gráfica (RECOMENDADO)
    python classify_image.py
    python classify_image.py --gui
    
    # Ou especificar o caminho da imagem diretamente
    python classify_image.py <caminho_da_imagem> [--model MODELO]

Por padrão, usa TODOS os modelos disponíveis e retorna uma resposta final baseada em consenso.

Modelos disponíveis:
    - all: Usa TODOS os modelos e retorna resposta final (PADRÃO)
    - cnn: CNN Simples (70.89% acurácia)
    - svm: Support Vector Machine (68.15% acurácia)
    - random_forest: Random Forest (63.01% acurácia)
    - resnet50: ResNet50 com Transfer Learning (55.14% acurácia)

Exemplos:
    # Usando interface gráfica (MAIS FÁCIL)
    python classify_image.py
    python classify_image.py --gui
    
    # Especificando caminho da imagem
    python classify_image.py minha_imagem.jpg  # Usa todos os modelos
    python classify_image.py minha_imagem.jpg --model all  # Explícito: todos os modelos
    python classify_image.py minha_imagem.jpg --model cnn  # Apenas CNN
"""

import sys
import argparse
from pathlib import Path
import json
from collections import OrderedDict
import io
import contextlib

# Verificar dependências críticas
MISSING_DEPS = []

try:
    import numpy as np
except ImportError:
    MISSING_DEPS.append("numpy (pip install numpy)")

try:
    import cv2
except ImportError:
    MISSING_DEPS.append("opencv-python (pip install opencv-python)")

try:
    import torch
    from torchvision import transforms
    import torchvision.models as torch_models
except ImportError:
    MISSING_DEPS.append("torch e torchvision (pip install torch torchvision)")

try:
    from PIL import Image, ImageOps
except ImportError:
    MISSING_DEPS.append("Pillow (pip install Pillow)")

try:
    import joblib
except ImportError:
    MISSING_DEPS.append("joblib (pip install joblib)")

# Tentar importar tkinter para interface gráfica
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# Verificar dependências antes de continuar
if MISSING_DEPS:
    print("="*80)
    print("ERRO: DEPENDÊNCIAS FALTANDO")
    print("="*80)
    print("\nAs seguintes dependências são necessárias mas não estão instaladas:\n")
    for dep in MISSING_DEPS:
        print(f"  ❌ {dep}")
    print("\n💡 SOLUÇÃO: Instale todas as dependências executando:")
    print("   pip install -r requirements.txt")
    print("\n   Ou instale manualmente:")
    for dep in MISSING_DEPS:
        print(f"   {dep}")
    print("\n" + "="*80)
    sys.exit(1)

from src.config import MODELS_DIR, IMG_SIZE, IMG_SIZE_CLASSIC
from src.models import SimpleCNN
from src.utils import setup_device


def load_and_preprocess_image_for_classic(image_path, img_size=IMG_SIZE_CLASSIC):
    """
    Carrega e pré-processa imagem para modelos clássicos (SVM, Random Forest)
    
    Args:
        image_path: Caminho da imagem
        img_size: Tamanho para redimensionar (padrão: 64x64)
    
    Returns:
        image_array: Array numpy normalizado e flatten
    """
    # Carregar imagem usando PIL (suporta melhor caracteres Unicode no Windows)
    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        # Tentar com cv2 como fallback usando np.fromfile (evita problemas de encoding)
        try:
            # Usar np.fromfile que lida melhor com caminhos Unicode no Windows
            image_path_str = str(image_path)
            image = cv2.imdecode(np.fromfile(image_path_str, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Erro ao carregar imagem com cv2: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        except Exception as e2:
            raise ValueError(f"Erro ao carregar imagem: {image_path}\n  PIL: {e}\n  OpenCV: {e2}")
    
    # Corrigir orientação EXIF
    try:
        pil_image = ImageOps.exif_transpose(pil_image)
    except Exception:
        pass
    
    # Converter para RGB se necessário
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Converter para numpy array
    image = np.array(pil_image)
    
    # Redimensionar
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
    
    # Validar formato
    if image.shape != (*img_size, 3):
        raise ValueError(f"Formato de imagem inválido após pré-processamento: {image.shape}")
    
    # Normalizar para [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Flatten (transformar em vetor)
    image_flat = image.flatten()
    
    return image_flat


def load_and_preprocess_image_for_deep_learning(image_path, img_size=IMG_SIZE):
    """
    Carrega e pré-processa imagem para modelos deep learning (CNN, ResNet50)
    
    Args:
        image_path: Caminho da imagem
        img_size: Tamanho para redimensionar (padrão: 224x224)
    
    Returns:
        image_tensor: Tensor PyTorch normalizado
    """
    # Carregar imagem usando PIL diretamente (suporta melhor caracteres Unicode no Windows)
    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        # Tentar com cv2 como fallback usando np.fromfile (evita problemas de encoding no Windows)
        try:
            # Usar np.fromfile que lida melhor com caminhos Unicode no Windows
            image_path_str = str(image_path)
            image = cv2.imdecode(np.fromfile(image_path_str, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Erro ao carregar imagem com cv2: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        except Exception as e2:
            raise ValueError(f"Erro ao carregar imagem: {image_path}\n  PIL: {e}\n  OpenCV: {e2}")
    
    # Corrigir orientação EXIF
    try:
        pil_image = ImageOps.exif_transpose(pil_image)
    except Exception:
        pass
    
    # Converter para RGB se necessário
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Aplicar transformações (mesmas do treinamento)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Aplicar transformações
    image_tensor = transform(pil_image)
    
    # Adicionar dimensão de batch
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def load_classic_model(model_name):
    """
    Carrega modelo clássico (SVM ou Random Forest) e seus componentes (scaler, PCA)
    
    Args:
        model_name: Nome do modelo ('svm' ou 'random_forest')
    
    Returns:
        model: Modelo treinado
        scaler: StandardScaler usado no treinamento
        pca: PCA usado no treinamento (ou None se não foi usado)
        metadata: Metadados do modelo
    """
    model_path = MODELS_DIR / f'{model_name}_model.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}\n"
            f"Execute o pipeline clássico primeiro para treinar o modelo."
        )
    
    # Carregar modelo e metadados
    import joblib
    data = joblib.load(model_path)
    model = data['model']
    metadata = data.get('metadata', {})
    
    # Carregar scaler (Random Forest pode usar o mesmo scaler do SVM ou não precisar)
    scaler_path = MODELS_DIR / f'{model_name}_scaler.pkl'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    elif model_name == 'random_forest':
        # Random Forest pode usar o scaler do SVM se existir, ou pode não precisar
        svm_scaler_path = MODELS_DIR / 'svm_scaler.pkl'
        if svm_scaler_path.exists():
            print(f"  [INFO] Usando scaler do SVM para Random Forest (modelos compartilham mesmo pré-processamento)")
            scaler = joblib.load(svm_scaler_path)
        else:
            raise FileNotFoundError(
                f"Scaler não encontrado: {scaler_path} nem {svm_scaler_path}\n"
                f"Os modelos clássicos precisam do scaler para fazer predições.\n"
                f"Execute o pipeline clássico novamente para gerar os arquivos necessários."
            )
    else:
        raise FileNotFoundError(
            f"Scaler não encontrado: {scaler_path}\n"
            f"O modelo precisa do scaler para fazer predições."
        )
    
    # Carregar PCA (se existir)
    pca_path = MODELS_DIR / f'{model_name}_pca.pkl'
    pca = None
    if pca_path.exists():
        pca = joblib.load(pca_path)
        print(f"  PCA carregado: {pca.n_components} componentes")
    
    return model, scaler, pca, metadata


def load_deep_learning_model(model_name):
    """
    Carrega modelo deep learning (CNN ou ResNet50)
    
    Args:
        model_name: Nome do modelo ('cnn' ou 'resnet50')
    
    Returns:
        model: Modelo treinado
        device: Dispositivo (CPU ou GPU)
        metadata: Metadados do modelo
    """
    if model_name == 'cnn':
        model_path = MODELS_DIR / 'simple_cnn.pth'
    elif model_name == 'resnet50':
        model_path = MODELS_DIR / 'resnet50_transfer.pth'
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}\n"
            f"Execute o pipeline de deep learning primeiro para treinar o modelo."
        )
    
    # Carregar checkpoint
    device = setup_device(use_gpu=False)  # Usar CPU para predição (mais estável)
    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint.get('metadata', {})
    
    # Recriar modelo
    if model_name == 'cnn':
        # Buscar parâmetros do campo correto (hyperparameters, não model_params)
        hyperparams = metadata.get('hyperparameters', {})
        model_params = metadata.get('model_params', {})  # Fallback para compatibilidade
        
        # Usar valores do JSON ou fallback para valores padrão
        dropout_rate = hyperparams.get('dropout_rate') or model_params.get('dropout_rate', 0.5)
        hidden_units = hyperparams.get('hidden_units') or model_params.get('hidden_units', 512)
        num_classes = metadata.get('num_classes', 2)
        
        print(f"  [INFO] Recriando CNN com: hidden_units={hidden_units}, dropout_rate={dropout_rate:.4f}, num_classes={num_classes}")
        
        model = SimpleCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units
        )
    elif model_name == 'resnet50':
        # Recriar ResNet50
        num_classes = metadata.get('num_classes', 2)
        model = torch_models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
        
        # Nota: Para predição, não precisamos configurar unfreeze_layers
    
    # Carregar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    return model, device, metadata


def classify_with_classic_model(image_path, model_name='svm'):
    """
    Classifica imagem usando modelo clássico (SVM ou Random Forest)
    
    Args:
        image_path: Caminho da imagem
        model_name: Nome do modelo ('svm' ou 'random_forest')
    
    Returns:
        prediction: Classe predita
        confidence: Probabilidade/confiança da predição
        class_names: Nomes das classes
        metadata: Metadados do modelo
    """
    print(f"\n{'='*60}")
    print(f"CLASSIFICANDO COM {model_name.upper()}")
    print(f"{'='*60}")
    
    # Carregar modelo
    print(f"Carregando modelo {model_name}...")
    model, scaler, pca, metadata = load_classic_model(model_name)
    
    # Obter nomes das classes
    class_names = metadata.get('class_names', ['aiartdata', 'realart'])
    
    # Carregar e pré-processar imagem
    print(f"Carregando e pré-processando imagem: {image_path}")
    image_flat = load_and_preprocess_image_for_classic(image_path)
    
    # Aplicar scaler
    print("Aplicando normalização (StandardScaler)...")
    image_scaled = scaler.transform([image_flat])
    
    # Aplicar PCA (se existir)
    if pca is not None:
        print(f"Aplicando PCA ({pca.n_components} componentes)...")
        image_scaled = pca.transform(image_scaled)
    
    # Fazer predição
    print("Fazendo predição...")
    prediction = model.predict(image_scaled)[0]
    
    # Obter probabilidades (se disponível)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(image_scaled)[0]
        confidence = float(probabilities[prediction])
        prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    else:
        # Para modelos sem predict_proba, usar decisão
        if hasattr(model, 'decision_function'):
            decision = model.decision_function(image_scaled)[0]
            confidence = abs(decision) / (abs(decision) + 1) if decision != 0 else 0.5
            prob_dict = None
        else:
            confidence = 1.0
            prob_dict = None
    
    predicted_class = class_names[prediction]
    
    return predicted_class, confidence, class_names, prob_dict, metadata


def classify_with_deep_learning_model(image_path, model_name='cnn'):
    """
    Classifica imagem usando modelo deep learning (CNN ou ResNet50)
    
    Args:
        image_path: Caminho da imagem
        model_name: Nome do modelo ('cnn' ou 'resnet50')
    
    Returns:
        prediction: Classe predita
        confidence: Probabilidade da predição
        class_names: Nomes das classes
        prob_dict: Dicionário com probabilidades de cada classe
        metadata: Metadados do modelo
    """
    print(f"\n{'='*60}")
    print(f"CLASSIFICANDO COM {model_name.upper()}")
    print(f"{'='*60}")
    
    # Carregar modelo
    print(f"Carregando modelo {model_name}...")
    model, device, metadata = load_deep_learning_model(model_name)
    
    # Obter nomes das classes
    class_names = metadata.get('class_names', ['aiartdata', 'realart'])
    
    # Carregar e pré-processar imagem
    print(f"Carregando e pré-processando imagem: {image_path}")
    image_tensor = load_and_preprocess_image_for_deep_learning(image_path)
    image_tensor = image_tensor.to(device)
    
    # Fazer predição
    print("Fazendo predição...")
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        prediction = torch.argmax(probabilities, dim=0).item()
    
    confidence = float(probabilities[prediction])
    prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    predicted_class = class_names[prediction]
    
    return predicted_class, confidence, class_names, prob_dict, metadata


def load_model_metadata(model_name):
    """
    Carrega metadados de um modelo a partir do arquivo JSON
    """
    model_file_map = {
        'svm': 'svm_model.json',
        'random_forest': 'random_forest_model.json',
        'cnn': 'simple_cnn.json',
        'resnet50': 'resnet50_transfer.json'
    }
    
    json_path = MODELS_DIR / model_file_map.get(model_name)
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    except Exception:
        return None


def classify_with_all_models(image_path):
    """
    Classifica uma imagem usando TODOS os modelos disponíveis e retorna resultado final
    
    Returns:
        final_prediction: Classe final predita ('aiartdata' ou 'realart')
        results: Dicionário com resultados de cada modelo
    """
    results = OrderedDict()
    
    # Lista de modelos a tentar (na ordem de melhor performance esperada)
    models_to_try = [
        ('cnn', True),
        ('svm', False),
        ('random_forest', False),
        ('resnet50', True)
    ]
    
    print(f"\n{'='*70}")
    print(f"CLASSIFICANDO IMAGEM COM TODOS OS MODELOS DISPONÍVEIS")
    print(f"{'='*70}")
    print(f"Imagem: {image_path}\n")
    
    # Classificar com cada modelo disponível
    for model_name, is_deep_learning in models_to_try:
        # Verificar se o modelo existe
        if is_deep_learning:
            model_file = MODELS_DIR / ('simple_cnn.pth' if model_name == 'cnn' else 'resnet50_transfer.pth')
        else:
            model_file = MODELS_DIR / f'{model_name}_model.pkl'
        
        if not model_file.exists():
            continue  # Pular modelos não encontrados silenciosamente
        
        try:
            # Classificar com o modelo (suprimindo saída detalhada)
            with contextlib.redirect_stdout(io.StringIO()):
                if is_deep_learning:
                    predicted_class, confidence, class_names, prob_dict, metadata = classify_with_deep_learning_model(
                        image_path, model_name
                    )
                else:
                    predicted_class, confidence, class_names, prob_dict, metadata = classify_with_classic_model(
                        image_path, model_name
                    )
            
            # Carregar metadados completos do JSON
            json_metadata = load_model_metadata(model_name)
            if json_metadata:
                metadata = {**metadata, **json_metadata}
            
            # Armazenar resultados
            results[model_name] = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': prob_dict or {},
                'accuracy': metadata.get('metrics', {}).get('accuracy', 0),
                'model_name': metadata.get('model_name', model_name.upper())
            }
            
        except Exception as e:
            # Mostrar erro específico para ajudar no diagnóstico
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Erros comuns e suas soluções
            if 'ModuleNotFoundError' in error_type or 'ImportError' in error_type:
                if 'joblib' in error_msg.lower():
                    print(f"  ⚠️  {model_name.upper()} pulado: joblib não instalado (pip install joblib)")
                elif 'cv2' in error_msg.lower() or 'opencv' in error_msg.lower():
                    print(f"  ⚠️  {model_name.upper()} pulado: opencv-python não instalado (pip install opencv-python)")
                else:
                    print(f"  ⚠️  {model_name.upper()} pulado: {error_type} - {error_msg[:100]}")
            elif 'FileNotFoundError' in error_type:
                print(f"  ⚠️  {model_name.upper()} pulado: Arquivo auxiliar não encontrado - {error_msg[:100]}")
            else:
                print(f"  ⚠️  {model_name.upper()} pulado: {error_type} - {error_msg[:100]}")
            continue  # Pular modelos com erro
    
    if not results:
        # Verificar se arquivos existem
        model_files_exist = []
        for model_name, is_deep_learning in [('cnn', True), ('svm', False), ('random_forest', False), ('resnet50', True)]:
            if is_deep_learning:
                model_file = MODELS_DIR / ('simple_cnn.pth' if model_name == 'cnn' else 'resnet50_transfer.pth')
            else:
                model_file = MODELS_DIR / f'{model_name}_model.pkl'
            if model_file.exists():
                model_files_exist.append(model_name)
        
        error_msg = "Nenhum modelo disponível para classificação!\n\n"
        
        if model_files_exist:
            error_msg += f"📁 Modelos encontrados no disco: {', '.join(model_files_exist)}\n"
            error_msg += "❌ Mas nenhum pôde ser carregado devido a erros.\n\n"
            error_msg += "💡 Soluções possíveis:\n"
            error_msg += "   1. Instale dependências: pip install -r requirements.txt\n"
            error_msg += "   2. Execute diagnóstico: python diagnose_classification.py\n"
            error_msg += "   3. Verifique erros acima (⚠️) para detalhes específicos\n"
        else:
            error_msg += "❌ Nenhum modelo encontrado no diretório!\n"
            error_msg += "   Execute o pipeline de treinamento primeiro: python main.py\n"
        
        raise FileNotFoundError(error_msg)
    
    # Calcular predição final baseada em votação ponderada por acurácia e confiança
    votes_ai = 0
    votes_human = 0
    total_weighted_votes = 0
    
    for model_name, result in results.items():
        # Peso = acurácia do modelo * confiança na predição
        weight = result['accuracy'] * result['confidence']
        total_weighted_votes += weight
        
        if result['predicted_class'] == 'aiartdata':
            votes_ai += weight
        else:  # realart
            votes_human += weight
    
    # Normalizar votos
    if total_weighted_votes > 0:
        votes_ai_normalized = votes_ai / total_weighted_votes
        votes_human_normalized = votes_human / total_weighted_votes
    else:
        # Se nenhum modelo funcionou, usar votação simples por maioria
        ai_count = sum(1 for r in results.values() if r['predicted_class'] == 'aiartdata')
        human_count = len(results) - ai_count
        votes_ai_normalized = ai_count / len(results) if results else 0.5
        votes_human_normalized = human_count / len(results) if results else 0.5
    
    # Decisão final
    if votes_ai_normalized > votes_human_normalized:
        final_prediction = 'aiartdata'
        final_confidence = votes_ai_normalized
    else:
        final_prediction = 'realart'
        final_confidence = votes_human_normalized
    
    return final_prediction, final_confidence, results


def print_results(predicted_class, confidence, class_names, prob_dict, metadata, model_name):
    """
    Imprime resultados da classificação de forma formatada
    """
    # Traduzir nomes de classes para português
    class_translations = {
        'aiartdata': 'Arte Gerada por IA',
        'realart': 'Arte Criada por Humano',
        'classe_a': 'Classe A',
        'classe_b': 'Classe B'
    }
    
    predicted_display = class_translations.get(predicted_class, predicted_class)
    
    print(f"\n{'='*60}")
    print("RESULTADO DA CLASSIFICAÇÃO")
    print(f"{'='*60}")
    print(f"\nModelo usado: {metadata.get('model_name', model_name.upper())}")
    print(f"Acurácia do modelo: {metadata.get('metrics', {}).get('accuracy', 0):.2%}")
    print(f"\n{'─'*60}")
    print(f"PREDIÇÃO: {predicted_display}")
    print(f"Confiança: {confidence:.2%}")
    print(f"{'─'*60}")
    
    if prob_dict:
        print(f"\nProbabilidades por classe:")
        for class_name, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
            display_name = class_translations.get(class_name, class_name)
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"  {display_name:25s}: {prob:6.2%} {bar}")
    
    print(f"\n{'='*60}\n")


def print_all_models_results(final_prediction, final_confidence, results):
    """
    Imprime resultados usando todos os modelos com resposta final
    """
    class_translations = {
        'aiartdata': 'Arte Gerada por IA',
        'realart': 'Arte Criada por Humano'
    }
    
    final_display = class_translations.get(final_prediction, final_prediction)
    
    print(f"{'='*70}")
    print("RESULTADO FINAL - CLASSIFICAÇÃO COM TODOS OS MODELOS")
    print(f"{'='*70}\n")
    print(f"{'─'*70}")
    print(f"RESPOSTA FINAL: {final_display}")
    print(f"Confiança: {final_confidence:.2%}")
    print(f"{'─'*70}\n")
    
    # Mostrar resultados individuais de cada modelo
    print("Resultados individuais de cada modelo:")
    print(f"{'─'*70}")
    
    for model_name, result in results.items():
        pred_display = class_translations.get(result['predicted_class'], result['predicted_class'])
        match_icon = "✅" if result['predicted_class'] == final_prediction else "❌"
        
        print(f"{match_icon} {result['model_name']:<20} → {pred_display:<25} "
              f"(Conf: {result['confidence']:.1%}, Acurácia: {result['accuracy']:.1%})")
    
    print(f"{'─'*70}")
    
    # Contar concordância
    matching = sum(1 for r in results.values() if r['predicted_class'] == final_prediction)
    total = len(results)
    agreement = (matching / total) * 100 if total > 0 else 0
    
    print(f"\nConcordância: {matching}/{total} modelos ({agreement:.1f}%) concordam com a resposta final")
    print(f"{'='*70}\n")


def select_image_file():
    """
    Abre uma janela de seleção de arquivo para escolher a imagem
    
    Returns:
        Path ou None: Caminho da imagem selecionada ou None se cancelado
    """
    if not TKINTER_AVAILABLE:
        print("⚠️  AVISO: tkinter não está disponível.")
        print("   Instale tkinter ou forneça o caminho da imagem como argumento.")
        print("   Exemplo: python classify_image.py caminho/para/imagem.jpg")
        return None
    
    try:
        # Criar janela root (oculta)
        root = tk.Tk()
        root.withdraw()  # Ocultar janela principal
        root.attributes('-topmost', True)  # Trazer para frente
        
        # Abrir diálogo de seleção de arquivo
        file_path = filedialog.askopenfilename(
            title="Selecione uma imagem para classificar",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
                ("GIF", "*.gif"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        root.destroy()  # Fechar janela
        
        if not file_path:
            return None
        
        return Path(file_path)
    except Exception as e:
        print(f"⚠️  ERRO ao abrir seletor de arquivo: {e}")
        print("   Por favor, forneça o caminho da imagem como argumento.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Classifica uma imagem como AI Art ou Human Art usando TODOS os modelos por padrão',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Selecionar imagem usando interface gráfica (RECOMENDADO)
  python classify_image.py
  python classify_image.py --gui
  
  # Ou especificar o caminho da imagem diretamente
  python classify_image.py minha_imagem.jpg
  python classify_image.py minha_imagem.jpg --model all
  
  # Usar apenas um modelo específico
  python classify_image.py minha_imagem.jpg --model cnn
  python classify_image.py minha_imagem.jpg --model svm
  python classify_image.py minha_imagem.jpg --model random_forest
  python classify_image.py minha_imagem.jpg --model resnet50
        """
    )
    
    parser.add_argument(
        'image_path', 
        type=str, 
        nargs='?',  # Tornar opcional
        default=None,
        help='Caminho da imagem a ser classificada (opcional - se não fornecido, abre seletor de arquivo)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='all',
        choices=['all', 'svm', 'random_forest', 'cnn', 'resnet50'],
        help='Modelo a ser usado: "all" usa todos os modelos (PADRÃO), ou escolha um específico'
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Forçar abertura da interface gráfica para seleção de arquivo'
    )
    
    args = parser.parse_args()
    
    # Obter caminho da imagem
    if args.gui or args.image_path is None:
        if not TKINTER_AVAILABLE:
            print("❌ ERRO: tkinter não está disponível e nenhum caminho de imagem foi fornecido.")
            print("\n💡 Soluções:")
            print("   1. Instale tkinter (geralmente vem com Python)")
            print("   2. Ou forneça o caminho da imagem como argumento:")
            print("      python classify_image.py caminho/para/imagem.jpg")
            sys.exit(1)
        
        # Abrir interface gráfica para selecionar imagem
        print("📁 Abrindo seletor de arquivo...")
        image_path = select_image_file()
        
        if image_path is None:
            print("❌ Nenhuma imagem selecionada. Operação cancelada.")
            sys.exit(0)
        
        print(f"✅ Imagem selecionada: {image_path}")
    else:
        # Usar caminho fornecido
        image_path = Path(args.image_path)
    
    # Verificar se a imagem existe
    if not image_path.exists():
        print(f"❌ ERRO: Imagem não encontrada: {image_path}")
        sys.exit(1)
    
    try:
        # Se usar 'all', classificar com todos os modelos
        if args.model == 'all':
            final_prediction, final_confidence, results = classify_with_all_models(image_path)
            print_all_models_results(final_prediction, final_confidence, results)
        else:
            # Classificar com modelo específico
            if args.model in ['svm', 'random_forest']:
                predicted_class, confidence, class_names, prob_dict, metadata = classify_with_classic_model(
                    image_path, args.model
                )
            else:
                predicted_class, confidence, class_names, prob_dict, metadata = classify_with_deep_learning_model(
                    image_path, args.model
                )
            
            # Imprimir resultados
            print_results(predicted_class, confidence, class_names, prob_dict, metadata, args.model)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERRO: {e}")
        print("\n💡 Dica: Execute o pipeline de treinamento primeiro:")
        print("  python main.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRO ao classificar imagem: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
