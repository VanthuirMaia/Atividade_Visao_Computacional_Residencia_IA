# -*- coding: utf-8 -*-
"""
Funções utilitárias para o projeto
"""

import numpy as np
from datetime import datetime
from pathlib import Path
import cv2
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def setup_device(use_gpu=True):
    """
    Configura o dispositivo (CPU ou GPU) para treinamento

    Args:
        use_gpu: Se True, tenta usar GPU, caso contrário usa CPU

    Returns:
        device: Dispositivo configurado (torch.device)
    """
    # Sempre tentar importar torch primeiro
    try:
        import torch
    except ImportError:
        print("❌ PyTorch não instalado!")
        print("   Execute: pip install torch torchvision")
        raise ImportError("PyTorch não está instalado. Instale com: pip install torch torchvision")
    
    if use_gpu:
        # Verificar se CUDA está disponível
        if not torch.cuda.is_available():
            print(f"\n{'='*60}")
            print("CONFIGURAÇÃO DE DISPOSITIVO")
            print(f"{'='*60}")
            print("⚠️  GPU solicitada mas não disponível!")
            print(f"   torch.cuda.is_available() = {torch.cuda.is_available()}")
            print("   Possíveis razões:")
            print("     - PyTorch instalado sem suporte CUDA (versão CPU-only)")
            print("     - GPU não compatível ou drivers não instalados")
            print("     - CUDA não está instalado no sistema")
            print("     - GPU está ocupada por outro processo")
            
            # Diagnóstico adicional
            try:
                print(f"\n   Diagnóstico PyTorch:")
                print(f"     Versão PyTorch: {torch.__version__}")
                print(f"     Versão CUDA compilada: {torch.version.cuda if torch.version.cuda else 'N/A'}")
                print(f"     cuDNN disponível: {torch.backends.cudnn.is_available()}")
                if torch.backends.cudnn.is_available():
                    print(f"     Versão cuDNN: {torch.backends.cudnn.version()}")
            except Exception as e:
                print(f"     Erro ao obter informações: {e}")
            
            print(f"\n   Usando CPU como fallback...")
            print(f"{'='*60}\n")
            return torch.device('cpu')
        
        # GPU está disponível
        try:
            device = torch.device('cuda:0')  # Especificar índice explícito
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            
            print(f"\n{'='*60}")
            print("CONFIGURAÇÃO DE DISPOSITIVO")
            print(f"{'='*60}")
            print(f"✅ GPU DETECTADA E CONFIGURADA!")
            print(f"   GPU: {gpu_name}")
            print(f"   Dispositivo: {device}")
            print(f"   Número de GPUs disponíveis: {gpu_count}")
            if gpu_count > 1:
                print(f"   Usando GPU 0 de {gpu_count} disponíveis")
            
            # Mostrar informações de memória GPU
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                cached = torch.cuda.memory_reserved(0) / (1024**3)
                free_mem = total_mem - cached
                
                print(f"\n   Memória GPU:")
                print(f"     Total: {total_mem:.2f} GB")
                print(f"     Livre: {free_mem:.2f} GB")
                print(f"     Reservada: {cached:.2f} GB")
                print(f"     Alocada: {allocated:.2f} GB")
                
                if free_mem < 1.0:
                    print(f"\n   ⚠️  AVISO: Pouca memória GPU disponível!")
                    print(f"   Considere reduzir BATCH_SIZE em src/config.py")
            except Exception as e:
                print(f"   [INFO] Não foi possível obter informações de memória: {e}")
            
            print(f"{'='*60}\n")
            
            # Teste rápido para confirmar que a GPU está funcionando
            # IMPORTANTE: Se teste falhar, continuar com GPU mesmo assim
            # O teste pode falhar por problemas temporários, mas GPU pode funcionar durante treinamento
            try:
                print(f"   Testando GPU com operação simples...")
                test_tensor = torch.randn(10, 10).to(device)
                result = test_tensor @ test_tensor.t()
                
                # Verificar se resultado realmente está na GPU
                if result.device.type != 'cuda':
                    print(f"   ⚠️  AVISO: Resultado não está na GPU! Está em: {result.device}")
                    print(f"   Isso é estranho, mas continuando com GPU mesmo assim...")
                else:
                    print(f"   ✅ Resultado confirmado na GPU: {result.device}")
                
                del test_tensor, result
                torch.cuda.empty_cache()
                print(f"   ✅ Teste de GPU bem-sucedido!")
                print(f"   ✅ Dispositivo GPU confirmado e funcionando!\n")
            except RuntimeError as e:
                error_msg = str(e).lower()
                if 'out of memory' in error_msg:
                    print(f"   ⚠️  AVISO: GPU sem memória disponível para teste")
                    print(f"   Isso pode ser normal se GPU estiver ocupada")
                    print(f"   ✅ Continuando com GPU (tentará usar durante treinamento)...\n")
                    # IMPORTANTE: Não retornar CPU - deixar tentar usar GPU durante treinamento
                else:
                    print(f"   ⚠️  RuntimeError no teste de GPU: {e}")
                    print(f"   Tipo de erro: RuntimeError")
                    print(f"   ✅ Continuando com GPU mesmo assim (pode funcionar durante treinamento)...\n")
                    # IMPORTANTE: Não retornar CPU automaticamente - pode ser problema temporário
            except Exception as e:
                print(f"   ⚠️  ERRO inesperado no teste de GPU: {type(e).__name__}: {e}")
                print(f"   ✅ Continuando com GPU mesmo assim (pode ser problema temporário)...\n")
                # IMPORTANTE: Não retornar CPU - deixar tentar usar GPU
            
            # SEMPRE retornar device GPU se CUDA está disponível (mesmo se teste falhar)
            return device
            
        except Exception as e:
            print(f"\n   ⚠️  ERRO ao configurar GPU: {type(e).__name__}: {e}")
            print(f"   Trocando para CPU como fallback...")
            print(f"{'='*60}\n")
            return torch.device('cpu')
    else:
        print(f"\n{'='*60}")
        print("CONFIGURAÇÃO DE DISPOSITIVO")
        print(f"{'='*60}")
        print("ℹ️  Usando CPU (configurado manualmente)")
        print(f"{'='*60}\n")
        try:
            import torch
            return torch.device('cpu')
        except ImportError:
            # Se PyTorch não estiver instalado, retornar string como fallback
            return 'cpu'


def load_images_from_directory(directory, img_size=(224, 224)):
    """
    Carrega imagens de um diretório com padronização completa

    Padronizações aplicadas:
    - Suporta múltiplos formatos (JPG, JPEG, PNG)
    - Converte para RGB (3 canais)
    - Remove transparência (alpha channel)
    - Corrige orientação EXIF
    - Redimensiona para tamanho padrão
    - Valida e trata imagens corrompidas

    Args:
        directory: Caminho do diretório
        img_size: Tamanho para redimensionar as imagens

    Returns:
        images: Array de imagens padronizadas
        labels: Array de labels
        class_names: Lista de nomes das classes
    """
    # Converter para Path se necessário
    directory = Path(directory)

    images = []
    labels = []
    class_names = sorted([d.name for d in directory.iterdir() if d.is_dir()])

    # Estatísticas para relatório
    stats = {
        'total': 0,
        'loaded': 0,
        'errors': 0,
        'formats': {},
        'grayscale_converted': 0,
        'alpha_removed': 0,
        'exif_corrected': 0
    }

    for class_idx, class_name in enumerate(class_names):
        class_dir = directory / class_name
        if not class_dir.is_dir():
            continue

        for img_path in class_dir.iterdir():
            stats['total'] += 1

            # Verificar se é arquivo de imagem
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                continue

            try:
                # Usar PIL para melhor tratamento de formatos e EXIF
                with Image.open(img_path) as pil_img:
                    # Detectar formato
                    file_ext = img_path.suffix.lower()
                    stats['formats'][file_ext] = stats['formats'].get(file_ext, 0) + 1

                    # Corrigir orientação EXIF (importante para arte)
                    try:
                        pil_img = ImageOps.exif_transpose(pil_img)
                        stats['exif_corrected'] += 1
                    except Exception:
                        pass  # Sem dados EXIF ou erro ao processar

                    # Converter para RGB (remove alpha channel, converte grayscale)
                    if pil_img.mode == 'RGBA':
                        # Criar fundo branco para transparência
                        rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
                        rgb_img.paste(pil_img, mask=pil_img.split()[3])
                        pil_img = rgb_img
                        stats['alpha_removed'] += 1
                    elif pil_img.mode == 'L':  # Grayscale
                        pil_img = pil_img.convert('RGB')
                        stats['grayscale_converted'] += 1
                    elif pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')

                    # Converter PIL para numpy array
                    img = np.array(pil_img)

                    # Validar dimensões mínimas
                    if img.shape[0] < 32 or img.shape[1] < 32:
                        print(f"AVISO: Imagem muito pequena ignorada: {img_path} ({img.shape})")
                        stats['errors'] += 1
                        continue

                    # Redimensionar
                    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

                    # Validar resultado final
                    if img.shape == (*img_size, 3):
                        images.append(img)
                        labels.append(class_idx)
                        stats['loaded'] += 1
                    else:
                        print(f"ERRO: Formato inválido após processamento: {img_path} ({img.shape})")
                        stats['errors'] += 1

            except Exception as e:
                print(f"ERRO ao carregar {img_path}: {e}")
                stats['errors'] += 1

    # Relatório de estatísticas
    print(f"\n{'='*60}")
    print("ESTATÍSTICAS DE CARREGAMENTO DE IMAGENS")
    print(f"{'='*60}")
    print(f"Total de arquivos processados: {stats['total']}")
    print(f"Imagens carregadas com sucesso: {stats['loaded']}")
    print(f"Erros encontrados: {stats['errors']}")
    print(f"\nFormatos encontrados:")
    for fmt, count in sorted(stats['formats'].items()):
        print(f"  {fmt}: {count}")
    if stats['grayscale_converted'] > 0:
        print(f"\nImagens em escala de cinza convertidas: {stats['grayscale_converted']}")
    if stats['alpha_removed'] > 0:
        print(f"Canais alpha removidos: {stats['alpha_removed']}")
    if stats['exif_corrected'] > 0:
        print(f"Orientações EXIF corrigidas: {stats['exif_corrected']}")
    print(f"{'='*60}\n")

    if len(images) == 0:
        raise ValueError(f"Nenhuma imagem válida foi carregada de {directory}")

    # Converter para arrays numpy
    images_array = np.array(images)
    labels_array = np.array(labels)
    
    # Verificar quantas classes únicas foram carregadas
    unique_labels = np.unique(labels_array)
    num_classes_loaded = len(unique_labels)
    
    # Estatísticas por classe
    print(f"\nDistribuição de classes carregadas:")
    for label_idx in unique_labels:
        count = np.sum(labels_array == label_idx)
        class_name = class_names[label_idx] if label_idx < len(class_names) else f"Classe {label_idx}"
        print(f"  {class_name} (label {label_idx}): {count} imagens")
    
    # Validar que temos pelo menos 2 classes
    if num_classes_loaded < 2:
        available_classes = [class_names[i] for i in unique_labels] if unique_labels.size > 0 else []
        missing_classes = [class_names[i] for i in range(len(class_names)) if i not in unique_labels]
        
        error_msg = (
            f"\nERRO: Apenas {num_classes_loaded} classe(s) foi(ram) carregada(s), "
            f"mas são necessárias pelo menos 2 classes para classificação.\n"
            f"Classes disponíveis no diretório: {class_names}\n"
            f"Classes com imagens válidas: {available_classes}\n"
        )
        if missing_classes:
            error_msg += f"Classes sem imagens válidas: {missing_classes}\n"
        error_msg += (
            f"\nPossíveis causas:\n"
            f"1. Apenas uma classe tem imagens válidas no diretório {directory}\n"
            f"2. Todas as imagens de outras classes foram rejeitadas (muito pequenas, corrompidas, etc.)\n"
            f"3. Estrutura de diretórios incorreta\n"
            f"\nVerifique:\n"
            f"- Se há subpastas com nomes de classes em {directory}\n"
            f"- Se cada subpasta contém imagens válidas (JPG, PNG, JPEG)\n"
            f"- Se as imagens têm tamanho mínimo de 32x32 pixels"
        )
        raise ValueError(error_msg)
    
    # Verificar se temos imagens suficientes por classe (pelo menos 1)
    for label_idx in unique_labels:
        count = np.sum(labels_array == label_idx)
        if count == 0:
            class_name = class_names[label_idx] if label_idx < len(class_names) else f"Classe {label_idx}"
            raise ValueError(
                f"Classe '{class_name}' (label {label_idx}) não tem imagens válidas após processamento."
            )
    
    return images_array, labels_array, class_names


def preprocess_images_classic(images):
    """
    Pré-processa imagens para modelos clássicos (flatten)

    Args:
        images: Array de imagens

    Returns:
        processed: Imagens processadas (flattened)
    """
    n_samples = images.shape[0]
    return images.reshape(n_samples, -1) / 255.0


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calcula métricas de classificação

    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        class_names: Nomes das classes

    Returns:
        metrics: Dicionário com métricas
        report: Relatório de classificação
        cm: Matriz de confusão
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }

    return metrics, report, cm


def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """
    Plota matriz de confusão

    Args:
        cm: Matriz de confusão
        class_names: Nomes das classes
        title: Título do gráfico
        save_path: Caminho para salvar a figura
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results_table(results, save_path):
    """
    Salva tabela de resultados em CSV

    Args:
        results: Lista de dicionários com resultados
        save_path: Caminho para salvar o CSV
    """
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Resultados salvos em: {save_path}")


def print_results_summary(results):
    """
    Imprime resumo dos resultados

    Args:
        results: Lista de dicionários com resultados
    """
    print("\n" + "="*80)
    print("RESUMO DOS RESULTADOS")
    print("="*80)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*80)
