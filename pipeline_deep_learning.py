# -*- coding: utf-8 -*-
"""
Pipeline Deep Learning - Classificação de Imagens
Aplica modelos de deep learning com e sem transfer learning
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    MODELS_DIR, RESULTS_DIR, FIGURES_DIR, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, USE_AUGMENTATION, AUGMENTATION_PARAMS, IMG_SIZE
)
from utils import (
    load_images_from_directory, calculate_metrics, plot_confusion_matrix,
    save_results_table, print_results_summary, setup_device
)

class ImageDataset(Dataset):
    """
    Dataset personalizado para imagens
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SimpleCNN(nn.Module):
    """
    CNN simples sem transfer learning
    """
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Torna o cálculo dinâmico
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive_pool(x)  # Garante tamanho fixo independente da entrada
        x = x.view(-1, 128 * 7 * 7)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepLearningPipeline:
    """
    Pipeline de deep learning para classificação de imagens
    """
    
    def __init__(self, train_dir, test_dir, val_dir=None, use_gpu=True):
        """
        Inicializa o pipeline
        
        Args:
            train_dir: Diretório de treinamento
            test_dir: Diretório de teste
            val_dir: Diretório de validação (opcional)
            use_gpu: Se True, usa GPU se disponível
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.device = setup_device(use_gpu)
        self.results = []
        
    def get_transforms(self, is_training=False):
        """
        Retorna transformações de dados
        
        Args:
            is_training: Se True, aplica data augmentation
        
        Returns:
            transform: Transformações
        """
        if is_training and USE_AUGMENTATION:
            transform_list = [
                transforms.ToPILImage(),
                transforms.RandomRotation(AUGMENTATION_PARAMS['rotation_range']),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(AUGMENTATION_PARAMS['width_shift_range'],
                              AUGMENTATION_PARAMS['height_shift_range'])
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
            # Adicionar RandomHorizontalFlip apenas se estiver habilitado
            if AUGMENTATION_PARAMS.get('horizontal_flip', False):
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            transform = transforms.Compose(transform_list)
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        return transform
    
    def load_data(self):
        """
        Carrega e prepara os dados
        """
        print("Carregando dados de treinamento...")
        X_train, y_train, self.class_names = load_images_from_directory(
            self.train_dir, img_size=IMG_SIZE
        )
        
        print("Carregando dados de teste...")
        X_test, y_test, _ = load_images_from_directory(
            self.test_dir, img_size=IMG_SIZE
        )
        
        print(f"Classes encontradas: {self.class_names}")
        print(f"Treinamento: {X_train.shape[0]} amostras")
        print(f"Teste: {X_test.shape[0]} amostras")
        print(f"Tamanho das imagens: {X_train.shape[1:]} pixels")
        print(f"Canais: {X_train.shape[3] if len(X_train.shape) > 3 else 1}")
        
        # Criar datasets
        train_transform = self.get_transforms(is_training=True)
        test_transform = self.get_transforms(is_training=False)
        
        train_dataset = ImageDataset(X_train, y_train, transform=train_transform)
        test_dataset = ImageDataset(X_test, y_test, transform=test_transform)
        
        # Criar dataloaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        
        self.num_classes = len(self.class_names)
        print(f"Número de classes: {self.num_classes}")
    
    def train_model(self, model, train_loader, epochs, learning_rate, model_name):
        """
        Treina um modelo
        
        Args:
            model: Modelo a ser treinado
            train_loader: DataLoader de treinamento
            epochs: Número de épocas
            learning_rate: Taxa de aprendizado
            model_name: Nome do modelo
        
        Returns:
            model: Modelo treinado
            history: Histórico de treinamento
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        history = {'train_loss': [], 'train_acc': []}
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            scheduler.step(epoch_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Época [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        return model, history
    
    def evaluate_model(self, model, test_loader):
        """
        Avalia um modelo
        
        Args:
            model: Modelo a ser avaliado
            test_loader: DataLoader de teste
        
        Returns:
            metrics: Métricas de avaliação
            y_true: Labels verdadeiros
            y_pred: Labels preditos
        """
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        metrics, report, cm = calculate_metrics(
            np.array(y_true), np.array(y_pred), self.class_names
        )
        
        return metrics, y_true, y_pred, cm
    
    def train_simple_cnn(self, epochs=EPOCHS, learning_rate=LEARNING_RATE):
        """
        Treina CNN simples sem transfer learning
        """
        print("\n" + "="*80)
        print("TREINANDO MODELO: CNN Simples (sem Transfer Learning)")
        print("="*80)
        
        model = SimpleCNN(self.num_classes)
        print(f"Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")
        
        model, history = self.train_model(
            model, self.train_loader, epochs, learning_rate, "SimpleCNN"
        )
        
        # Avaliar
        metrics, y_true, y_pred, cm = self.evaluate_model(model, self.test_loader)
        
        print(f"\nAcurácia - Teste: {metrics['accuracy']:.4f}")
        print(f"Precisão - Teste: {metrics['precision']:.4f}")
        print(f"Recall - Teste: {metrics['recall']:.4f}")
        print(f"F1-Score - Teste: {metrics['f1_score']:.4f}")
        
        # Salvar modelo
        model_path = os.path.join(MODELS_DIR, 'simple_cnn.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Modelo salvo em: {model_path}")
        
        # Plotar matriz de confusão
        cm_path = os.path.join(FIGURES_DIR, 'simple_cnn_confusion_matrix.png')
        plot_confusion_matrix(cm, self.class_names,
                            'CNN Simples - Matriz de Confusão', cm_path)
        
        # Adicionar aos resultados
        self.results.append({
            'Modelo': 'CNN Simples',
            'Acurácia': f"{metrics['accuracy']:.4f}",
            'Precisão': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'Transfer Learning': 'Não',
            'Data Augmentation': 'Sim' if USE_AUGMENTATION else 'Não'
        })
    
    def train_resnet_transfer(self, epochs=EPOCHS, learning_rate=LEARNING_RATE):
        """
        Treina ResNet com transfer learning
        """
        print("\n" + "="*80)
        print("TREINANDO MODELO: ResNet50 (com Transfer Learning)")
        print("="*80)
        
        # Carregar modelo pré-treinado
        model = models.resnet50(weights='IMAGENET1K_V2')
        
        # Congelar camadas convolucionais
        for param in model.parameters():
            param.requires_grad = False
        
        # Substituir camada final
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        
        # Descongelar última camada
        for param in model.fc.parameters():
            param.requires_grad = True
        
        print(f"Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")
        print(f"Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        model, history = self.train_model(
            model, self.train_loader, epochs, learning_rate, "ResNet50"
        )
        
        # Avaliar
        metrics, y_true, y_pred, cm = self.evaluate_model(model, self.test_loader)
        
        print(f"\nAcurácia - Teste: {metrics['accuracy']:.4f}")
        print(f"Precisão - Teste: {metrics['precision']:.4f}")
        print(f"Recall - Teste: {metrics['recall']:.4f}")
        print(f"F1-Score - Teste: {metrics['f1_score']:.4f}")
        
        # Salvar modelo
        model_path = os.path.join(MODELS_DIR, 'resnet50_transfer.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Modelo salvo em: {model_path}")
        
        # Plotar matriz de confusão
        cm_path = os.path.join(FIGURES_DIR, 'resnet50_confusion_matrix.png')
        plot_confusion_matrix(cm, self.class_names,
                            'ResNet50 Transfer Learning - Matriz de Confusão', cm_path)
        
        # Adicionar aos resultados
        self.results.append({
            'Modelo': 'ResNet50',
            'Acurácia': f"{metrics['accuracy']:.4f}",
            'Precisão': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'Transfer Learning': 'Sim',
            'Data Augmentation': 'Sim' if USE_AUGMENTATION else 'Não'
        })
    
    def save_results(self):
        """
        Salva resultados finais
        """
        results_path = os.path.join(RESULTS_DIR, 'deep_learning_results.csv')
        save_results_table(self.results, results_path)
        print_results_summary(self.results)

def main():
    """
    Função principal para executar o pipeline de deep learning
    """
    from config import TRAIN_DIR, TEST_DIR, VAL_DIR, USE_GPU
    
    pipeline = DeepLearningPipeline(TRAIN_DIR, TEST_DIR, VAL_DIR, use_gpu=USE_GPU)
    pipeline.load_data()
    
    # Treinar modelos
    pipeline.train_simple_cnn(epochs=EPOCHS, learning_rate=LEARNING_RATE)
    pipeline.train_resnet_transfer(epochs=EPOCHS, learning_rate=LEARNING_RATE)
    
    # Salvar resultados
    pipeline.save_results()
    
    print("\nPipeline de deep learning concluído!")

if __name__ == "__main__":
    main()

