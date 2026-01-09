# -*- coding: utf-8 -*-
"""
Pipeline Deep Learning - Classificação de Imagens
Aplica modelos de deep learning com e sem transfer learning
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from ..config import (
    MODELS_DIR, RESULTS_DIR, FIGURES_DIR, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, USE_AUGMENTATION, AUGMENTATION_PARAMS, IMG_SIZE
)
from ..utils import (
    load_images_from_directory, calculate_metrics, plot_confusion_matrix,
    save_results_table, print_results_summary, setup_device
)
from ..models import SimpleCNN


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


def sample_hyperparameters(param_space):
    """
    Amostra aleatoriamente hiperparâmetros do espaço definido

    Args:
        param_space: Dicionário com espaço de hiperparâmetros

    Returns:
        params: Dicionário com hiperparâmetros amostrados
    """
    params = {}
    for key, value in param_space.items():
        if isinstance(value, tuple) and len(value) == 2:
            if isinstance(value[0], float):
                # Log-uniform para learning rate
                if key == 'learning_rate':
                    log_low, log_high = np.log10(value[0]), np.log10(value[1])
                    params[key] = 10 ** np.random.uniform(log_low, log_high)
                else:
                    params[key] = np.random.uniform(value[0], value[1])
            elif isinstance(value[0], int):
                params[key] = np.random.randint(value[0], value[1] + 1)
        elif isinstance(value, list):
            params[key] = random.choice(value)
        else:
            params[key] = value
    return params


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

        # Guardar dados brutos para Random Search
        self.X_train_raw = X_train
        self.y_train_raw = y_train
        self.X_test_raw = X_test
        self.y_test_raw = y_test

        # Criar datasets
        train_transform = self.get_transforms(is_training=True)
        test_transform = self.get_transforms(is_training=False)

        train_dataset = ImageDataset(X_train, y_train, transform=train_transform)
        test_dataset = ImageDataset(X_test, y_test, transform=test_transform)

        # Criar dataloaders com batch size padrão
        self.train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        self.num_classes = len(self.class_names)
        print(f"Número de classes: {self.num_classes}")

    def create_dataloaders(self, batch_size, val_split=0.2):
        """
        Cria dataloaders com batch size específico e split de validação
        """
        train_transform = self.get_transforms(is_training=True)
        test_transform = self.get_transforms(is_training=False)

        full_train_dataset = ImageDataset(
            self.X_train_raw, self.y_train_raw, transform=train_transform
        )

        # Split treino/validação
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Dataset de validação usa transformações de teste
        val_dataset_proper = ImageDataset(
            self.X_train_raw, self.y_train_raw, transform=test_transform
        )
        val_indices = val_dataset.indices
        val_dataset = torch.utils.data.Subset(val_dataset_proper, val_indices)

        test_dataset = ImageDataset(
            self.X_test_raw, self.y_test_raw, transform=test_transform
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        return train_loader, val_loader, test_loader

    def train_single_config(self, model, train_loader, val_loader, epochs, learning_rate, patience=5):
        """
        Treina modelo com uma configuração específica
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=False
        )

        best_val_acc = 0.0
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validação
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        return best_val_acc, model

    def train_model(self, model, train_loader, epochs, learning_rate, model_name):
        """
        Treina um modelo
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

    def train_simple_cnn(self, use_random_search=True, n_iter=10, final_epochs=EPOCHS):
        """
        Treina CNN simples sem transfer learning com Random Search
        """
        print("\n" + "="*80)
        print("TREINANDO MODELO: CNN Simples (sem Transfer Learning)")
        print("="*80)

        best_params = {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'dropout_rate': 0.5,
            'hidden_units': 512
        }

        if use_random_search:
            print(f"\nExecutando Random Search ({n_iter} iterações)...")
            param_space = {
                'learning_rate': (0.0001, 0.01),
                'batch_size': [16, 32, 64],
                'dropout_rate': (0.3, 0.7),
                'hidden_units': [256, 512, 1024]
            }

            best_val_acc = 0.0
            search_epochs = min(15, final_epochs)

            for i in range(n_iter):
                params = sample_hyperparameters(param_space)
                print(f"\n  Iteração {i+1}/{n_iter}: lr={params['learning_rate']:.6f}, "
                      f"batch={params['batch_size']}, dropout={params['dropout_rate']:.2f}, "
                      f"hidden={params['hidden_units']}")

                train_loader, val_loader, _ = self.create_dataloaders(
                    params['batch_size'], val_split=0.2
                )

                model = SimpleCNN(
                    self.num_classes,
                    dropout_rate=params['dropout_rate'],
                    hidden_units=params['hidden_units']
                )

                val_acc, _ = self.train_single_config(
                    model, train_loader, val_loader, search_epochs,
                    params['learning_rate'], patience=5
                )

                print(f"    Val Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = params.copy()

            print(f"\nMelhores hiperparâmetros encontrados:")
            print(f"  Learning Rate: {best_params['learning_rate']:.6f}")
            print(f"  Batch Size: {best_params['batch_size']}")
            print(f"  Dropout Rate: {best_params['dropout_rate']:.2f}")
            print(f"  Hidden Units: {best_params['hidden_units']}")
            print(f"  Melhor Val Acc: {best_val_acc:.4f}")

        # Treinamento final
        print(f"\nTreinamento final com {final_epochs} épocas...")
        model = SimpleCNN(
            self.num_classes,
            dropout_rate=best_params['dropout_rate'],
            hidden_units=best_params['hidden_units']
        )
        print(f"Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")

        train_transform = self.get_transforms(is_training=True)
        test_transform = self.get_transforms(is_training=False)
        train_dataset = ImageDataset(self.X_train_raw, self.y_train_raw, transform=train_transform)
        test_dataset = ImageDataset(self.X_test_raw, self.y_test_raw, transform=test_transform)

        final_train_loader = DataLoader(
            train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=0
        )
        final_test_loader = DataLoader(
            test_dataset, batch_size=best_params['batch_size'], shuffle=False, num_workers=0
        )

        model, history = self.train_model(
            model, final_train_loader, final_epochs, best_params['learning_rate'], "SimpleCNN"
        )

        metrics, y_true, y_pred, cm = self.evaluate_model(model, final_test_loader)

        print(f"\nAcurácia - Teste: {metrics['accuracy']:.4f}")
        print(f"Precisão - Teste: {metrics['precision']:.4f}")
        print(f"Recall - Teste: {metrics['recall']:.4f}")
        print(f"F1-Score - Teste: {metrics['f1_score']:.4f}")

        model_path = MODELS_DIR / 'simple_cnn.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Modelo salvo em: {model_path}")

        cm_path = FIGURES_DIR / 'simple_cnn_confusion_matrix.png'
        plot_confusion_matrix(cm, self.class_names,
                            'CNN Simples - Matriz de Confusão', cm_path)

        self.results.append({
            'Modelo': 'CNN Simples',
            'Acurácia': f"{metrics['accuracy']:.4f}",
            'Precisão': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'Transfer Learning': 'Não',
            'Data Augmentation': 'Sim' if USE_AUGMENTATION else 'Não',
            'Otimização': f'Random Search ({n_iter} iter)' if use_random_search else 'Padrão'
        })

    def create_resnet_model(self, unfreeze_layers=0):
        """
        Cria modelo ResNet50 com transfer learning
        """
        model = models.resnet50(weights='IMAGENET1K_V2')

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)

        for param in model.fc.parameters():
            param.requires_grad = True

        if unfreeze_layers > 0:
            layers = [model.layer4, model.layer3, model.layer2, model.layer1]
            for i, layer in enumerate(layers[:unfreeze_layers]):
                for param in layer.parameters():
                    param.requires_grad = True

        return model

    def train_resnet_transfer(self, use_random_search=True, n_iter=10, final_epochs=EPOCHS):
        """
        Treina ResNet com transfer learning e Random Search
        """
        print("\n" + "="*80)
        print("TREINANDO MODELO: ResNet50 (com Transfer Learning)")
        print("="*80)

        best_params = {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'unfreeze_layers': 0
        }

        if use_random_search:
            print(f"\nExecutando Random Search ({n_iter} iterações)...")
            param_space = {
                'learning_rate': (0.00001, 0.001),
                'batch_size': [16, 32, 64],
                'unfreeze_layers': [0, 1, 2]
            }

            best_val_acc = 0.0
            search_epochs = min(10, final_epochs)

            for i in range(n_iter):
                params = sample_hyperparameters(param_space)
                print(f"\n  Iteração {i+1}/{n_iter}: lr={params['learning_rate']:.6f}, "
                      f"batch={params['batch_size']}, unfreeze={params['unfreeze_layers']}")

                train_loader, val_loader, _ = self.create_dataloaders(
                    params['batch_size'], val_split=0.2
                )

                model = self.create_resnet_model(unfreeze_layers=params['unfreeze_layers'])

                val_acc, _ = self.train_single_config(
                    model, train_loader, val_loader, search_epochs,
                    params['learning_rate'], patience=5
                )

                print(f"    Val Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = params.copy()

            print(f"\nMelhores hiperparâmetros encontrados:")
            print(f"  Learning Rate: {best_params['learning_rate']:.6f}")
            print(f"  Batch Size: {best_params['batch_size']}")
            print(f"  Unfreeze Layers: {best_params['unfreeze_layers']}")
            print(f"  Melhor Val Acc: {best_val_acc:.4f}")

        # Treinamento final
        print(f"\nTreinamento final com {final_epochs} épocas...")
        model = self.create_resnet_model(unfreeze_layers=best_params['unfreeze_layers'])

        print(f"Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")
        print(f"Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        train_transform = self.get_transforms(is_training=True)
        test_transform = self.get_transforms(is_training=False)
        train_dataset = ImageDataset(self.X_train_raw, self.y_train_raw, transform=train_transform)
        test_dataset = ImageDataset(self.X_test_raw, self.y_test_raw, transform=test_transform)

        final_train_loader = DataLoader(
            train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=0
        )
        final_test_loader = DataLoader(
            test_dataset, batch_size=best_params['batch_size'], shuffle=False, num_workers=0
        )

        model, history = self.train_model(
            model, final_train_loader, final_epochs, best_params['learning_rate'], "ResNet50"
        )

        metrics, y_true, y_pred, cm = self.evaluate_model(model, final_test_loader)

        print(f"\nAcurácia - Teste: {metrics['accuracy']:.4f}")
        print(f"Precisão - Teste: {metrics['precision']:.4f}")
        print(f"Recall - Teste: {metrics['recall']:.4f}")
        print(f"F1-Score - Teste: {metrics['f1_score']:.4f}")

        model_path = MODELS_DIR / 'resnet50_transfer.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Modelo salvo em: {model_path}")

        cm_path = FIGURES_DIR / 'resnet50_confusion_matrix.png'
        plot_confusion_matrix(cm, self.class_names,
                            'ResNet50 Transfer Learning - Matriz de Confusão', cm_path)

        self.results.append({
            'Modelo': 'ResNet50',
            'Acurácia': f"{metrics['accuracy']:.4f}",
            'Precisão': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'Transfer Learning': 'Sim',
            'Data Augmentation': 'Sim' if USE_AUGMENTATION else 'Não',
            'Otimização': f'Random Search ({n_iter} iter)' if use_random_search else 'Padrão'
        })

    def save_results(self):
        """
        Salva resultados finais
        """
        results_path = RESULTS_DIR / 'deep_learning_results.csv'
        save_results_table(self.results, results_path)
        print_results_summary(self.results)


def main():
    """
    Função principal para executar o pipeline de deep learning
    """
    from ..config import TRAIN_DIR, TEST_DIR, VAL_DIR, USE_GPU

    pipeline = DeepLearningPipeline(TRAIN_DIR, TEST_DIR, VAL_DIR, use_gpu=USE_GPU)
    pipeline.load_data()

    pipeline.train_simple_cnn(use_random_search=True, n_iter=10, final_epochs=EPOCHS)
    pipeline.train_resnet_transfer(use_random_search=True, n_iter=10, final_epochs=EPOCHS)

    pipeline.save_results()

    print("\nPipeline de deep learning concluído!")


if __name__ == "__main__":
    main()
