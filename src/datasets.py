# -*- coding: utf-8 -*-
"""
Datasets com Lazy Loading
Carrega imagens do disco sob demanda para economizar memória
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Callable
from PIL import Image, ImageOps
import cv2

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


class LazyImageDataset(Dataset):
    """
    Dataset que carrega imagens do disco sob demanda (lazy loading)
    Economiza memória RAM ao não manter todas as imagens na memória
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        img_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        cache_size: int = 0
    ):
        """
        Args:
            image_paths: Lista de caminhos das imagens
            labels: Lista de labels correspondentes
            img_size: Tamanho para redimensionar as imagens
            transform: Transformações a aplicar (torchvision)
            cache_size: Número de imagens a manter em cache (0 = sem cache)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.transform = transform
        self.cache_size = cache_size
        self._cache = {}
        self._cache_order = []

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> np.ndarray:
        """
        Carrega e pré-processa uma imagem

        Args:
            path: Caminho da imagem

        Returns:
            img: Imagem como array numpy (RGB, uint8)
        """
        try:
            with Image.open(path) as pil_img:
                # Corrigir orientação EXIF
                try:
                    pil_img = ImageOps.exif_transpose(pil_img)
                except Exception:
                    pass

                # Converter para RGB
                if pil_img.mode == 'RGBA':
                    rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
                    rgb_img.paste(pil_img, mask=pil_img.split()[3])
                    pil_img = rgb_img
                elif pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                # Converter para numpy
                img = np.array(pil_img)

                # Redimensionar
                img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)

                return img

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar {path}: {e}")

    def _get_from_cache(self, idx: int) -> Optional[np.ndarray]:
        """Obtém imagem do cache se disponível"""
        return self._cache.get(idx)

    def _add_to_cache(self, idx: int, img: np.ndarray):
        """Adiciona imagem ao cache com política LRU"""
        if self.cache_size <= 0:
            return

        # Remover item mais antigo se cache cheio
        if len(self._cache) >= self.cache_size:
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]

        self._cache[idx] = img
        self._cache_order.append(idx)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Carrega uma imagem e seu label

        Args:
            idx: Índice da imagem

        Returns:
            image: Imagem (transformada se transform fornecido)
            label: Label da imagem
        """
        # Tentar obter do cache
        img = self._get_from_cache(idx)

        if img is None:
            # Carregar do disco
            img = self._load_image(self.image_paths[idx])
            self._add_to_cache(idx, img)

        label = self.labels[idx]

        # Aplicar transformações
        if self.transform:
            img = self.transform(img)

        return img, label

    def clear_cache(self):
        """Limpa o cache de imagens"""
        self._cache.clear()
        self._cache_order.clear()


class LazyClassicDataset:
    """
    Dataset para pipeline clássico com lazy loading
    Processa imagens em batches para economizar memória
    """

    def __init__(
        self,
        directory: Path,
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 100
    ):
        """
        Args:
            directory: Diretório com subpastas por classe
            img_size: Tamanho para redimensionar as imagens
            batch_size: Tamanho do batch para processamento
        """
        self.directory = Path(directory)
        self.img_size = img_size
        self.batch_size = batch_size

        # Descobrir classes e imagens
        self.class_names = sorted([
            d.name for d in self.directory.iterdir() if d.is_dir()
        ])
        self.image_paths = []
        self.labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.directory / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)

        print(f"Encontradas {len(self.image_paths)} imagens em {len(self.class_names)} classes")

    def _load_single_image(self, path: Path) -> Optional[np.ndarray]:
        """Carrega uma única imagem"""
        try:
            with Image.open(path) as pil_img:
                try:
                    pil_img = ImageOps.exif_transpose(pil_img)
                except Exception:
                    pass

                if pil_img.mode == 'RGBA':
                    rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
                    rgb_img.paste(pil_img, mask=pil_img.split()[3])
                    pil_img = rgb_img
                elif pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                img = np.array(pil_img)

                if img.shape[0] < 32 or img.shape[1] < 32:
                    return None

                img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
                return img

        except Exception:
            return None

    def load_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega todas as imagens (use com cuidado!)

        Returns:
            images: Array de imagens
            labels: Array de labels
        """
        images = []
        labels = []

        for i, (path, label) in enumerate(zip(self.image_paths, self.labels)):
            img = self._load_single_image(path)
            if img is not None:
                images.append(img)
                labels.append(label)

            if (i + 1) % 100 == 0:
                print(f"Carregado: {i + 1}/{len(self.image_paths)}", end='\r')

        print(f"\nTotal carregado: {len(images)} imagens")
        return np.array(images), np.array(labels)

    def iter_batches(self, shuffle: bool = True):
        """
        Itera sobre os dados em batches

        Args:
            shuffle: Se True, embaralha os dados

        Yields:
            batch_images: Batch de imagens
            batch_labels: Batch de labels
        """
        indices = list(range(len(self.image_paths)))
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]

            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                img = self._load_single_image(self.image_paths[idx])
                if img is not None:
                    batch_images.append(img)
                    batch_labels.append(self.labels[idx])

            if batch_images:
                yield np.array(batch_images), np.array(batch_labels)

    def get_flattened_batches(self, normalize: bool = True):
        """
        Itera retornando batches flatten para modelos clássicos

        Args:
            normalize: Se True, normaliza para [0, 1]

        Yields:
            batch_flat: Batch de imagens flatten
            batch_labels: Batch de labels
        """
        for images, labels in self.iter_batches():
            flat = images.reshape(images.shape[0], -1)
            if normalize:
                flat = flat / 255.0
            yield flat, labels


def create_lazy_dataloaders(
    train_dir: Path,
    test_dir: Path,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    transform_train=None,
    transform_test=None,
    cache_size: int = 100,
    num_workers: int = 0
) -> Tuple:
    """
    Cria DataLoaders com lazy loading

    Args:
        train_dir: Diretório de treinamento
        test_dir: Diretório de teste
        img_size: Tamanho das imagens
        batch_size: Tamanho do batch
        transform_train: Transformações para treino
        transform_test: Transformações para teste
        cache_size: Tamanho do cache por dataset
        num_workers: Número de workers para carregamento

    Returns:
        train_loader: DataLoader de treinamento
        test_loader: DataLoader de teste
        class_names: Nomes das classes
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch é necessário para usar DataLoaders")

    # Coletar caminhos de imagens
    train_paths = []
    train_labels = []
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    for class_idx, class_name in enumerate(class_names):
        class_dir = train_dir / class_name
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                train_paths.append(img_path)
                train_labels.append(class_idx)

    test_paths = []
    test_labels = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = test_dir / class_name
        if class_dir.exists():
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    test_paths.append(img_path)
                    test_labels.append(class_idx)

    print(f"Treinamento: {len(train_paths)} imagens")
    print(f"Teste: {len(test_paths)} imagens")
    print(f"Classes: {class_names}")

    # Criar datasets
    train_dataset = LazyImageDataset(
        train_paths, train_labels, img_size,
        transform=transform_train, cache_size=cache_size
    )
    test_dataset = LazyImageDataset(
        test_paths, test_labels, img_size,
        transform=transform_test, cache_size=cache_size
    )

    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, class_names
