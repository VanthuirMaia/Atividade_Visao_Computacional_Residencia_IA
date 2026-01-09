# -*- coding: utf-8 -*-
"""
Módulo de Gerenciamento de Memória
Implementa estratégias para evitar estouro de memória
"""

import gc
import os
import psutil
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import numpy as np

# Tentar importar torch (opcional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MemoryMonitor:
    """
    Monitor de uso de memória RAM e GPU
    """

    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        """
        Args:
            warning_threshold: Limite para aviso (padrão: 80%)
            critical_threshold: Limite crítico (padrão: 90%)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.process = psutil.Process(os.getpid())

    def get_ram_usage(self) -> Tuple[float, float, float]:
        """
        Retorna uso de RAM

        Returns:
            used_gb: RAM usada em GB
            total_gb: RAM total em GB
            percent: Porcentagem de uso
        """
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024 ** 3)
        total_gb = memory.total / (1024 ** 3)
        percent = memory.percent / 100
        return used_gb, total_gb, percent

    def get_process_memory(self) -> float:
        """
        Retorna memória usada pelo processo atual em GB
        """
        return self.process.memory_info().rss / (1024 ** 3)

    def get_gpu_usage(self) -> Optional[Tuple[float, float, float]]:
        """
        Retorna uso de memória GPU (se disponível)

        Returns:
            used_gb: VRAM usada em GB
            total_gb: VRAM total em GB
            percent: Porcentagem de uso
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None

        try:
            used = torch.cuda.memory_allocated() / (1024 ** 3)
            cached = torch.cuda.memory_reserved() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            percent = (used + cached) / total if total > 0 else 0
            return used + cached, total, percent
        except Exception:
            return None

    def check_memory(self, context: str = "") -> bool:
        """
        Verifica uso de memória e emite alertas

        Args:
            context: Contexto para mensagem de log

        Returns:
            ok: True se memória está em níveis aceitáveis
        """
        used_gb, total_gb, percent = self.get_ram_usage()
        process_gb = self.get_process_memory()

        prefix = f"[{context}] " if context else ""

        if percent >= self.critical_threshold:
            warnings.warn(
                f"{prefix}CRÍTICO: Uso de RAM em {percent*100:.1f}% "
                f"({used_gb:.1f}/{total_gb:.1f} GB). "
                f"Processo usando {process_gb:.2f} GB."
            )
            return False
        elif percent >= self.warning_threshold:
            warnings.warn(
                f"{prefix}AVISO: Uso de RAM em {percent*100:.1f}% "
                f"({used_gb:.1f}/{total_gb:.1f} GB)."
            )

        # Verificar GPU
        gpu_usage = self.get_gpu_usage()
        if gpu_usage:
            gpu_used, gpu_total, gpu_percent = gpu_usage
            if gpu_percent >= self.critical_threshold:
                warnings.warn(
                    f"{prefix}CRÍTICO: Uso de VRAM em {gpu_percent*100:.1f}% "
                    f"({gpu_used:.1f}/{gpu_total:.1f} GB)."
                )
                return False

        return True

    def print_status(self):
        """
        Imprime status atual de memória
        """
        used_gb, total_gb, percent = self.get_ram_usage()
        process_gb = self.get_process_memory()

        print(f"\n{'='*50}")
        print("STATUS DE MEMÓRIA")
        print(f"{'='*50}")
        print(f"RAM Total: {total_gb:.1f} GB")
        print(f"RAM Usada: {used_gb:.1f} GB ({percent*100:.1f}%)")
        print(f"Processo Atual: {process_gb:.2f} GB")

        gpu_usage = self.get_gpu_usage()
        if gpu_usage:
            gpu_used, gpu_total, gpu_percent = gpu_usage
            print(f"VRAM Total: {gpu_total:.1f} GB")
            print(f"VRAM Usada: {gpu_used:.1f} GB ({gpu_percent*100:.1f}%)")

        print(f"{'='*50}\n")


def clear_memory(clear_gpu: bool = True):
    """
    Limpa memória não utilizada

    Args:
        clear_gpu: Se True, limpa também cache da GPU
    """
    # Garbage collection Python
    gc.collect()

    # Limpar cache GPU
    if clear_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_optimal_batch_size(
    base_batch_size: int,
    sample_memory_mb: float,
    available_memory_gb: Optional[float] = None,
    safety_factor: float = 0.7
) -> int:
    """
    Calcula batch size ótimo baseado na memória disponível

    Args:
        base_batch_size: Batch size desejado
        sample_memory_mb: Memória estimada por amostra em MB
        available_memory_gb: Memória disponível (auto-detecta se None)
        safety_factor: Fator de segurança (usar apenas X% da memória)

    Returns:
        optimal_batch_size: Batch size ajustado
    """
    if available_memory_gb is None:
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024 ** 3)

    available_mb = available_memory_gb * 1024 * safety_factor
    max_batch_size = int(available_mb / sample_memory_mb)

    # Limitar ao batch size base ou ao máximo calculado
    optimal = min(base_batch_size, max(1, max_batch_size))

    # Arredondar para potência de 2 mais próxima (mais eficiente)
    power_of_2 = 1
    while power_of_2 * 2 <= optimal:
        power_of_2 *= 2

    return power_of_2


class AdaptiveBatchSize:
    """
    Gerenciador de batch size adaptativo
    Reduz automaticamente o batch size em caso de erro de memória
    """

    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 4,
        reduction_factor: float = 0.5
    ):
        """
        Args:
            initial_batch_size: Batch size inicial
            min_batch_size: Batch size mínimo
            reduction_factor: Fator de redução em caso de erro
        """
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.reduction_factor = reduction_factor
        self.current_batch_size = initial_batch_size
        self.reduction_count = 0

    def get_batch_size(self) -> int:
        """Retorna batch size atual"""
        return self.current_batch_size

    def reduce(self) -> bool:
        """
        Reduz batch size

        Returns:
            success: True se conseguiu reduzir, False se já está no mínimo
        """
        new_size = int(self.current_batch_size * self.reduction_factor)
        new_size = max(new_size, self.min_batch_size)

        if new_size < self.current_batch_size:
            self.current_batch_size = new_size
            self.reduction_count += 1
            print(f"Batch size reduzido para {self.current_batch_size}")
            clear_memory()
            return True
        return False

    def reset(self):
        """Reseta para batch size inicial"""
        self.current_batch_size = self.initial_batch_size
        self.reduction_count = 0

    def execute_with_retry(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Executa função com retry automático reduzindo batch size

        Args:
            func: Função a executar
            *args: Argumentos posicionais
            max_retries: Número máximo de tentativas
            **kwargs: Argumentos nomeados

        Returns:
            result: Resultado da função
        """
        for attempt in range(max_retries):
            try:
                clear_memory()
                return func(*args, **kwargs)
            except (RuntimeError, MemoryError) as e:
                error_msg = str(e).lower()
                if 'out of memory' in error_msg or 'cuda' in error_msg:
                    if not self.reduce():
                        raise MemoryError(
                            f"Memória insuficiente mesmo com batch size mínimo "
                            f"({self.min_batch_size})"
                        )
                    print(f"Tentativa {attempt + 1}/{max_retries} com batch size "
                          f"{self.current_batch_size}")
                else:
                    raise

        raise MemoryError(f"Falhou após {max_retries} tentativas")


class ChunkedDataProcessor:
    """
    Processador de dados em chunks para evitar carregar tudo na memória
    """

    def __init__(self, chunk_size: int = 100, memory_monitor: Optional[MemoryMonitor] = None):
        """
        Args:
            chunk_size: Tamanho do chunk
            memory_monitor: Monitor de memória (cria um se não fornecido)
        """
        self.chunk_size = chunk_size
        self.monitor = memory_monitor or MemoryMonitor()

    def process_in_chunks(
        self,
        data_paths: List[Path],
        process_func: Callable,
        **kwargs
    ) -> List:
        """
        Processa dados em chunks

        Args:
            data_paths: Lista de caminhos dos arquivos
            process_func: Função para processar cada arquivo
            **kwargs: Argumentos adicionais para process_func

        Returns:
            results: Lista de resultados
        """
        results = []
        total = len(data_paths)

        for i in range(0, total, self.chunk_size):
            chunk = data_paths[i:i + self.chunk_size]
            chunk_results = []

            for path in chunk:
                try:
                    result = process_func(path, **kwargs)
                    chunk_results.append(result)
                except Exception as e:
                    warnings.warn(f"Erro ao processar {path}: {e}")

            results.extend(chunk_results)

            # Verificar memória e limpar se necessário
            if not self.monitor.check_memory(f"Chunk {i//self.chunk_size + 1}"):
                clear_memory()

            # Progresso
            progress = min(i + self.chunk_size, total)
            print(f"Processado: {progress}/{total} ({progress/total*100:.1f}%)", end='\r')

        print()  # Nova linha após progresso
        return results


def estimate_memory_usage(
    num_images: int,
    image_size: Tuple[int, int],
    channels: int = 3,
    dtype_bytes: int = 4  # float32
) -> float:
    """
    Estima uso de memória para um conjunto de imagens

    Args:
        num_images: Número de imagens
        image_size: Tamanho (altura, largura)
        channels: Número de canais
        dtype_bytes: Bytes por elemento

    Returns:
        memory_gb: Memória estimada em GB
    """
    pixels_per_image = image_size[0] * image_size[1] * channels
    bytes_per_image = pixels_per_image * dtype_bytes
    total_bytes = num_images * bytes_per_image
    return total_bytes / (1024 ** 3)


def check_available_memory(required_gb: float, safety_margin: float = 0.2) -> bool:
    """
    Verifica se há memória suficiente disponível

    Args:
        required_gb: Memória requerida em GB
        safety_margin: Margem de segurança adicional

    Returns:
        available: True se há memória suficiente
    """
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 ** 3)
    return available_gb >= required_gb * (1 + safety_margin)
