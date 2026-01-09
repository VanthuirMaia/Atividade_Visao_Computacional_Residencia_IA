# -*- coding: utf-8 -*-
"""
Módulo de pipelines de classificação
"""

from .classic import ClassicPipeline
from .deep_learning import DeepLearningPipeline

__all__ = ["ClassicPipeline", "DeepLearningPipeline"]
