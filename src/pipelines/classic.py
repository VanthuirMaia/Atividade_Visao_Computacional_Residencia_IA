# -*- coding: utf-8 -*-
"""
Pipeline Clássico - Classificação de Imagens
Aplica modelos tradicionais de machine learning
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint, loguniform
import joblib

from ..config import MODELS_DIR, RESULTS_DIR, FIGURES_DIR
from ..utils import (
    load_images_from_directory, preprocess_images_classic,
    calculate_metrics, plot_confusion_matrix, save_results_table,
    print_results_summary
)


class ClassicPipeline:
    """
    Pipeline clássico para classificação de imagens
    """

    def __init__(self, train_dir, test_dir, val_dir=None):
        """
        Inicializa o pipeline

        Args:
            train_dir: Diretório de treinamento
            test_dir: Diretório de teste
            val_dir: Diretório de validação (opcional)
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.scaler = StandardScaler()
        self.results = []

    def load_data(self):
        """
        Carrega e pré-processa os dados
        """
        print("Carregando dados de treinamento...")
        X_train, y_train, self.class_names = load_images_from_directory(self.train_dir)

        print("Carregando dados de teste...")
        X_test, y_test, _ = load_images_from_directory(self.test_dir)

        # Validar número de classes
        unique_train_classes = len(np.unique(y_train))
        unique_test_classes = len(np.unique(y_test)) if len(y_test) > 0 else 0
        
        if unique_train_classes < 2:
            raise ValueError(
                f"Apenas {unique_train_classes} classe(s) encontrada(s) nos dados de treinamento. "
                f"São necessárias pelo menos 2 classes para classificação. "
                f"Classes esperadas: {self.class_names}"
            )

        print(f"Classes encontradas: {self.class_names}")
        print(f"Classes únicas no treinamento: {unique_train_classes}")
        print(f"Treinamento: {X_train.shape[0]} amostras")
        print(f"Teste: {X_test.shape[0]} amostras")
        print(f"Tamanho das imagens: {X_train.shape[1:]} pixels")
        print(f"Canais: {X_train.shape[3] if len(X_train.shape) > 3 else 1}")
        
        # Mostrar distribuição de classes
        print(f"\nDistribuição de classes no treinamento:")
        for class_idx, class_name in enumerate(self.class_names):
            count = np.sum(y_train == class_idx)
            print(f"  {class_name}: {count} amostras")

        # Pré-processamento para modelos clássicos
        print("\nPré-processando imagens...")
        X_train_flat = preprocess_images_classic(X_train)
        X_test_flat = preprocess_images_classic(X_test)

        # Normalização
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)

        self.X_train = X_train_scaled
        self.y_train = y_train
        self.X_test = X_test_scaled
        self.y_test = y_test

        if self.val_dir and self.val_dir.exists():
            print("Carregando dados de validação...")
            X_val, y_val, _ = load_images_from_directory(self.val_dir)
            X_val_flat = preprocess_images_classic(X_val)
            self.X_val = self.scaler.transform(X_val_flat)
            self.y_val = y_val
        else:
            self.X_val = None
            self.y_val = None

    def train_svm(self, use_random_search=True, n_iter=50):
        """
        Treina modelo SVM

        Args:
            use_random_search: Se True, usa Random Search para otimização
            n_iter: Número de iterações do Random Search
        """
        # Validar que os dados foram carregados
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("Dados não carregados! Chame load_data() antes de treinar.")
        
        # Validar número de classes
        unique_classes = len(np.unique(self.y_train))
        if unique_classes < 2:
            raise ValueError(
                f"Apenas {unique_classes} classe(s) encontrada(s) nos dados de treinamento. "
                f"SVM requer pelo menos 2 classes. Classes esperadas: {self.class_names}"
            )
        
        print("\n" + "="*80)
        print("TREINANDO MODELO: Support Vector Machine (SVM)")
        print("="*80)

        if use_random_search:
            print(f"Otimizando hiperparâmetros com Random Search ({n_iter} iterações)...")
            param_distributions = {
                'C': loguniform(0.01, 100),
                'gamma': loguniform(0.0001, 1),
                'kernel': ['rbf', 'linear', 'poly'],
                'degree': randint(2, 5),
                'class_weight': [None, 'balanced']
            }

            svm = SVC(random_state=42)
            random_search = RandomizedSearchCV(
                svm, param_distributions, n_iter=n_iter, cv=3,
                scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
            )
            random_search.fit(self.X_train, self.y_train)

            self.svm_model = random_search.best_estimator_
            print(f"Melhores parâmetros: {random_search.best_params_}")
            print(f"Melhor score (CV): {random_search.best_score_:.4f}")
        else:
            print("Treinando com parâmetros padrão...")
            self.svm_model = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
            self.svm_model.fit(self.X_train, self.y_train)

        # Predições
        y_pred_train = self.svm_model.predict(self.X_train)
        y_pred_test = self.svm_model.predict(self.X_test)

        # Métricas
        metrics_train, _, _ = calculate_metrics(self.y_train, y_pred_train, self.class_names)
        metrics_test, report_test, cm_test = calculate_metrics(self.y_test, y_pred_test, self.class_names)

        print(f"\nAcurácia - Treinamento: {metrics_train['accuracy']:.4f}")
        print(f"Acurácia - Teste: {metrics_test['accuracy']:.4f}")
        print(f"Precisão - Teste: {metrics_test['precision']:.4f}")
        print(f"Recall - Teste: {metrics_test['recall']:.4f}")
        print(f"F1-Score - Teste: {metrics_test['f1_score']:.4f}")

        # Salvar modelo
        model_path = MODELS_DIR / 'svm_model.pkl'
        joblib.dump(self.svm_model, model_path)
        print(f"Modelo salvo em: {model_path}")

        # Salvar scaler
        scaler_path = MODELS_DIR / 'svm_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)

        # Plotar matriz de confusão
        cm_path = FIGURES_DIR / 'svm_confusion_matrix.png'
        plot_confusion_matrix(cm_test, self.class_names,
                            'SVM - Matriz de Confusão', cm_path)

        # Adicionar aos resultados
        self.results.append({
            'Modelo': 'SVM',
            'Acurácia': f"{metrics_test['accuracy']:.4f}",
            'Precisão': f"{metrics_test['precision']:.4f}",
            'Recall': f"{metrics_test['recall']:.4f}",
            'F1-Score': f"{metrics_test['f1_score']:.4f}",
            'Otimização': f'Random Search ({n_iter} iter)' if use_random_search else 'Padrão'
        })

    def train_random_forest(self, use_random_search=True, n_iter=50):
        """
        Treina modelo Random Forest

        Args:
            use_random_search: Se True, usa Random Search para otimização
            n_iter: Número de iterações do Random Search
        """
        # Validar que os dados foram carregados
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("Dados não carregados! Chame load_data() antes de treinar.")
        
        # Validar número de classes
        unique_classes = len(np.unique(self.y_train))
        if unique_classes < 2:
            raise ValueError(
                f"Apenas {unique_classes} classe(s) encontrada(s) nos dados de treinamento. "
                f"Random Forest requer pelo menos 2 classes. Classes esperadas: {self.class_names}"
            )
        
        print("\n" + "="*80)
        print("TREINANDO MODELO: Random Forest")
        print("="*80)

        if use_random_search:
            print(f"Otimizando hiperparâmetros com Random Search ({n_iter} iterações)...")
            param_distributions = {
                'n_estimators': randint(50, 300),
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            }

            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            random_search = RandomizedSearchCV(
                rf, param_distributions, n_iter=n_iter, cv=3,
                scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
            )
            random_search.fit(self.X_train, self.y_train)

            self.rf_model = random_search.best_estimator_
            print(f"Melhores parâmetros: {random_search.best_params_}")
            print(f"Melhor score (CV): {random_search.best_score_:.4f}")
        else:
            print("Treinando com parâmetros padrão...")
            self.rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
            )
            self.rf_model.fit(self.X_train, self.y_train)

        # Predições
        y_pred_train = self.rf_model.predict(self.X_train)
        y_pred_test = self.rf_model.predict(self.X_test)

        # Métricas
        metrics_train, _, _ = calculate_metrics(self.y_train, y_pred_train, self.class_names)
        metrics_test, report_test, cm_test = calculate_metrics(self.y_test, y_pred_test, self.class_names)

        print(f"\nAcurácia - Treinamento: {metrics_train['accuracy']:.4f}")
        print(f"Acurácia - Teste: {metrics_test['accuracy']:.4f}")
        print(f"Precisão - Teste: {metrics_test['precision']:.4f}")
        print(f"Recall - Teste: {metrics_test['recall']:.4f}")
        print(f"F1-Score - Teste: {metrics_test['f1_score']:.4f}")

        # Salvar modelo
        model_path = MODELS_DIR / 'random_forest_model.pkl'
        joblib.dump(self.rf_model, model_path)
        print(f"Modelo salvo em: {model_path}")

        # Plotar matriz de confusão
        cm_path = FIGURES_DIR / 'random_forest_confusion_matrix.png'
        plot_confusion_matrix(cm_test, self.class_names,
                            'Random Forest - Matriz de Confusão', cm_path)

        # Adicionar aos resultados
        self.results.append({
            'Modelo': 'Random Forest',
            'Acurácia': f"{metrics_test['accuracy']:.4f}",
            'Precisão': f"{metrics_test['precision']:.4f}",
            'Recall': f"{metrics_test['recall']:.4f}",
            'F1-Score': f"{metrics_test['f1_score']:.4f}",
            'Otimização': f'Random Search ({n_iter} iter)' if use_random_search else 'Padrão'
        })

    def save_results(self):
        """
        Salva resultados finais
        """
        results_path = RESULTS_DIR / 'classic_pipeline_results.csv'
        save_results_table(self.results, results_path)
        print_results_summary(self.results)


def main():
    """
    Função principal para executar o pipeline clássico
    """
    from ..config import TRAIN_DIR, TEST_DIR, VAL_DIR

    pipeline = ClassicPipeline(TRAIN_DIR, TEST_DIR, VAL_DIR)
    pipeline.load_data()

    # Treinar modelos com Random Search
    pipeline.train_svm(use_random_search=True, n_iter=50)
    pipeline.train_random_forest(use_random_search=True, n_iter=50)

    # Salvar resultados
    pipeline.save_results()

    print("\nPipeline clássico concluído!")


if __name__ == "__main__":
    main()
