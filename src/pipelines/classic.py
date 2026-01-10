# -*- coding: utf-8 -*-
"""
Pipeline Clássico - Classificação de Imagens
Aplica modelos tradicionais de machine learning
"""

import numpy as np
import time
import os
import multiprocessing
from datetime import timedelta
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import uniform, randint, loguniform
import joblib

from ..config import (
    MODELS_DIR, RESULTS_DIR, FIGURES_DIR, IMG_SIZE_CLASSIC,
    CLASSIC_USE_PCA, CLASSIC_PCA_COMPONENTS, CLASSIC_USE_LINEAR_SVM,
    CLASSIC_MAX_SAMPLES, CLASSIC_SVM_N_JOBS, CLASSIC_RF_N_JOBS, CLASSIC_CV_FOLDS
)
from ..utils import (
    load_images_from_directory, preprocess_images_classic,
    calculate_metrics, plot_confusion_matrix, save_results_table,
    print_results_summary
)
from ..memory import estimate_memory_usage, check_available_memory
from ..model_saver import save_model_with_metadata, create_model_metadata


class ClassicPipeline:
    """
    Pipeline clássico para classificação de imagens
    """

    def __init__(self, train_dir, test_dir, val_dir=None, n_jobs=None):
        """
        Inicializa o pipeline

        Args:
            train_dir: Diretório de treinamento
            test_dir: Diretório de teste
            val_dir: Diretório de validação (opcional)
            n_jobs: Número de jobs paralelos (-1 = todos os cores, None = 1)
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.scaler = StandardScaler()
        self.results = []
        
        # Configurar paralelização
        if n_jobs is None:
            # Usar todos os cores disponíveis por padrão
            self.n_jobs = -1
        else:
            self.n_jobs = n_jobs
        
        # Informações sobre CPU e paralelização
        self.num_cores = multiprocessing.cpu_count()
        actual_jobs = self.num_cores if self.n_jobs == -1 else self.n_jobs
        
        print("\n" + "="*60)
        print("CONFIGURAÇÃO DE PROCESSAMENTO - MODELOS CLÁSSICOS")
        print("="*60)
        print(f"  Modelos clássicos (SVM, Random Forest) usam CPU")
        print(f"     Não há suporte nativo para GPU no scikit-learn")
        print(f"\n   Paralelização CPU:")
        print(f"     Número de cores disponíveis: {self.num_cores}")
        print(f"     Jobs paralelos configurados: {actual_jobs} {'(todos os cores)' if self.n_jobs == -1 else f'(limitado a {self.n_jobs})'}")
        print(f"     n_jobs = {self.n_jobs}")
        print(f"\n   Para usar GPU com modelos clássicos, considere:")
        print(f"     - RAPIDS cuML (cuSVM, cuRF)")
        print(f"     - ThunderSVM (SVM com GPU)")
        print(f"     - XGBoost/LightGBM com GPU")
        print("="*60 + "\n")

    def load_data(self):
        """
        Carrega e pré-processa os dados
        Usa tamanho de imagem menor para modelos clássicos (economiza memória)
        """
        print("\n" + "="*60)
        print("CARREGANDO DADOS - MODELOS CLÁSSICOS")
        print("="*60)
        print(f"Tamanho de imagem: {IMG_SIZE_CLASSIC} (otimizado para modelos clássicos)")
        print(f"Tamanho padrão: (224, 224) (usado apenas para deep learning)")
        print("="*60 + "\n")
        
        print("Carregando dados de treinamento...")
        X_train, y_train, self.class_names = load_images_from_directory(
            self.train_dir, img_size=IMG_SIZE_CLASSIC
        )

        print("Carregando dados de teste...")
        X_test, y_test, _ = load_images_from_directory(
            self.test_dir, img_size=IMG_SIZE_CLASSIC
        )

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

        # Limitar número de amostras se configurado
        if CLASSIC_MAX_SAMPLES is not None and len(X_train) > CLASSIC_MAX_SAMPLES:
            print(f"\n  AVISO: Limitando amostras de treinamento de {len(X_train)} para {CLASSIC_MAX_SAMPLES}")
            indices = np.random.choice(len(X_train), CLASSIC_MAX_SAMPLES, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
            print(f"   Amostras selecionadas aleatoriamente mantendo proporção de classes")

        # Pré-processamento para modelos clássicos
        print("\nPré-processando imagens...")
        X_train_flat = preprocess_images_classic(X_train)
        X_test_flat = preprocess_images_classic(X_test)
        
        # Estimar memória necessária
        n_features = X_train_flat.shape[1]
        n_samples = X_train_flat.shape[0]
        estimated_mem_gb = (n_samples * n_features * 8) / (1024**3)  # float64
        print(f"   Features após flatten: {n_features:,}")
        print(f"   Memória estimada para dados: {estimated_mem_gb:.2f} GB")

        # Normalização
        print("   Aplicando StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        # PCA para redução de dimensionalidade (opcional)
        self.pca = None
        if CLASSIC_USE_PCA:
            print(f"\n   Aplicando PCA para redução de dimensionalidade...")
            
            if CLASSIC_PCA_COMPONENTS is None:
                # Auto: reduzir para 95% variância
                self.pca = PCA(n_components=0.95, random_state=42)
                print(f"   Modo: Auto (95% variância explicada)")
            else:
                # Número fixo de componentes
                n_components = min(CLASSIC_PCA_COMPONENTS, min(n_samples - 1, n_features))
                self.pca = PCA(n_components=n_components, random_state=42)
                print(f"   Modo: Fixo ({n_components} componentes)")
            
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
            
            n_features_after_pca = X_train_scaled.shape[1]
            reduction = ((n_features - n_features_after_pca) / n_features) * 100
            estimated_mem_after_gb = (n_samples * n_features_after_pca * 8) / (1024**3)
            print(f"   Features após PCA: {n_features_after_pca:,} ({reduction:.1f}% redução)")
            print(f"   Memória estimada após PCA: {estimated_mem_after_gb:.2f} GB")
            
            if hasattr(self.pca, 'explained_variance_ratio_'):
                total_variance = self.pca.explained_variance_ratio_.sum()
                print(f"   Variância explicada: {total_variance:.2%}")
            print()  # Linha em branco

        self.X_train = X_train_scaled
        self.y_train = y_train
        self.X_test = X_test_scaled
        self.y_test = y_test

        if self.val_dir and self.val_dir.exists():
            print("\nCarregando dados de validação...")
            X_val, y_val, _ = load_images_from_directory(self.val_dir, img_size=IMG_SIZE_CLASSIC)
            X_val_flat = preprocess_images_classic(X_val)
            X_val_scaled = self.scaler.transform(X_val_flat)
            
            # Aplicar PCA se foi usado no treinamento
            if self.pca is not None:
                X_val_scaled = self.pca.transform(X_val_scaled)
            
            self.X_val = X_val_scaled
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
        print(f"   Dispositivo: CPU (scikit-learn não suporta GPU)")
        
        # Verificar memória antes de treinar
        n_samples, n_features = self.X_train.shape
        estimated_mem_gb = (n_samples * n_features * 8 * CLASSIC_CV_FOLDS) / (1024**3)
        print(f"\n   Verificação de memória:")
        print(f"     Amostras: {n_samples:,}")
        print(f"     Features: {n_features:,}")
        print(f"     Memória estimada para treinamento: ~{estimated_mem_gb:.2f} GB")
        
        if not check_available_memory(estimated_mem_gb, safety_margin=0.3):
            print(f"      AVISO: Memória estimada pode exceder disponível!")
            print(f"     Recomendações:")
            print(f"       - Reduzir CLASSIC_MAX_SAMPLES em config.py")
            print(f"       - Ativar CLASSIC_USE_PCA = True")
            print(f"       - Usar CLASSIC_USE_LINEAR_SVM = True")
            print(f"       - Reduzir CLASSIC_CV_FOLDS para 2")
        
        # Determinar jobs para SVM (pode ser diferente do padrão para economizar memória)
        svm_n_jobs = CLASSIC_SVM_N_JOBS if CLASSIC_SVM_N_JOBS is not None else 1
        print(f"\n   Paralelização SVM: {svm_n_jobs} job(s) (configurado para economizar memória)")
        
        # Usar LinearSVC se configurado
        use_linear_svm = CLASSIC_USE_LINEAR_SVM
        if use_linear_svm:
            print(f"   Tipo: LinearSVC (kernel linear, mais eficiente em memória)")
        else:
            print(f"   Tipo: SVC (suporta kernels não-lineares, mas usa mais memória)")

        # Iniciar medição de tempo
        start_time = time.time()

        if use_random_search:
            print(f"\n   Otimizando hiperparâmetros com Random Search ({n_iter} iterações)...")
            print(f"   CV folds: {CLASSIC_CV_FOLDS} (reduzido para economizar memória)")
            search_start = time.time()
            
            if use_linear_svm:
                # LinearSVC: apenas kernel linear, menos parâmetros
                param_distributions = {
                    'C': loguniform(0.01, 100),
                    'loss': ['hinge', 'squared_hinge'],
                    'class_weight': [None, 'balanced'],
                    'dual': [True, False]  # False pode ser mais rápido para n_samples > n_features
                }
                svm = LinearSVC(random_state=42, max_iter=2000)
            else:
                # SVC tradicional: múltiplos kernels
                param_distributions = {
                    'C': loguniform(0.01, 100),
                    'gamma': loguniform(0.0001, 1),
                    'kernel': ['rbf', 'linear', 'poly'],
                    'degree': randint(2, 5),
                    'class_weight': [None, 'balanced']
                }
                svm = SVC(random_state=42)
            
            random_search = RandomizedSearchCV(
                svm, param_distributions, n_iter=n_iter, cv=CLASSIC_CV_FOLDS,
                scoring='accuracy', n_jobs=svm_n_jobs, verbose=1, random_state=42
            )
            random_search.fit(self.X_train, self.y_train)

            search_time = time.time() - search_start
            search_time_str = str(timedelta(seconds=int(search_time)))
            
            self.svm_model = random_search.best_estimator_
            print(f"Melhores parâmetros: {random_search.best_params_}")
            print(f"Melhor score (CV): {random_search.best_score_:.4f}")
            print(f"Tempo de Random Search: {search_time_str} ({search_time:.2f} segundos)")
        else:
            print("Treinando com parâmetros padrão...")
            train_start = time.time()
            if use_linear_svm:
                self.svm_model = LinearSVC(C=1.0, random_state=42, max_iter=2000)
            else:
                self.svm_model = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
            self.svm_model.fit(self.X_train, self.y_train)
            train_time = time.time() - train_start
            print(f"Tempo de treinamento: {train_time:.2f} segundos")

        # Predições
        pred_start = time.time()
        y_pred_train = self.svm_model.predict(self.X_train)
        y_pred_test = self.svm_model.predict(self.X_test)
        pred_time = time.time() - pred_start

        # Métricas
        metrics_train, _, _ = calculate_metrics(self.y_train, y_pred_train, self.class_names)
        metrics_test, report_test, cm_test = calculate_metrics(self.y_test, y_pred_test, self.class_names)

        # Calcular tempo total
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))

        print(f"\nAcurácia - Treinamento: {metrics_train['accuracy']:.4f}")
        print(f"Acurácia - Teste: {metrics_test['accuracy']:.4f}")
        print(f"Precisão - Teste: {metrics_test['precision']:.4f}")
        print(f"Recall - Teste: {metrics_test['recall']:.4f}")
        print(f"F1-Score - Teste: {metrics_test['f1_score']:.4f}")
        print(f"\nTempo total de execução: {total_time_str} ({total_time:.2f} segundos)")

        # Preparar hiperparâmetros para metadados
        if use_random_search:
            best_hyperparams = random_search.best_params_
        else:
            if use_linear_svm:
                best_hyperparams = {'C': 1.0, 'loss': 'squared_hinge'}
            else:
                best_hyperparams = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
        
        # Criar metadados
        metadata = create_model_metadata(
            model_name='SVM',
            metrics=metrics_test,
            hyperparams=best_hyperparams,
            training_info={
                'use_random_search': use_random_search,
                'n_iter': n_iter if use_random_search else 0,
                'cv_folds': CLASSIC_CV_FOLDS if use_random_search else 0,
                'pca_used': self.pca is not None,
                'pca_components': self.pca.n_components if self.pca is not None else None,
                'use_linear_svm': use_linear_svm,
                'img_size_classic': IMG_SIZE_CLASSIC,
                'max_samples': CLASSIC_MAX_SAMPLES,
                'total_time_seconds': total_time,
                'device': 'CPU',
                'n_jobs': svm_n_jobs
            },
            class_names=self.class_names
        )
        
        # Salvar modelo com metadados
        model_path = MODELS_DIR / 'svm_model.pkl'
        save_model_with_metadata(
            model=self.svm_model,
            model_path=model_path,
            metadata=metadata,
            model_type='sklearn'
        )

        # Salvar scaler e PCA (necessários para fazer predições)
        scaler_path = MODELS_DIR / 'svm_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler salvo em: {scaler_path}")
        
        if self.pca is not None:
            pca_path = MODELS_DIR / 'svm_pca.pkl'
            joblib.dump(self.pca, pca_path)
            print(f"PCA salvo em: {pca_path} (n_components={self.pca.n_components})")

        # Plotar matriz de confusão
        cm_path = FIGURES_DIR / 'svm_confusion_matrix.png'
        plot_confusion_matrix(cm_test, self.class_names,
                            'SVM - Matriz de Confusão', cm_path)

        # Adicionar aos resultados
        actual_jobs = self.num_cores if self.n_jobs == -1 else self.n_jobs
        self.results.append({
            'Modelo': 'SVM',
            'Acurácia': f"{metrics_test['accuracy']:.4f}",
            'Precisão': f"{metrics_test['precision']:.4f}",
            'Recall': f"{metrics_test['recall']:.4f}",
            'F1-Score': f"{metrics_test['f1_score']:.4f}",
            'Otimização': f'Random Search ({n_iter} iter)' if use_random_search else 'Padrão',
            'Dispositivo': 'CPU',
            'Jobs Paralelos': str(actual_jobs),
            'Tempo Total (s)': f"{total_time:.2f}",
            'Tempo Total (hh:mm:ss)': total_time_str
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
        print(f"   Dispositivo: CPU (scikit-learn não suporta GPU)")
        
        # Determinar jobs para Random Forest (pode usar mais paralelização que SVM)
        rf_n_jobs = CLASSIC_RF_N_JOBS if CLASSIC_RF_N_JOBS is not None else self.n_jobs
        if rf_n_jobs == -1:
            actual_jobs = self.num_cores
        else:
            actual_jobs = rf_n_jobs
        print(f"   Paralelização: {actual_jobs} job(s) paralelo(s) (Random Forest pode usar mais cores eficientemente)")

        # Iniciar medição de tempo
        start_time = time.time()

        if use_random_search:
            print(f"\n   Otimizando hiperparâmetros com Random Search ({n_iter} iterações)...")
            print(f"   CV folds: {CLASSIC_CV_FOLDS} (reduzido para economizar memória)")
            search_start = time.time()
            
            param_distributions = {
                'n_estimators': randint(50, 300),
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            }

            rf = RandomForestClassifier(random_state=42, n_jobs=rf_n_jobs)
            random_search = RandomizedSearchCV(
                rf, param_distributions, n_iter=n_iter, cv=CLASSIC_CV_FOLDS,
                scoring='accuracy', n_jobs=rf_n_jobs, verbose=1, random_state=42
            )
            random_search.fit(self.X_train, self.y_train)

            search_time = time.time() - search_start
            search_time_str = str(timedelta(seconds=int(search_time)))
            
            self.rf_model = random_search.best_estimator_
            print(f"Melhores parâmetros: {random_search.best_params_}")
            print(f"Melhor score (CV): {random_search.best_score_:.4f}")
            print(f"Tempo de Random Search: {search_time_str} ({search_time:.2f} segundos)")
        else:
            print("Treinando com parâmetros padrão...")
            train_start = time.time()
            self.rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=None, random_state=42, n_jobs=self.n_jobs
            )
            self.rf_model.fit(self.X_train, self.y_train)
            train_time = time.time() - train_start
            print(f"Tempo de treinamento: {train_time:.2f} segundos")

        # Predições
        pred_start = time.time()
        y_pred_train = self.rf_model.predict(self.X_train)
        y_pred_test = self.rf_model.predict(self.X_test)
        pred_time = time.time() - pred_start

        # Métricas
        metrics_train, _, _ = calculate_metrics(self.y_train, y_pred_train, self.class_names)
        metrics_test, report_test, cm_test = calculate_metrics(self.y_test, y_pred_test, self.class_names)

        # Calcular tempo total
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))

        print(f"\nAcurácia - Treinamento: {metrics_train['accuracy']:.4f}")
        print(f"Acurácia - Teste: {metrics_test['accuracy']:.4f}")
        print(f"Precisão - Teste: {metrics_test['precision']:.4f}")
        print(f"Recall - Teste: {metrics_test['recall']:.4f}")
        print(f"F1-Score - Teste: {metrics_test['f1_score']:.4f}")
        print(f"\nTempo total de execução: {total_time_str} ({total_time:.2f} segundos)")

        # Preparar hiperparâmetros para metadados
        if use_random_search:
            best_hyperparams = random_search.best_params_
        else:
            best_hyperparams = {'n_estimators': 100, 'max_depth': None}
        
        # Criar metadados
        metadata = create_model_metadata(
            model_name='Random Forest',
            metrics=metrics_test,
            hyperparams=best_hyperparams,
            training_info={
                'use_random_search': use_random_search,
                'n_iter': n_iter if use_random_search else 0,
                'cv_folds': CLASSIC_CV_FOLDS if use_random_search else 0,
                'pca_used': self.pca is not None,
                'pca_components': self.pca.n_components if self.pca is not None else None,
                'img_size_classic': IMG_SIZE_CLASSIC,
                'max_samples': CLASSIC_MAX_SAMPLES,
                'total_time_seconds': total_time,
                'device': 'CPU',
                'n_jobs': actual_jobs
            },
            class_names=self.class_names
        )
        
        # Salvar modelo com metadados
        model_path = MODELS_DIR / 'random_forest_model.pkl'
        save_model_with_metadata(
            model=self.rf_model,
            model_path=model_path,
            metadata=metadata,
            model_type='sklearn'
        )

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
            'Otimização': f'Random Search ({n_iter} iter)' if use_random_search else 'Padrão',
            'Dispositivo': 'CPU',
            'Jobs Paralelos': str(actual_jobs),
            'Tempo Total (s)': f"{total_time:.2f}",
            'Tempo Total (hh:mm:ss)': total_time_str
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
