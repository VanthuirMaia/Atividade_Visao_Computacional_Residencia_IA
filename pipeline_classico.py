# -*- coding: utf-8 -*-
"""
Pipeline Clássico - Classificação de Imagens
Aplica modelos tradicionais de machine learning
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
from config import MODELS_DIR, RESULTS_DIR, FIGURES_DIR
from utils import (
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
        
        print(f"Classes encontradas: {self.class_names}")
        print(f"Treinamento: {X_train.shape[0]} amostras")
        print(f"Teste: {X_test.shape[0]} amostras")
        print(f"Tamanho das imagens: {X_train.shape[1:]} pixels")
        print(f"Canais: {X_train.shape[3] if len(X_train.shape) > 3 else 1}")
        
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
        
        if self.val_dir and os.path.exists(self.val_dir):
            print("Carregando dados de validação...")
            X_val, y_val, _ = load_images_from_directory(self.val_dir)
            X_val_flat = preprocess_images_classic(X_val)
            self.X_val = self.scaler.transform(X_val_flat)
            self.y_val = y_val
        else:
            self.X_val = None
            self.y_val = None
    
    def train_svm(self, use_grid_search=True):
        """
        Treina modelo SVM
        
        Args:
            use_grid_search: Se True, usa Grid Search para otimização
        """
        print("\n" + "="*80)
        print("TREINANDO MODELO: Support Vector Machine (SVM)")
        print("="*80)
        
        if use_grid_search:
            print("Otimizando hiperparâmetros com Grid Search...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear', 'poly']
            }
            
            svm = SVC(random_state=42)
            grid_search = GridSearchCV(
                svm, param_grid, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            self.svm_model = grid_search.best_estimator_
            print(f"Melhores parâmetros: {grid_search.best_params_}")
            print(f"Melhor score (CV): {grid_search.best_score_:.4f}")
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
        model_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
        joblib.dump(self.svm_model, model_path)
        print(f"Modelo salvo em: {model_path}")
        
        # Salvar scaler
        scaler_path = os.path.join(MODELS_DIR, 'svm_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Plotar matriz de confusão
        cm_path = os.path.join(FIGURES_DIR, 'svm_confusion_matrix.png')
        plot_confusion_matrix(cm_test, self.class_names, 
                            'SVM - Matriz de Confusão', cm_path)
        
        # Adicionar aos resultados
        self.results.append({
            'Modelo': 'SVM',
            'Acurácia': f"{metrics_test['accuracy']:.4f}",
            'Precisão': f"{metrics_test['precision']:.4f}",
            'Recall': f"{metrics_test['recall']:.4f}",
            'F1-Score': f"{metrics_test['f1_score']:.4f}",
            'Otimização': 'Grid Search' if use_grid_search else 'Padrão'
        })
    
    def train_logistic_regression(self, use_grid_search=True):
        """
        Treina modelo Regressão Logística
        
        Args:
            use_grid_search: Se True, usa Grid Search para otimização
        """
        print("\n" + "="*80)
        print("TREINANDO MODELO: Regressão Logística")
        print("="*80)
        
        if use_grid_search:
            print("Otimizando hiperparâmetros com Grid Search...")
            # Grid Search separado para diferentes penalties e solvers compatíveis
            # lbfgs: apenas l2
            # liblinear: l1 e l2
            # saga: l1, l2 e elasticnet
            
            param_grid = [
                {
                    'penalty': ['l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [100, 500, 1000]
                },
                {
                    'penalty': ['l1'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [100, 500, 1000]
                },
                {
                    'penalty': ['elasticnet'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['saga'],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_iter': [500, 1000]
                }
            ]
            
            lr = LogisticRegression(random_state=42, n_jobs=-1, multi_class='ovr')
            grid_search = GridSearchCV(
                lr, param_grid, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            self.lr_model = grid_search.best_estimator_
            print(f"Melhores parâmetros: {grid_search.best_params_}")
            print(f"Melhor score (CV): {grid_search.best_score_:.4f}")
        else:
            print("Treinando com parâmetros padrão...")
            self.lr_model = LogisticRegression(
                C=1.0, penalty='l2', solver='lbfgs', 
                max_iter=500, random_state=42, n_jobs=-1, multi_class='ovr'
            )
            self.lr_model.fit(self.X_train, self.y_train)
        
        # Predições
        y_pred_train = self.lr_model.predict(self.X_train)
        y_pred_test = self.lr_model.predict(self.X_test)
        
        # Métricas
        metrics_train, _, _ = calculate_metrics(self.y_train, y_pred_train, self.class_names)
        metrics_test, report_test, cm_test = calculate_metrics(self.y_test, y_pred_test, self.class_names)
        
        print(f"\nAcurácia - Treinamento: {metrics_train['accuracy']:.4f}")
        print(f"Acurácia - Teste: {metrics_test['accuracy']:.4f}")
        print(f"Precisão - Teste: {metrics_test['precision']:.4f}")
        print(f"Recall - Teste: {metrics_test['recall']:.4f}")
        print(f"F1-Score - Teste: {metrics_test['f1_score']:.4f}")
        
        # Salvar modelo
        model_path = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
        joblib.dump(self.lr_model, model_path)
        print(f"Modelo salvo em: {model_path}")
        
        # Plotar matriz de confusão
        cm_path = os.path.join(FIGURES_DIR, 'logistic_regression_confusion_matrix.png')
        plot_confusion_matrix(cm_test, self.class_names,
                            'Regressão Logística - Matriz de Confusão', cm_path)
        
        # Adicionar aos resultados
        self.results.append({
            'Modelo': 'Regressão Logística',
            'Acurácia': f"{metrics_test['accuracy']:.4f}",
            'Precisão': f"{metrics_test['precision']:.4f}",
            'Recall': f"{metrics_test['recall']:.4f}",
            'F1-Score': f"{metrics_test['f1_score']:.4f}",
            'Otimização': 'Grid Search' if use_grid_search else 'Padrão'
        })
    
    def save_results(self):
        """
        Salva resultados finais
        """
        results_path = os.path.join(RESULTS_DIR, 'classic_pipeline_results.csv')
        save_results_table(self.results, results_path)
        print_results_summary(self.results)

def main():
    """
    Função principal para executar o pipeline clássico
    """
    from config import TRAIN_DIR, TEST_DIR, VAL_DIR
    
    pipeline = ClassicPipeline(TRAIN_DIR, TEST_DIR, VAL_DIR)
    pipeline.load_data()
    
    # Treinar modelos
    pipeline.train_svm(use_grid_search=True)
    pipeline.train_logistic_regression(use_grid_search=True)
    
    # Salvar resultados
    pipeline.save_results()
    
    print("\nPipeline clássico concluído!")

if __name__ == "__main__":
    main()

