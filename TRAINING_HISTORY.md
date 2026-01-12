# Histórico de Treinamentos - Resultados de Execução

Este arquivo documenta **todos os resultados de treinamento** de cada modelo, incluindo data/hora completa, métricas, configurações e observações.

> **Última atualização**: 11 de janeiro de 2026

---

## 📊 Resumo Executivo

| Modelo | Melhor Acurácia | Data/Hora | Status |
|--------|----------------|-----------|--------|
| **CNN Simples** | **70.89%** | 2026-01-10 11:27:04 | ✅ Melhor resultado |
| **SVM** | **68.15%** | 2026-01-10 (primeiro treinamento) | ✅ Excelente |
| **Random Forest** | **65.75%** | 2026-01-11 08:00:13 | ✅ Melhorou |
| **ResNet50** | **55.14%** | 2026-01-10 13:24:05 | ⚠️ Requer ajustes |

---

## 🗂️ Índice

1. [Pipeline Clássico - SVM](#1-pipeline-clássico---svm)
2. [Pipeline Clássico - Random Forest](#2-pipeline-clássico---random-forest)
3. [Pipeline Deep Learning - CNN Simples](#3-pipeline-deep-learning---cnn-simples)
4. [Pipeline Deep Learning - ResNet50](#4-pipeline-deep-learning---resnet50)
5. [Evolução das Métricas](#5-evolução-das-métricas)
6. [Observações e Análises](#6-observações-e-análises)

---

## 1. Pipeline Clássico - SVM

### Execução #1 - 10 de janeiro de 2026

**Data/Hora**: 2026-01-10 ~07:31:55  
**Acurácia**: **68.15%** (0.6815)  
**Precisão**: 68.68% (0.6868)  
**Recall**: 68.15% (0.6815)  
**F1-Score**: 68.23% (0.6823)

**Configurações**:
- Tamanho de imagem: 64×64 pixels
- Otimização: Random Search (50 iterações)
- CV Folds: 2
- PCA: Sim (500 componentes)
- Dispositivo: CPU
- Paralelização: 1 job (economia de memória)
- Tempo total: 2.13 segundos

**Hiperparâmetros Otimizados**:
- C: 7.73
- gamma: 0.0001
- kernel: rbf
- degree: 3
- class_weight: balanced

**Observações**:
- Excelente resultado para modelo clássico
- Treinamento muito rápido (< 3 segundos)
- Melhor modelo clássico desta execução

---

### Execução #2 - 11 de janeiro de 2026

**Data/Hora**: 2026-01-11 07:59:42  
**Acurácia**: **64.38%** (0.6438)  
**Precisão**: 64.17% (0.6417)  
**Recall**: 64.38% (0.6438)  
**F1-Score**: 63.75% (0.6375)

**Configurações**:
- Tamanho de imagem: **128×128 pixels** (alterado de 64×64)
- Otimização: Random Search (50 iterações)
- CV Folds: 2
- PCA: Sim (500 componentes)
- Dispositivo: CPU
- Paralelização: 1 job
- Tempo total: 2.20 segundos

**Hiperparâmetros Otimizados**:
- C: 7.73
- gamma: 0.0001
- kernel: rbf
- degree: 3
- class_weight: balanced

**Observações**:
- Acurácia ligeiramente inferior à primeira execução (-3.77%)
- Possível necessidade de ajuste de hiperparâmetros para novo tamanho de imagem
- Tempo de execução similar (2.20s vs 2.13s)
- Com mais pixels (128×128 vs 64×64), o PCA pode estar descartando informação relevante

---

## 2. Pipeline Clássico - Random Forest

### Execução #1 - 10 de janeiro de 2026

**Data/Hora**: 2026-01-10 ~07:32:26  
**Acurácia**: **63.01%** (0.6301)  
**Precisão**: 62.94% (0.6294)  
**Recall**: 63.01% (0.6301)  
**F1-Score**: 61.73% (0.6173)

**Configurações**:
- Tamanho de imagem: 64×64 pixels
- Otimização: Random Search (50 iterações)
- CV Folds: 2
- PCA: Sim (500 componentes)
- Dispositivo: CPU
- Paralelização: 12 jobs (todos os cores)
- Tempo total: 30.77 segundos

**Hiperparâmetros Otimizados**:
- n_estimators: 282
- min_samples_split: 14
- min_samples_leaf: 9
- max_features: None
- bootstrap: True
- class_weight: balanced

**Observações**:
- Resultado adequado para Random Forest
- Segunda melhor performance entre modelos clássicos
- Treinamento rápido (30 segundos)

---

### Execução #2 - 11 de janeiro de 2026

**Data/Hora**: 2026-01-11 08:00:13  
**Acurácia**: **65.75%** (0.6575) 🎉  
**Precisão**: 65.71% (0.6571)  
**Recall**: 65.75% (0.6575)  
**F1-Score**: 65.73% (0.6573)

**Configurações**:
- Tamanho de imagem: **128×128 pixels** (alterado de 64×64)
- Otimização: Random Search (50 iterações)
- CV Folds: 2
- PCA: Sim (500 componentes)
- Dispositivo: CPU
- Paralelização: 12 jobs (todos os cores)
- Tempo total: 29.11 segundos

**Hiperparâmetros Otimizados**:
- n_estimators: 139
- min_samples_split: 17
- min_samples_leaf: 7
- max_depth: 10
- max_features: None
- bootstrap: True
- class_weight: balanced

**Observações**:
- ✅ **Melhoria de +2.74%** em relação à primeira execução (63.01% → 65.75%)
- Random Forest se beneficiou do aumento do tamanho da imagem (128×128)
- Tempo de execução similar (29.11s vs 30.77s)
- Superou o SVM nesta execução (65.75% vs 64.38%)

---

## 3. Pipeline Deep Learning - CNN Simples

### Execução #1 - 10 de janeiro de 2026

**Data/Hora**: 2026-01-10 11:27:04  
**Acurácia**: **70.89%** (0.7089) 🏆  
**Precisão**: 70.80% (0.7080)  
**Recall**: 70.89% (0.7089)  
**F1-Score**: 70.65% (0.7065)

**Configurações**:
- Tamanho de imagem: 224×224 pixels
- Transfer Learning: Não (treinado do zero)
- Data Augmentation: Sim
- Otimização: Random Search (10 iterações)
- Épocas finais: 50
- Dispositivo: CPU
- Tempo total: 53 minutos 28 segundos (3208.31s)
  - Random Search: 34 minutos 21 segundos (2061.64s)
  - Treinamento final: 18 minutos 57 segundos (1137.87s)

**Hiperparâmetros Otimizados**:
- learning_rate: 0.00013
- batch_size: 16
- dropout_rate: 0.35
- hidden_units: 1024

**Observações**:
- 🏆 **Melhor resultado geral de todos os modelos** (70.89%)
- Demonstra que modelo simples pode superar modelos complexos com dataset pequeno
- Treinamento do zero (sem transfer learning)
- Data augmentation contribuiu para melhor performance
- Balanceamento excelente entre precisão e recall

---

## 4. Pipeline Deep Learning - ResNet50

### Execução #1 - 10 de janeiro de 2026

**Data/Hora**: 2026-01-10 13:24:05  
**Acurácia**: **55.14%** (0.5514)  
**Precisão**: 30.40% (0.3040)  
**Recall**: 55.14% (0.5514)  
**F1-Score**: 39.19% (0.3919)

**Configurações**:
- Tamanho de imagem: 224×224 pixels
- Transfer Learning: Sim (pré-treinado ImageNet)
- Fine-tuning: unfreeze_layers=2 (FC + layer4 + layer3)
- Data Augmentation: Sim
- Otimização: Random Search (10 iterações)
- Épocas finais: 50
- Dispositivo: CPU
- Tempo total: 1 hora 57 minutos (7020.57s)
  - Random Search: 1 hora 08 minutos (4084.51s)
  - Treinamento final: 48 minutos 40 segundos (2920.78s)

**Hiperparâmetros Otimizados**:
- learning_rate: 0.00012
- batch_size: 8
- unfreeze_layers: 2 (FC + layer4 + layer3 treinadas)

**Observações**:
- ⚠️ Performance abaixo do esperado (55.14% - próximo de aleatório para 2 classes)
- Precisão muito baixa (30.40%) indica muitos falsos positivos
- Possíveis causas:
  - Dataset pequeno (~975 imagens) pode não ser suficiente para fine-tuning eficaz
  - Fine-tuning parcial pode precisar de mais ajustes
  - Necessidade de mais épocas, ajuste de learning rate ou batch size maior
  - Treinamento em CPU (mais lento, pode ter impacto na convergência)
- Requer investigação adicional e ajustes de hiperparâmetros

---

## 5. Evolução das Métricas

### Acurácia ao Longo do Tempo

| Data | SVM | Random Forest | CNN Simples | ResNet50 |
|------|-----|---------------|-------------|----------|
| 2026-01-10 | 68.15% | 63.01% | 70.89% | 55.14% |
| 2026-01-11 | 64.38% ⬇️ | **65.75%** ⬆️ | - | - |

### Comparação: 64×64 vs 128×128 (Modelos Clássicos)

| Modelo | 64×64 (10/01) | 128×128 (11/01) | Variação |
|--------|---------------|-----------------|----------|
| **SVM** | 68.15% | 64.38% | **-3.77%** ⬇️ |
| **Random Forest** | 63.01% | **65.75%** | **+2.74%** ⬆️ |

**Análise**:
- Random Forest se beneficiou do aumento de resolução (128×128)
- SVM apresentou piora, possivelmente devido à necessidade de ajuste de hiperparâmetros
- Com PCA ativado (500 componentes), o impacto do tamanho da imagem pode ser limitado

---

## 6. Observações e Análises

### Melhores Resultados por Modelo

1. **CNN Simples**: 70.89% (2026-01-10) - Melhor resultado geral
2. **SVM**: 68.15% (2026-01-10) - Melhor resultado histórico
3. **Random Forest**: 65.75% (2026-01-11) - Melhor resultado atual
4. **ResNet50**: 55.14% (2026-01-10) - Requer melhorias

### Tendências Identificadas

- **Modelos Clássicos**:
  - Random Forest melhorou com 128×128 (+2.74%)
  - SVM piorou com 128×128 (-3.77%)
  - Ambos são muito rápidos (< 30 segundos)

- **Deep Learning**:
  - CNN Simples mantém liderança (70.89%)
  - ResNet50 requer ajustes significativos
  - Data augmentation é essencial

### Configurações que Funcionam Bem

✅ **Para Modelos Clássicos**:
- PCA com 500 componentes
- Random Search com 50 iterações
- CV folds = 2 (economia de memória)
- Tamanho de imagem: 128×128 para RF, 64×64 para SVM (testar)

✅ **Para Deep Learning**:
- Data augmentation ativo
- Random Search com 10 iterações
- Early stopping (patience=5)
- Tamanho de imagem: 224×224

### Próximos Passos Sugeridos

1. **SVM com 128×128**:
   - Reajustar hiperparâmetros (especialmente C e gamma)
   - Testar diferentes números de componentes PCA
   - Avaliar se 64×64 é realmente melhor para SVM

2. **ResNet50**:
   - Aumentar número de épocas
   - Ajustar learning rate
   - Testar fine-tuning mais profundo (unfreeze_layers=3 ou mais)
   - Considerar usar GPU para treinamento mais rápido

3. **CNN Simples**:
   - Tentar aumentar batch size
   - Testar diferentes arquiteturas (mais camadas)
   - Avaliar impacto de diferentes taxas de dropout

4. **Random Forest**:
   - Testar com 128×128 sem PCA para comparar
   - Aumentar número de estimadores
   - Avaliar impacto de diferentes estratégias de split

---

## 📝 Como Adicionar Novos Resultados

Ao executar novos treinamentos, adicione uma nova entrada seguindo este formato:

```markdown
### Execução #N - [DATA]

**Data/Hora**: YYYY-MM-DD HH:MM:SS  
**Acurácia**: XX.XX% (0.XXXX)  
**Precisão**: XX.XX% (0.XXXX)  
**Recall**: XX.XX% (0.XXXX)  
**F1-Score**: XX.XX% (0.XXXX)

**Configurações**:
- [Lista de configurações]

**Hiperparâmetros Otimizados**:
- [Lista de hiperparâmetros]

**Observações**:
- [Observações sobre a execução]
```

As informações completas podem ser obtidas dos arquivos JSON em `outputs/models/`.

---

**Arquivo criado em**: 2026-01-11  
**Versão do Projeto**: 1.0.1
