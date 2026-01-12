# Histórico de Treinamentos - Resultados de Execução

Este arquivo documenta **todos os resultados de treinamento** de cada modelo, incluindo data/hora completa, métricas, configurações e observações.

> **Última atualização**: 12 de janeiro de 2026

---

## 📊 Resumo Executivo

| Modelo | Melhor Acurácia | Data/Hora | Status |
|--------|----------------|-----------|--------|
| **ResNet50** | **77.40%** 🏆 | 2026-01-12 14:25:17 | ✅ Novo melhor resultado geral |
| **CNN Simples** | **70.89%** | 2026-01-10 11:27:04 | ✅ Excelente |
| **SVM** | **68.15%** | 2026-01-10 ~07:31:55 | ✅ Excelente |
| **Random Forest** | **65.75%** | 2026-01-11 08:00:13 | ✅ Bom resultado |

---

## 🗂️ Índice

1. [Sessão de Treinamento - 10 de janeiro de 2026](#1-sessão-de-treinamento---10-de-janeiro-de-2026)
2. [Sessão de Treinamento - 11 de janeiro de 2026](#2-sessão-de-treinamento---11-de-janeiro-de-2026)
3. [Sessão de Treinamento - 12 de janeiro de 2026](#3-sessão-de-treinamento---12-de-janeiro-de-2026)
4. [Evolução das Métricas](#4-evolução-das-métricas)
5. [Observações e Análises](#5-observações-e-análises)

---

## 1. Sessão de Treinamento - 10 de janeiro de 2026

### Resumo da Sessão

Nesta sessão foram treinados 4 modelos: 2 clássicos (SVM e Random Forest) e 2 de deep learning (CNN Simples e ResNet50). O destaque foi a CNN Simples que alcançou a melhor acurácia da sessão (70.89%).

---

### 1.1. SVM - Execução #1

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

### 1.2. Random Forest - Execução #1

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

### 1.3. CNN Simples - Execução #1

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
- 🏆 **Melhor resultado da sessão** (70.89%)
- Demonstra que modelo simples pode superar modelos complexos com dataset pequeno
- Treinamento do zero (sem transfer learning)
- Data augmentation contribuiu para melhor performance
- Balanceamento excelente entre precisão e recall

---

### 1.4. ResNet50 - Execução #1

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

## 2. Sessão de Treinamento - 11 de janeiro de 2026

### Resumo da Sessão

Nesta sessão foram retreinados os modelos clássicos (SVM e Random Forest) com tamanho de imagem aumentado para 128×128 pixels. O Random Forest melhorou significativamente, enquanto o SVM apresentou piora.

---

### 2.1. SVM - Execução #2

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

### 2.2. Random Forest - Execução #2

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

## 3. Sessão de Treinamento - 12 de janeiro de 2026

### Resumo da Sessão

Nesta sessão foram treinados todos os 4 modelos novamente. **Destaque histórico**: ResNet50 alcançou 77.40% de acurácia, superando todos os modelos anteriores e estabelecendo o novo recorde. Os modelos clássicos foram treinados com 100 iterações de Random Search e PCA aumentado para 800 componentes.

---

### 3.1. CNN Simples - Execução #2

**Data/Hora**: 2026-01-12 13:01:48  
**Acurácia**: **68.49%** (0.6849)  
**Precisão**: 68.62% (0.6862)  
**Recall**: 68.49% (0.6849)  
**F1-Score**: 67.80% (0.6780)

**Configurações**:
- Tamanho de imagem: 224×224 pixels
- Transfer Learning: Não (treinado do zero)
- Data Augmentation: Sim
- Otimização: Random Search (10 iterações)
- Épocas finais: 50
- Dispositivo: CPU
- Tempo total: 37 minutos 43 segundos (2263.97s)
  - Random Search: 23 minutos 32 segundos (1412.69s)
  - Treinamento final: 14 minutos 04 segundos (844.01s)

**Hiperparâmetros Otimizados**:
- learning_rate: 0.00241
- batch_size: 16
- dropout_rate: 0.576
- hidden_units: 1024

**Observações**:
- Resultado ligeiramente inferior à primeira execução (-2.40% em relação a 70.89%)
- Learning rate maior (0.00241 vs 0.00013) pode ter impactado a convergência
- Tempo de treinamento menor (37min vs 53min)

---

### 3.2. ResNet50 - Execução #2 🏆

**Data/Hora**: 2026-01-12 14:25:17  
**Acurácia**: **77.40%** (0.7740) 🏆  
**Precisão**: 77.77% (0.7777)  
**Recall**: 77.40% (0.7740)  
**F1-Score**: 77.45% (0.7745)

**Configurações**:
- Tamanho de imagem: 224×224 pixels
- Transfer Learning: Sim (pré-treinado ImageNet)
- Fine-tuning: unfreeze_layers=2 (FC + layer4 + layer3)
- Data Augmentation: Sim
- Otimização: Random Search (10 iterações)
- Épocas finais: 50
- Dispositivo: CPU
- Tempo total: 1 hora 23 minutos 27 segundos (5007.24s)
  - Random Search: 54 minutos 12 segundos (3252.44s)
  - Treinamento final: 29 minutos 05 segundos (1745.76s)

**Hiperparâmetros Otimizados**:
- learning_rate: 0.0000313
- batch_size: 4
- unfreeze_layers: 2 (FC + layer4 + layer3 treinadas)

**Observações**:
- 🏆 **NOVO RECORDE GERAL: +22.26%** de melhoria em relação à primeira execução (55.14% → 77.40%)
- ✅ **Melhor modelo de todos os tempos** - superou a CNN Simples (70.89%)
- Learning rate muito menor (0.0000313 vs 0.00012) foi crucial para o sucesso
- Balanceamento excelente entre precisão e recall
- Fine-tuning funcionou perfeitamente com os ajustes de hiperparâmetros
- Demonstra que ResNet50 pode ser muito eficaz com dataset pequeno quando bem ajustado

---

### 3.3. SVM - Execução #3

**Data/Hora**: 2026-01-12 14:26:08  
**Acurácia**: **64.89%** (0.6489)  
**Precisão**: 65.18% (0.6518)  
**Recall**: 64.89% (0.6489)  
**F1-Score**: 63.84% (0.6384)

**Configurações**:
- Tamanho de imagem: 128×128 pixels
- Otimização: Random Search (**100 iterações** - aumentado de 50)
- CV Folds: 2
- PCA: Sim (**800 componentes** - aumentado de 500)
- Dispositivo: CPU
- Paralelização: 12 jobs
- Tempo total: 6.94 segundos

**Hiperparâmetros Otimizados**:
- C: 7.73
- gamma: 0.0001
- kernel: rbf
- degree: 3
- class_weight: balanced

**Observações**:
- Resultado ligeiramente melhor que a execução #2 (+0.51%), mas ainda abaixo da execução #1
- Aumento de iterações (100 vs 50) e componentes PCA (800 vs 500) não trouxe ganhos significativos
- Tempo de execução aumentou (6.94s vs 2.20s) mas ainda muito rápido

---

### 3.4. Random Forest - Execução #3

**Data/Hora**: 2026-01-12 14:27:32  
**Acurácia**: **60.67%** (0.6067)  
**Precisão**: 60.43% (0.6043)  
**Recall**: 60.67% (0.6067)  
**F1-Score**: 59.84% (0.5984)

**Configurações**:
- Tamanho de imagem: 128×128 pixels
- Otimização: Random Search (**100 iterações** - aumentado de 50)
- CV Folds: 2
- PCA: Sim (**800 componentes** - aumentado de 500)
- Dispositivo: CPU
- Paralelização: 12 jobs
- Tempo total: 1 minuto 24 segundos (84.62s)

**Hiperparâmetros Otimizados**:
- n_estimators: 282
- min_samples_split: 14
- min_samples_leaf: 9
- max_features: None
- bootstrap: True
- class_weight: null (alterado de balanced)

**Observações**:
- Resultado inferior às execuções anteriores (-5.08% em relação à execução #2)
- Aumento de iterações (100 vs 50) e componentes PCA (800 vs 500) não melhorou resultados
- Mudança de class_weight de "balanced" para null pode ter impactado negativamente
- Tempo de execução aumentou significativamente (84.62s vs 29.11s)

---

## 4. Evolução das Métricas

### Acurácia ao Longo do Tempo

| Data | SVM | Random Forest | CNN Simples | ResNet50 |
|------|-----|---------------|-------------|----------|
| 2026-01-10 | 68.15% | 63.01% | **70.89%** 🏆 | 55.14% |
| 2026-01-11 | 64.38% ⬇️ | **65.75%** ⬆️ | - | - |
| 2026-01-12 | 64.89% ⬆️ | 60.67% ⬇️ | 68.49% ⬇️ | **77.40%** 🏆⬆️ |

### Evolução por Modelo

| Modelo | Melhor Acurácia | Data | Pior Acurácia | Data | Variação |
|--------|----------------|------|---------------|------|----------|
| **ResNet50** | **77.40%** 🏆 | 2026-01-12 | 55.14% | 2026-01-10 | **+22.26%** ⬆️ |
| **CNN Simples** | 70.89% | 2026-01-10 | 68.49% | 2026-01-12 | -2.40% ⬇️ |
| **SVM** | 68.15% | 2026-01-10 | 64.38% | 2026-01-11 | -3.77% ⬇️ |
| **Random Forest** | 65.75% | 2026-01-11 | 60.67% | 2026-01-12 | -5.08% ⬇️ |

### Comparação: 64×64 vs 128×128 (Modelos Clássicos)

| Modelo | 64×64 (10/01) | 128×128 (11/01) | 128×128 (12/01) | Melhor Resultado |
|--------|---------------|-----------------|-----------------|------------------|
| **SVM** | 68.15% | 64.38% ⬇️ | 64.89% ⬆️ | 68.15% (64×64) |
| **Random Forest** | 63.01% | **65.75%** ⬆️ | 60.67% ⬇️ | 65.75% (128×128) |

**Análise**:
- ResNet50 teve a maior evolução: +22.26% (de 55.14% para 77.40%)
- Random Forest teve melhor resultado com 128×128 (execução #2), mas piorou na execução #3
- SVM performa melhor com 64×64 pixels
- CNN Simples manteve performance estável (68-71%)

---

## 5. Observações e Análises

### Melhores Resultados por Modelo (Histórico)

1. **ResNet50**: **77.40%** (2026-01-12) 🏆 - **NOVO RECORDE GERAL**
2. **CNN Simples**: 70.89% (2026-01-10) - Melhor resultado histórico
3. **SVM**: 68.15% (2026-01-10) - Melhor resultado histórico
4. **Random Forest**: 65.75% (2026-01-11) - Melhor resultado histórico

### Tendências Identificadas

- **Modelos Clássicos**:
  - Random Forest melhorou com 128×128 na execução #2 (+2.74%), mas piorou na #3
  - SVM performa melhor com 64×64 pixels (68.15% vs 64.89%)
  - Aumento de iterações (100) e componentes PCA (800) não trouxeram ganhos consistentes
  - Ambos são muito rápidos (< 90 segundos mesmo com mais iterações)

- **Deep Learning**:
  - **ResNet50 teve melhoria dramática**: +22.26% com ajuste fino de learning rate
  - Learning rate muito menor (0.0000313) foi crucial para ResNet50
  - CNN Simples mantém performance estável (68-71%)
  - Data augmentation é essencial para ambos
  - Transfer learning + fine-tuning funcionam muito bem quando bem ajustados

### Configurações que Funcionam Bem

✅ **Para Modelos Clássicos**:
- PCA com 500 componentes (aumentar para 800 não trouxe ganhos significativos)
- Random Search com 50 iterações (100 iterações não melhorou resultados)
- CV folds = 2 (economia de memória)
- Tamanho de imagem: **64×64 para SVM** (melhor resultado), **128×128 para RF** (melhor na execução #2)
- class_weight="balanced" para Random Forest

✅ **Para Deep Learning**:
- Data augmentation ativo (essencial)
- Random Search com 10 iterações (suficiente)
- Early stopping (patience=5)
- Tamanho de imagem: 224×224
- **ResNet50**: Learning rate muito baixo (0.00003) é crucial para fine-tuning eficaz
- **CNN Simples**: Learning rate moderado (0.0001-0.001) funciona bem

### Próximos Passos Sugeridos

1. **ResNet50** (Melhor modelo atual - 77.40%):
   - ✅ Ajuste de learning rate funcionou perfeitamente
   - Testar fine-tuning mais profundo (unfreeze_layers=3 ou mais) para potencial melhoria
   - Considerar aumentar épocas para ver se há convergência adicional
   - Considerar usar GPU para treinamento mais rápido

2. **SVM**:
   - Manter 64×64 pixels (melhor resultado: 68.15%)
   - Reajustar hiperparâmetros (especialmente C e gamma) para 128×128 se necessário
   - Avaliar se PCA com 500 componentes é o ideal

3. **CNN Simples**:
   - Investigar por que segunda execução teve resultado inferior
   - Testar learning rates mais baixos (como no ResNet50)
   - Tentar aumentar batch size
   - Testar diferentes arquiteturas (mais camadas)

4. **Random Forest**:
   - Retornar para class_weight="balanced" (piorou na execução #3)
   - Manter 128×128 pixels (melhor resultado na execução #2: 65.75%)
   - Manter 50 iterações (100 não trouxe ganhos)
   - Manter PCA com 500 componentes (800 não trouxe ganhos)

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
