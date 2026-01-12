# Análise e Melhorias para Modelos Clássicos

## 📊 Análise dos Resultados Atuais

### Resultados Históricos

| Modelo | Melhor Resultado | Tamanho Imagem | Data | Status |
|--------|------------------|----------------|------|--------|
| **SVM** | **68.15%** | 64×64 | 2026-01-10 | ✅ Melhor histórico |
| **SVM** | 64.38% | 128×128 | 2026-01-11 | ⚠️ Piorou (-3.77%) |
| **SVM** | 59.74% | ? | Último CSV | ❌ Muito baixo |
| **Random Forest** | **65.75%** | 128×128 | 2026-01-11 | ✅ Melhor resultado |
| **Random Forest** | 63.01% | 64×64 | 2026-01-10 | ✅ Bom |
| **Random Forest** | 64.61% | ? | Último CSV | ✅ Adequado |

### Problemas Identificados

#### 1. **Configuração Inconsistente de Tamanho de Imagem**
- **Problema**: `IMG_SIZE_CLASSIC = (224, 224)` no `config.py`
- **Esperado**: `(128, 128)` conforme documentação
- **Impacto**: Imagens muito grandes (224×224) = 150,528 features por imagem
- **Com PCA 500**: Perde muita informação (99.67% de redução!)

#### 2. **SVM Piorou com 128×128**
- **Análise**: SVM se beneficiou de 64×64 (68.15%) mas piorou com 128×128 (64.38%)
- **Causa provável**: 
  - Hiperparâmetros não ajustados para novo tamanho
  - PCA com 500 componentes pode estar descartando informação relevante
  - Gamma muito baixo (0.0001) pode não ser ideal para imagens maiores

#### 3. **PCA com Poucos Componentes para Imagens Maiores**
- **Atual**: 500 componentes PCA
- **Com 64×64**: 12,288 features → 500 componentes (96% redução)
- **Com 128×128**: 49,152 features → 500 componentes (99% redução!)
- **Com 224×224**: 150,528 features → 500 componentes (99.67% redução!)

#### 4. **Último Resultado SVM Muito Baixo (59.74%)**
- **Possíveis causas**:
  - Configuração incorreta de tamanho de imagem
  - Hiperparâmetros não otimizados
  - Problema no carregamento de dados

---

## 🎯 Melhorias Propostas

### **Melhoria 1: Corrigir Tamanho de Imagem para Modelos Clássicos**

**Problema**: `IMG_SIZE_CLASSIC = (224, 224)` está muito grande

**Solução**: Ajustar para `(128, 128)` conforme documentação

```python
# src/config.py - LINHA 29
IMG_SIZE_CLASSIC = (128, 128)  # Tamanho otimizado para modelos clássicos
```

**Justificativa**:
- 128×128 = 49,152 features (balance entre qualidade e memória)
- 224×224 = 150,528 features (muito grande, desperdiça memória)
- 64×64 = 12,288 features (pode perder detalhes importantes)

**Impacto esperado**: 
- Redução de memória necessária
- Melhor aproveitamento do PCA
- Possível melhoria na acurácia

---

### **Melhoria 2: Ajustar Número de Componentes PCA Dinamicamente**

**Problema**: PCA fixo em 500 componentes não se adapta ao tamanho da imagem

**Solução**: Ajustar componentes PCA baseado no tamanho da imagem

```python
# src/config.py - NOVA CONFIGURAÇÃO
CLASSIC_PCA_COMPONENTS = None  # Auto: 95% variância (recomendado)
# OU ajustar dinamicamente:
# CLASSIC_PCA_COMPONENTS = 800  # Para 128×128 (melhor que 500)
# CLASSIC_PCA_COMPONENTS = 1000  # Para 224×224 (se necessário)
```

**Implementação sugerida** em `src/pipelines/classic.py`:

```python
# Ajustar componentes PCA baseado no tamanho da imagem
if IMG_SIZE_CLASSIC == (128, 128):
    suggested_components = 800  # Mais componentes para mais features
elif IMG_SIZE_CLASSIC == (224, 224):
    suggested_components = 1000
else:
    suggested_components = 500  # Padrão para 64×64

if CLASSIC_PCA_COMPONENTS is None:
    # Auto: 95% variância
    self.pca = PCA(n_components=0.95, random_state=42)
else:
    # Usar valor sugerido ou configurado
    n_components = min(CLASSIC_PCA_COMPONENTS, min(n_samples - 1, n_features))
    self.pca = PCA(n_components=n_components, random_state=42)
```

**Impacto esperado**:
- Melhor preservação de informação com imagens maiores
- Acurácia pode melhorar 2-5%

---

### **Melhoria 3: Ajustar Espaço de Busca de Hiperparâmetros do SVM**

**Problema**: Espaço de busca não considera tamanho da imagem

**Solução**: Ajustar gamma e C baseado no número de features

**Implementação sugerida** em `src/pipelines/classic.py`:

```python
# Ajustar espaço de busca baseado no número de features
n_features = self.X_train.shape[1]

if n_features > 1000:
    # Imagens maiores: gamma menor, C pode ser maior
    param_distributions = {
        'C': loguniform(0.1, 1000),  # Ampliado
        'gamma': loguniform(0.00001, 0.1),  # Range menor para imagens maiores
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': randint(2, 5),
        'class_weight': [None, 'balanced']
    }
else:
    # Imagens menores: espaço de busca padrão
    param_distributions = {
        'C': loguniform(0.01, 100),
        'gamma': loguniform(0.0001, 1),
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': randint(2, 5),
        'class_weight': [None, 'balanced']
    }
```

**Impacto esperado**:
- Melhor otimização de hiperparâmetros
- Acurácia pode melhorar 1-3%

---

### **Melhoria 4: Aumentar Número de Iterações do Random Search**

**Problema**: 50 iterações pode não ser suficiente para encontrar melhores hiperparâmetros

**Solução**: Aumentar para 100 iterações (ainda rápido para modelos clássicos)

**Implementação**:

```python
# No main.py ou ao chamar pipeline:
pipeline.train_svm(use_random_search=True, n_iter=100)  # Era 50
pipeline.train_random_forest(use_random_search=True, n_iter=100)  # Era 50
```

**Trade-off**:
- **Tempo**: ~2x mais tempo (ainda rápido: 4-5 segundos para SVM, 60s para RF)
- **Benefício**: Melhor chance de encontrar hiperparâmetros ótimos

**Impacto esperado**:
- Acurácia pode melhorar 1-2%

---

### **Melhoria 5: Testar Diferentes Estratégias de PCA**

**Opção A: PCA Auto (95% variância)**
```python
CLASSIC_PCA_COMPONENTS = None  # Auto: 95% variância
```

**Opção B: PCA com mais componentes para imagens maiores**
```python
# Para 128×128
CLASSIC_PCA_COMPONENTS = 800  # Mais componentes

# Para 224×224
CLASSIC_PCA_COMPONENTS = 1000  # Ainda mais componentes
```

**Opção C: Testar sem PCA (se tiver memória suficiente)**
```python
CLASSIC_USE_PCA = False  # Testar sem PCA
```

**Impacto esperado**:
- PCA Auto: Pode melhorar 1-3%
- Sem PCA: Pode melhorar 2-5% (mas usa muito mais memória)

---

### **Melhoria 6: Ajustar Random Forest para Imagens Maiores**

**Problema**: Espaço de busca pode não ser ideal para mais features

**Solução**: Ajustar `max_features` baseado no número de features

**Implementação sugerida**:

```python
n_features = self.X_train.shape[1]

if n_features > 1000:
    # Para muitas features, usar sqrt ou log2
    param_distributions = {
        'n_estimators': randint(100, 500),  # Mais árvores
        'max_depth': [None, 15, 20, 30, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2'],  # Remover None para muitas features
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
else:
    # Espaço de busca padrão
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
```

**Impacto esperado**:
- Acurácia pode melhorar 1-2%

---

## 📋 Plano de Implementação Recomendado

### **Fase 1: Correções Críticas (Alto Impacto, Baixo Risco)**

1. ✅ **Corrigir `IMG_SIZE_CLASSIC` para (128, 128)**
   - Impacto: Alto
   - Risco: Baixo
   - Tempo: 1 minuto

2. ✅ **Aumentar componentes PCA para 800 (para 128×128)**
   - Impacto: Médio-Alto
   - Risco: Baixo
   - Tempo: 1 minuto

### **Fase 2: Otimizações (Médio Impacto, Baixo Risco)**

3. ✅ **Aumentar iterações Random Search para 100**
   - Impacto: Médio
   - Risco: Baixo
   - Tempo: 2 minutos (configuração)

4. ✅ **Ajustar espaço de busca do SVM para imagens maiores**
   - Impacto: Médio
   - Risco: Baixo
   - Tempo: 5 minutos (código)

### **Fase 3: Testes Avançados (Alto Impacto, Médio Risco)**

5. ⚠️ **Testar PCA Auto (95% variância)**
   - Impacto: Alto
   - Risco: Médio (pode usar mais memória)
   - Tempo: 10 minutos (teste)

6. ⚠️ **Testar sem PCA (se tiver memória)**
   - Impacto: Alto
   - Risco: Alto (pode estourar memória)
   - Tempo: 15 minutos (teste)

---

## 🎯 Resultados Esperados

### **Cenário Conservador (Fase 1 + Fase 2)**

| Modelo | Resultado Atual | Resultado Esperado | Melhoria |
|--------|----------------|-------------------|----------|
| **SVM** | 64.38% | **70-72%** | +5-7% |
| **Random Forest** | 65.75% | **68-70%** | +2-4% |

### **Cenário Otimista (Fase 1 + Fase 2 + Fase 3)**

| Modelo | Resultado Atual | Resultado Esperado | Melhoria |
|--------|----------------|-------------------|----------|
| **SVM** | 64.38% | **72-75%** | +7-10% |
| **Random Forest** | 65.75% | **70-73%** | +4-7% |

---

## 🔧 Configurações Recomendadas

### **Configuração Otimizada para 128×128**

```python
# src/config.py

# Tamanho de imagem
IMG_SIZE_CLASSIC = (128, 128)  # ✅ CORRIGIDO

# PCA
CLASSIC_USE_PCA = True
CLASSIC_PCA_COMPONENTS = 800  # ✅ AUMENTADO (era 500)

# Random Search
# Aumentar n_iter para 100 no main.py
```

### **Configuração para Teste com PCA Auto**

```python
# src/config.py

CLASSIC_USE_PCA = True
CLASSIC_PCA_COMPONENTS = None  # ✅ AUTO: 95% variância
```

---

## 📊 Métricas de Sucesso

**Objetivo**: Melhorar acurácia dos modelos clássicos para competir melhor com CNN Simples (70.89%)

**Meta**:
- **SVM**: 70%+ (atual: 64.38%)
- **Random Forest**: 68%+ (atual: 65.75%)

**Critérios de Sucesso**:
- ✅ Acurácia aumenta em pelo menos 2%
- ✅ Tempo de treinamento permanece < 1 minuto
- ✅ Não estoura memória
- ✅ Resultados são consistentes entre execuções

---

## 🚀 Próximos Passos

1. **Implementar Fase 1** (correções críticas)
2. **Executar treinamento** e comparar resultados
3. **Se melhorou**: Implementar Fase 2
4. **Se não melhorou**: Investigar outras causas
5. **Testar Fase 3** apenas se tiver memória suficiente

---

**Data da Análise**: 11 de janeiro de 2026  
**Versão do Projeto**: 1.0.3
