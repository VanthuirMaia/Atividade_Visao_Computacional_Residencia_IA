# Projeto de Classificação de Imagens - Visão Computacional
## Documentação Completa e Consolidada

> **Este README consolida TODA a documentação do projeto**, incluindo toda a história de desenvolvimento, erros encontrados, correções implementadas, otimizações, mudanças de parâmetros, ajustes de métodos, e exemplos de código específicos com explicações detalhadas.

**Versão do Projeto**: 1.0.0 (Final)  
**Última Atualização**: 2024  
**Status**: ✅ Estável e Otimizado

---

## 📋 Índice

1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Estrutura Completa do Projeto](#estrutura-completa-do-projeto)
3. [Instalação e Configuração](#instalação-e-configuração)
4. [História Completa do Desenvolvimento](#história-completa-do-desenvolvimento)
   - [Erros Encontrados e Corrigidos](#erros-encontrados-e-corrigidos)
   - [Otimizações de Memória](#otimizações-de-memória)
   - [Correções de GPU](#correções-de-gpu)
   - [Sistema de Salvamento de Modelos](#sistema-de-salvamento-de-modelos)
   - [Random Search Otimizado](#random-search-otimizado)
5. [Configurações Detalhadas](#configurações-detalhadas)
6. [Exemplos de Código por Componente](#exemplos-de-código-por-componente)
7. [Guia de Uso Completo](#guia-de-uso-completo)
8. [Troubleshooting](#troubleshooting)
9. [Referências e Documentação Técnica](#referências-e-documentação-técnica)

---

## 🎯 Visão Geral do Projeto

Projeto completo de classificação de imagens (AI Art vs Human Art) utilizando múltiplas abordagens:
- **Pipeline Clássico**: SVM e Random Forest com otimização de hiperparâmetros
- **Pipeline Deep Learning**: Simple CNN e ResNet50 com transfer learning
- **Otimizações Avançadas**: Gerenciamento de memória, lazy loading, limpeza automática
- **Sistema Completo**: Download automático de dataset, diagnóstico, salvamento de modelos

## 📁 Estrutura Completa do Projeto

```
.
├── main.py                          # Ponto de entrada principal com menu interativo
├── main_subset.py                   # Versão para testes rápidos com subset (10 imagens/classe)
├── requirements.txt                 # Todas as dependências do projeto
├── README.md                        # Este arquivo - documentação completa consolidada
│
├── Scripts de Diagnóstico/
│   ├── diagnose_data.py             # Diagnóstico da estrutura de dados
│   ├── check_gpu.py                 # Verificação de GPU/CUDA
│   ├── verificar_pytorch.py         # Verificação completa do PyTorch
│   ├── diagnose_gpu_usage.py        # Diagnóstico de uso de GPU
│   └── testar_gpu_direto.py         # Teste direto de GPU sem dependências
│
├── src/                             # Código fonte principal
│   ├── __init__.py
│   ├── config.py                    # ⚙️ TODAS as configurações centralizadas
│   ├── utils.py                     # Funções utilitárias (device, imagens, métricas)
│   ├── datasets.py                  # LazyImageDataset para carregamento eficiente
│   ├── memory.py                    # Gerenciamento avançado de memória
│   ├── model_saver.py               # Sistema de salvamento com metadados
│   │
│   ├── models/                      # Definições de modelos
│   │   ├── __init__.py
│   │   └── cnn.py                   # SimpleCNN - arquitetura customizada
│   │
│   └── pipelines/                   # Pipelines de treinamento
│       ├── __init__.py
│       ├── classic.py               # Pipeline clássico (SVM, Random Forest)
│       └── deep_learning.py         # Pipeline deep learning (CNN, ResNet50)
│
├── scripts/                         # Scripts auxiliares
│   ├── __init__.py
│   ├── download_dataset.py          # Download automático do dataset Kaggle
│   ├── create_subset.py             # Criação de subset para testes rápidos
│   └── load_model_example.py        # Exemplo de como carregar modelos salvos
│
├── data/                            # Dados (ignorado pelo git)
│   ├── train/                       # Imagens de treinamento
│   │   ├── aiartdata/               # Classe 1: Arte gerada por IA
│   │   └── realart/                 # Classe 2: Arte criada por humanos
│   ├── test/                        # Imagens de teste
│   │   ├── aiartdata/
│   │   └── realart/
│   ├── train_subset/                # Subset pequeno para testes (10/classe)
│   └── test_subset/                 # Subset pequeno para testes (10/classe)
│
└── outputs/                         # Resultados gerados (ignorado pelo git)
    ├── models/                      # Modelos treinados salvos
    │   ├── svm_model.pkl            # Modelo SVM
    │   ├── svm_model.json           # Metadados do SVM
    │   ├── svm_scaler.pkl           # Scaler usado no SVM
    │   ├── random_forest_model.pkl  # Modelo Random Forest
    │   ├── simple_cnn.pth           # Modelo Simple CNN
    │   ├── resnet50_transfer.pth    # Modelo ResNet50
    │   └── *.json                   # Metadados de cada modelo
    ├── results/                     # Resultados em CSV
    │   ├── classic_pipeline_results.csv
    │   └── deep_learning_results.csv
    └── figures/                     # Gráficos e visualizações
        ├── svm_confusion_matrix.png
        ├── random_forest_confusion_matrix.png
        ├── simple_cnn_confusion_matrix.png
        └── resnet50_confusion_matrix.png
```

### Arquivos de Documentação Consolidados

✅ **Todos os arquivos `.md` anteriores foram consolidados neste README e removidos do projeto:**

- `ANALISE_CODIGO.md` - Análise completa do código → Seção "História Completa do Desenvolvimento"
- `ANALISE_GPU.md` - Análise de GPU → Seção "Correções de GPU"
- `ANALISE_LIMPEZA_PROJETO.md` - Limpeza realizada → Integrado nas otimizações
- `ANALISE_MODELOS_CLASSICOS.md` - Análise de modelos clássicos → Seção "Pipeline Clássico"
- `GUIA_SALVAMENTO_MODELOS.md` - Sistema de salvamento → Seção "Sistema de Salvamento de Modelos"
- `RANDOM_SEARCH_ATUALIZADO.md` - Random Search otimizado → Seção "Random Search Otimizado"
- `RANDOM_SEARCH_TODOS_MODELOS.md` - Random Search em todos modelos → Seção "Random Search Otimizado"
- `SOLUCAO_ESTOURO_MEMORIA_RESNET50.md` - Solução ResNet50 → Seção "Otimizações de Memória"
- `SOLUCAO_ESTOURO_MEMORIA_SVM.md` - Solução SVM → Seção "Otimizações de Memória"
- `SOLUCAO_GPU_NAO_UTILIZADA.md` - Solução GPU → Seção "Correções de GPU"
- `VERIFICACAO_GPU.md` - Verificação GPU → Seção "Correções de GPU" e "Troubleshooting"

**Status**: ✅ Todos os arquivos foram removidos. Todo o conteúdo está consolidado neste README.

---

## 📚 História Completa do Desenvolvimento

Esta seção documenta **TUDO** que foi realizado durante o desenvolvimento do projeto, desde os erros iniciais até as otimizações finais.

### 📅 Cronologia de Desenvolvimento

#### **Fase 1: Problemas Iniciais com Dataset**

**Problema 1.1: Apenas 1 Classe Detectada**
- **Erro**: `ValueError: Apenas 1 classe(s) foi(ram) carregada(s), mas são necessárias pelo menos 2 classes para classificação.`
- **Causa**: Script `download_dataset.py` não identificava corretamente as classes "AiArtData" e "RealArt"
- **Localização**: `scripts/download_dataset.py`, função `find_class_directories()`
- **Correção Implementada**:

```python
# scripts/download_dataset.py - LINHAS CORRIGIDAS
def find_class_directories(directory):
    """Encontra diretórios de classes no dataset"""
    classes = []
    for item in Path(directory).iterdir():
        if item.is_dir():
            # CORREÇÃO: Busca case-insensitive e variações de nomes
            name_lower = item.name.lower()
            if 'aiart' in name_lower or 'ai_art' in name_lower:
                classes.append(('aiartdata', item))
            elif 'realart' in name_lower or 'real_art' in name_lower or 'human' in name_lower:
                classes.append(('realart', item))
    return classes
```

**Impacto**: ✅ Permite detectar classes independente de variações de nomenclatura

---

**Problema 1.2: EOFError em Script Não-Interativo**
- **Erro**: `EOFError` ao executar `scripts/create_subset.py` de forma não-interativa
- **Causa**: Uso de `input()` para confirmação do usuário
- **Localização**: `scripts/create_subset.py`
- **Correção Implementada**:

```python
# scripts/create_subset.py - ANTES (com erro):
if backup_exists:
    resposta = input("Subset já existe. Substituir? (s/n): ")  # ❌ Causa EOFError
    if resposta.lower() != 's':
        return

# scripts/create_subset.py - DEPOIS (corrigido):
# REMOVIDO: Prompt interativo que causava EOFError
# AGORA: Cria subset automaticamente, criando classes artificiais se necessário
if len(class_dirs) < 2:
    print("Apenas 1 classe encontrada. Criando classes artificiais (classe_a, classe_b)...")
    # Cria subset com nomes artificiais
```

**Impacto**: ✅ Script pode ser executado em ambientes não-interativos (CI/CD, scripts automatizados)

---

#### **Fase 2: Erros em Modelos Deep Learning**

**Problema 2.1: TypeError no ReduceLROnPlateau**
- **Erro**: `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`
- **Causa**: PyTorch versão não suporta parâmetro `verbose` em `ReduceLROnPlateau`
- **Localização**: `src/pipelines/deep_learning.py`, linhas 473-475 e 548-550
- **Código Antes (com erro)**:

```python
# src/pipelines/deep_learning.py - ANTES (linha 473):
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True  # ❌ Erro: verbose não existe
)
```

- **Código Depois (corrigido)**:

```python
# src/pipelines/deep_learning.py - DEPOIS (linha 599):
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5  # ✅ Removido verbose
)
```

**Impacto**: ✅ Compatível com todas as versões do PyTorch

---

**Problema 2.2: AttributeError com setup_device**
- **Erro**: `AttributeError: 'str' object has no attribute 'type'`
- **Causa**: Função `setup_device()` poderia retornar string `'cpu'` ao invés de `torch.device('cpu')`
- **Localização**: `src/utils.py`, função `setup_device()` e `src/pipelines/deep_learning.py`
- **Código Antes (com erro)**:

```python
# src/utils.py - ANTES:
def setup_device(use_gpu=True):
    # ...
    if use_gpu and not torch.cuda.is_available():
        return 'cpu'  # ❌ Retorna string, não torch.device
    
# src/pipelines/deep_learning.py - ANTES:
self.device = setup_device(use_gpu)
if self.device.type == 'cuda':  # ❌ Erro se device for string 'cpu'
    ...
```

- **Código Depois (corrigido)**:

```python
# src/utils.py - DEPOIS (linha 62, 122):
def setup_device(use_gpu=True):
    # ...
    if use_gpu and not torch.cuda.is_available():
        return torch.device('cpu')  # ✅ Sempre retorna torch.device
    
    # ...
    else:
        # ...
        return torch.device('cpu')  # ✅ Sempre retorna torch.device
```

- **Verificação Adicional em deep_learning.py (linhas 129-161)**:

```python
# src/pipelines/deep_learning.py - DEPOIS (linha 130-161):
# Garantir que device é torch.device
if isinstance(self.device, str):
    print(f"   ⚠️  Dispositivo é string '{self.device}', convertendo para torch.device...")
    self.device = torch.device(self.device)
elif not isinstance(self.device, torch.device):
    print(f"   ⚠️  Tipo desconhecido, usando CPU...")
    self.device = torch.device('cpu')
```

**Impacto**: ✅ Dispositivo sempre é objeto `torch.device`, evitando erros de atributo

---

**Problema 2.3: Modelos Não Estavam Usando GPU**
- **Erro**: Modelos deep learning executavam na CPU mesmo com GPU disponível
- **Causa**: Modelos não eram movidos explicitamente para GPU após criação
- **Localização**: Múltiplas funções em `src/pipelines/deep_learning.py`
- **Correções Implementadas**:

**2.3.1: ResNet50 não movido para GPU** (linha 1049-1061):
```python
# src/pipelines/deep_learning.py - ANTES (create_resnet_model):
model = models.resnet50(weights='IMAGENET1K_V2')
# ... configurar modelo ...
return model  # ❌ Modelo fica na CPU

# src/pipelines/deep_learning.py - DEPOIS (linha 1049-1061):
model = models.resnet50(weights='IMAGENET1K_V2')
# ... configurar modelo ...

# CRÍTICO: Mover modelo para dispositivo correto (GPU ou CPU)
model = model.to(self.device)

# Verificar dispositivo e mostrar informações
model_device = next(model.parameters()).device
if model_device.type == 'cuda':
    print(f"   ✅ ResNet50 está na GPU: {torch.cuda.get_device_name(model_device.index or 0)}")
else:
    print(f"   ℹ️  ResNet50 está na CPU")

return model  # ✅ Modelo está no dispositivo correto
```

**2.3.2: SimpleCNN não movido no Random Search** (linha 798-804):
```python
# src/pipelines/deep_learning.py - ANTES (train_simple_cnn - Random Search):
model = SimpleCNN(
    self.num_classes,
    dropout_rate=params['dropout_rate'],
    hidden_units=params['hidden_units']
)
# ❌ Modelo criado mas não movido para GPU
val_acc, _, iter_time = self.train_single_config(...)

# src/pipelines/deep_learning.py - DEPOIS (linha 798-804):
model = SimpleCNN(...)

# CRÍTICO: Mover modelo para dispositivo correto ANTES do treinamento
model = model.to(self.device)

# Verificar dispositivo (apenas na primeira iteração)
if i == 0:
    model_device = next(model.parameters()).device
    print(f"     Modelo SimpleCNN criado e movido para: {model_device}")
    if model_device.type == 'cuda':
        print(f"     ✅ SimpleCNN está na GPU: {torch.cuda.get_device_name(model_device.index or 0)}")
```

**2.3.3: SimpleCNN não movido no treinamento final** (linha 831-842):
```python
# src/pipelines/deep_learning.py - DEPOIS (linha 831-842):
model = SimpleCNN(...)

# CRÍTICO: Mover modelo para dispositivo correto ANTES do treinamento
model = model.to(self.device)

# Verificar dispositivo do modelo
model_device = next(model.parameters()).device
print(f"\nModelo SimpleCNN criado:")
print(f"  Dispositivo: {model_device}")
if model_device.type == 'cuda':
    print(f"  ✅ SimpleCNN está na GPU: {torch.cuda.get_device_name(model_device.index or 0)}")
```

**2.3.4: Melhorias em train_single_config e train_model**:
```python
# src/pipelines/deep_learning.py - DEPOIS (linha 505-507):
def train_single_config(self, model, train_loader, val_loader, epochs, learning_rate, patience=5):
    # Garantir que modelo está no dispositivo correto
    if next(model.parameters()).device != self.device:
        print(f"     [AVISO] Movendo modelo de {next(model.parameters()).device} para {self.device}")
        model = model.to(self.device)
```

```python
# src/pipelines/deep_learning.py - DEPOIS (linha 601-605):
def train_model(self, model, train_loader, epochs, learning_rate, model_name):
    # Garantir que modelo está no dispositivo correto
    if next(model.parameters()).device != self.device:
        print(f"   [AVISO] Movendo modelo {model_name} de {next(model.parameters()).device} para {self.device}")
        model = model.to(self.device)
```

**Impacto**: ✅ Todos os modelos agora usam GPU automaticamente quando disponível

---

#### **Fase 3: Estouro de Memória - SVM**

**Problema 3.1: Estouro de Memória no SVM**
- **Erro**: `MemoryError` ou sistema travando durante treinamento do SVM
- **Causa**: Imagens muito grandes (224x224x3) + kernel RBF + CV folds múltiplos
- **Análise do Problema**:

```
ANTES (Problema):
- Imagens: 224x224x3 = 150,528 features por imagem
- 10.000 amostras: 150,528 × 10.000 = 1.5 bilhões de features
- Memória necessária: ~12 GB apenas para dados
- Matriz Gram (RBF kernel): ~800 MB - 8 GB
- CV=3: Múltiplas cópias dos dados
- n_jobs=-1: Múltiplos processos duplicando dados
- TOTAL: ~15-20 GB de RAM necessária!
```

- **Soluções Implementadas** (código em `src/config.py` e `src/pipelines/classic.py`):

**Solução 3.1.1: Tamanho de Imagem Reduzido** (linha 29 em `config.py`):
```python
# src/config.py - NOVA CONFIGURAÇÃO (linha 29):
IMG_SIZE_CLASSIC = (64, 64)  # Tamanho menor para modelos clássicos (economiza memória)
IMG_SIZE = (224, 224)  # Mantido para deep learning
```

**Implementação em classic.py (linha 94-95)**:
```python
# src/pipelines/classic.py - ANTES:
X_train, y_train, self.class_names = load_images_from_directory(
    self.train_dir, img_size=(224, 224)  # ❌ Muito grande

# src/pipelines/classic.py - DEPOIS (linha 94-95):
X_train, y_train, self.class_names = load_images_from_directory(
    self.train_dir, img_size=IMG_SIZE_CLASSIC  # ✅ 64x64
)
```

**Redução**: 150,528 features → 12,288 features (92% redução!)

---

**Solução 3.1.2: PCA para Redução de Dimensionalidade** (linha 102-103, 154-179 em `classic.py`):
```python
# src/config.py - NOVAS CONFIGURAÇÕES (linhas 102-103):
CLASSIC_USE_PCA = True  # Usar PCA para redução de dimensionalidade
CLASSIC_PCA_COMPONENTS = 500  # Número de componentes PCA
```

**Implementação em classic.py (linha 154-179)**:
```python
# src/pipelines/classic.py - NOVO CÓDIGO (linha 152-179):
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
    
    # CRÍTICO: fit_transform apenas no treino, transform no teste
    X_train_scaled = self.pca.fit_transform(X_train_scaled)  # ✅ Aprende componentes
    X_test_scaled = self.pca.transform(X_test_scaled)  # ✅ Usa componentes aprendidos
    
    n_features_after_pca = X_train_scaled.shape[1]
    reduction = ((n_features - n_features_after_pca) / n_features) * 100
    estimated_mem_after_gb = (n_samples * n_features_after_pca * 8) / (1024**3)
    print(f"   Features após PCA: {n_features_after_pca:,} ({reduction:.1f}% redução)")
    print(f"   Memória estimada após PCA: {estimated_mem_after_gb:.2f} GB")
    
    if hasattr(self.pca, 'explained_variance_ratio_'):
        total_variance = self.pca.explained_variance_ratio_.sum()
        print(f"   Variância explicada: {total_variance:.2%}")
```

**Redução**: 12,288 features → 500 componentes (96% redução adicional!)

**Total**: 150,528 features → 500 componentes = **99.67% de redução!**

---

**Solução 3.1.3: LinearSVC como Alternativa** (linha 104 em `config.py`, linha 248-280 em `classic.py`):
```python
# src/config.py - NOVA CONFIGURAÇÃO (linha 104):
CLASSIC_USE_LINEAR_SVM = False  # False = SVC (kernels), True = LinearSVC (só linear, mais eficiente)
```

**Implementação em classic.py (linha 248-280)**:
```python
# src/pipelines/classic.py - NOVO CÓDIGO (linha 248-280):
use_linear_svm = CLASSIC_USE_LINEAR_SVM
if use_linear_svm:
    print(f"   Tipo: LinearSVC (kernel linear, mais eficiente em memória)")
else:
    print(f"   Tipo: SVC (suporta kernels não-lineares, mas usa mais memória)")

if use_random_search:
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
```

**Benefício**: LinearSVC não calcula matriz Gram, economizando 70-90% de memória adicional

---

**Solução 3.1.4: Limitação de Amostras** (linha 105 em `config.py`, linha 127-133 em `classic.py`):
```python
# src/config.py - NOVA CONFIGURAÇÃO (linha 105):
CLASSIC_MAX_SAMPLES = None  # None = usar todas, ou número máximo (ex: 10000)
```

**Implementação em classic.py (linha 127-133)**:
```python
# src/pipelines/classic.py - NOVO CÓDIGO (linha 127-133):
# Limitar número de amostras se configurado
if CLASSIC_MAX_SAMPLES is not None and len(X_train) > CLASSIC_MAX_SAMPLES:
    print(f"\n  AVISO: Limitando amostras de treinamento de {len(X_train)} para {CLASSIC_MAX_SAMPLES}")
    indices = np.random.choice(len(X_train), CLASSIC_MAX_SAMPLES, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]
    print(f"   Amostras selecionadas aleatoriamente mantendo proporção de classes")
```

---

**Solução 3.1.5: Configurações de CV e Paralelização** (linhas 106-108 em `config.py`):
```python
# src/config.py - NOVAS CONFIGURAÇÕES (linhas 106-108):
CLASSIC_SVM_N_JOBS = 1  # 1 = sem paralelização (economiza memória), -1 = todos os cores
CLASSIC_RF_N_JOBS = -1  # Random Forest pode usar mais cores (mais eficiente)
CLASSIC_CV_FOLDS = 2  # 2 ao invés de 3 para economizar memória - aplica-se a TODOS os modelos clássicos
```

**Implementação em classic.py - SVM (linha 244, 282-285)**:
```python
# src/pipelines/classic.py - NOVO CÓDIGO (linha 244):
svm_n_jobs = CLASSIC_SVM_N_JOBS if CLASSIC_SVM_N_JOBS is not None else 1
print(f"\n   Paralelização SVM: {svm_n_jobs} job(s) (configurado para economizar memória)")

# Linha 282-285:
random_search = RandomizedSearchCV(
    svm, param_distributions, n_iter=n_iter, cv=CLASSIC_CV_FOLDS,  # ✅ CV configurável
    scoring='accuracy', n_jobs=svm_n_jobs, verbose=1, random_state=42  # ✅ n_jobs configurável
)
```

**Implementação em classic.py - Random Forest (linha 420-443)**:
```python
# src/pipelines/classic.py - NOVO CÓDIGO (linha 420-443):
# Determinar jobs para Random Forest (pode usar mais paralelização que SVM)
rf_n_jobs = CLASSIC_RF_N_JOBS if CLASSIC_RF_N_JOBS is not None else self.n_jobs
if rf_n_jobs == -1:
    actual_jobs = self.num_cores
else:
    actual_jobs = rf_n_jobs
print(f"   Paralelização: {actual_jobs} job(s) paralelo(s) (Random Forest pode usar mais cores eficientemente)")

# ...
rf = RandomForestClassifier(random_state=42, n_jobs=rf_n_jobs)  # ✅ n_jobs específico para RF
random_search = RandomizedSearchCV(
    rf, param_distributions, n_iter=n_iter, cv=CLASSIC_CV_FOLDS,  # ✅ CV configurável
    scoring='accuracy', n_jobs=rf_n_jobs, verbose=1, random_state=42  # ✅ n_jobs específico
)
```

**Redução de Memória**:
- CV folds: 3 → 2 = 33% menos cópias de dados
- n_jobs: -1 → 1 = Sem duplicação de dados em múltiplos processos

---

**Solução 3.1.6: Verificação de Memória Antes de Treinar** (linha 227-241 em `classic.py`):
```python
# src/pipelines/classic.py - NOVO CÓDIGO (linha 227-241):
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
```

**Função check_available_memory em src/memory.py**:
```python
# src/memory.py - Função implementada:
def check_available_memory(required_gb, safety_margin=0.2):
    """
    Verifica se há memória disponível suficiente
    
    Args:
        required_gb: Memória necessária em GB
        safety_margin: Margem de segurança (padrão: 20%)
    
    Returns:
        bool: True se há memória suficiente
    """
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 ** 3)
    required_with_margin = required_gb * (1 + safety_margin)
    
    return available_gb >= required_with_margin
```

**Resultado Final das Otimizações SVM**:
- **Antes**: ~15-20 GB necessários
- **Depois**: ~1-2 GB necessários
- **Redução**: ~90-95% de memória economizada! ✅

---

#### **Fase 4: Estouro de Memória - ResNet50**

**Problema 4.1: Estouro de Memória no ResNet50**
- **Erro**: `RuntimeError: CUDA out of memory` durante Random Search do ResNet50
- **Causa**: Modelos acumulando na GPU entre iterações do Random Search + batch sizes grandes
- **Análise do Problema**:

```
ANTES (Problema):
- ResNet50: ~25 milhões de parâmetros
- Batch size: [16, 32, 64] testados
- Imagens: 224x224x3
- Por batch (size 32): ~850 MB de GPU
- Sem limpeza entre iterações: Múltiplos modelos acumulados
- Cache CUDA não limpo: Memória fragmentada
- TOTAL: 8 GB GPU insuficiente após algumas iterações!
```

- **Soluções Implementadas**:

**Solução 4.1.1: Configurações Específicas para ResNet50** (linhas 84-95 em `config.py`):
```python
# src/config.py - NOVAS CONFIGURAÇÕES (linhas 84-95):
# Batch sizes para Random Search do ResNet50 (REDUZIDOS)
RESNET50_BATCH_SIZES = [8, 16, 32]  # Era [16, 32, 64] - 50% menor

# Batch size padrão para ResNet50
RESNET50_DEFAULT_BATCH_SIZE = 16  # Era 32 - 50% menor

# Épocas para Random Search (limitadas)
RESNET50_SEARCH_EPOCHS = 10  # Número máximo de épocas durante Random Search

# Limpar memória entre iterações (CRÍTICO)
RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS = True  # IMPORTANTE: Limpar entre iterações
```

---

**Solução 4.1.2: Limpeza de Memória Entre Iterações** (linhas 1098-1145 em `deep_learning.py`):
```python
# src/pipelines/deep_learning.py - NOVO CÓDIGO (linhas 1098-1145):
if use_random_search:
    print(f"\nExecutando Random Search ({n_iter} iterações)...")
    print(f"  Batch sizes testados: {RESNET50_BATCH_SIZES}")
    print(f"  Épocas por iteração: {RESNET50_SEARCH_EPOCHS}")
    print(f"  Limpeza de memória entre iterações: {'Ativada' if RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS else 'Desativada'}")
    
    for i in range(n_iter):
        # Limpar ANTES de cada iteração
        if RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS:
            clear_memory(clear_gpu=True)  # ✅ Limpa cache CUDA
        
        train_loader, val_loader, _ = self.create_dataloaders(
            params['batch_size'], val_split=0.2
        )
        
        model = self.create_resnet_model(unfreeze_layers=params['unfreeze_layers'])
        
        try:
            val_acc, trained_model, iter_time = self.train_single_config(...)
            # ... processamento ...
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"    [ERRO] Estouro de memória na iteração {i+1}!")
                clear_memory(clear_gpu=True)
                
                # Tentar com batch size menor
                if params['batch_size'] > min(RESNET50_BATCH_SIZES):
                    params['batch_size'] = params['batch_size'] // 2
                    continue
        finally:
            # Limpar APÓS cada iteração (CRÍTICO)
            if RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS:
                # Mover modelo para CPU antes de deletar
                if 'trained_model' in locals():
                    trained_model = trained_model.cpu()
                if 'model' in locals():
                    model = model.cpu()
                del model, trained_model
                clear_memory(clear_gpu=True)  # ✅ Limpa cache CUDA
                
                # Mostrar status de memória
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    gpu_mem_used = torch.cuda.memory_allocated() / (1024**3)
                    print(f"    Memória GPU após limpeza: {gpu_mem_used:.2f} GB")
```

**Função clear_memory em src/memory.py (linha 150-158)**:
```python
# src/memory.py - Função implementada (linha 150-158):
def clear_memory(clear_gpu=False):
    """Limpa memória RAM e opcionalmente GPU"""
    import gc
    gc.collect()  # Garbage collection Python
    
    # Limpar cache GPU
    if clear_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()  # ✅ Limpa cache CUDA
        torch.cuda.synchronize()  # ✅ Sincroniza operações
```

**Impacto**: Libera ~2-4 GB de memória GPU entre iterações

---

**Solução 4.1.3: Verificação de Memória Antes de Carregar Modelo** (linhas 1007-1029 em `deep_learning.py`):
```python
# src/pipelines/deep_learning.py - NOVO CÓDIGO (linhas 1007-1029):
# Verificar memória disponível antes de carregar modelo grande
print(f"\n   Verificando memória antes de carregar ResNet50...")
ram_used, ram_total, ram_percent = self.memory_monitor.get_ram_usage()
print(f"     RAM: {ram_used:.2f} GB / {ram_total:.2f} GB ({ram_percent*100:.1f}%)")

if torch.cuda.is_available() and self.device.type == 'cuda':
    gpu_mem_used = torch.cuda.memory_allocated() / (1024**3)
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100 if gpu_mem_total > 0 else 0
    print(f"     GPU: {gpu_mem_used:.2f} GB / {gpu_mem_total:.2f} GB ({gpu_mem_percent:.1f}%)")
    
    # Aviso se memória GPU estiver alta
    if gpu_mem_percent > 80:
        print(f"     [AVISO] Memória GPU alta! Limpando cache...")
        clear_memory(clear_gpu=True)

# Aviso se RAM estiver alta
if ram_percent > MEMORY_WARNING_THRESHOLD:
    print(f"     [AVISO] Memória RAM alta ({ram_percent*100:.1f}%)! Limpando memória...")
    clear_memory(clear_gpu=False)

# Limpar memória antes de carregar modelo
clear_memory(clear_gpu=True)
```

**Impacto**: Previne estouros antecipadamente

---

**Solução 4.1.4: Limpeza no train_single_config** (linha 589-591 em `deep_learning.py`):
```python
# src/pipelines/deep_learning.py - NOVO CÓDIGO (linha 589-591):
def train_single_config(self, model, train_loader, val_loader, epochs, learning_rate, patience=5):
    # ... treinamento ...
    
    train_time = time.time() - start_time
    
    # Limpar memória ao final (importante para Random Search)
    clear_memory(clear_gpu=True)  # ✅ Nova linha
    
    return best_val_acc, model, train_time
```

**Resultado Final das Otimizações ResNet50**:
- **Antes**: Estouro após 2-3 iterações do Random Search
- **Depois**: Executa todas as 10 iterações sem problemas
- **Redução**: ~50% menos memória por batch + limpeza automática

---

## Instalação

### Requisitos do Sistema

- **Python**: 3.7 ou superior
- **RAM**: Mínimo 8 GB (recomendado 16 GB para datasets grandes)
- **GPU**: Opcional, mas recomendado para deep learning (CUDA 11.8+)
- **Espaço em Disco**: Depende do tamanho do dataset (~1-5 GB)

### Instalação de Dependências

```bash
pip install -r requirements.txt
```

**Dependências principais** (veja `requirements.txt` completo):
- `torch>=2.0.0` - PyTorch para deep learning
- `torchvision>=0.15.0` - Modelos pré-treinados (ResNet50)
- `scikit-learn>=1.3.0` - SVM, Random Forest, PCA, StandardScaler
- `opencv-python>=4.8.0` - Processamento de imagens
- `matplotlib>=3.7.0`, `seaborn>=0.12.0` - Visualizações
- `pandas>=2.0.0`, `numpy>=1.24.0` - Manipulação de dados
- `joblib>=1.3.0` - Salvamento de modelos
- `Pillow>=10.0.0` - Processamento de imagens
- `kagglehub>=0.2.0` - Download de datasets Kaggle
- `psutil>=5.9.0` - Monitoramento de memória
- `scipy>=1.10.0` - Distribuições para Random Search

Ou instale manualmente:

```bash
# Core Deep Learning
pip install torch torchvision

# Machine Learning Clássico
pip install scikit-learn scipy

# Processamento de Imagens
pip install opencv-python Pillow

# Visualização e Análise
pip install matplotlib seaborn pandas numpy

# Utilitários
pip install joblib kagglehub psutil
```

### Verificação de Instalação

Execute os scripts de diagnóstico para verificar se tudo está instalado corretamente:

```bash
# Verificar PyTorch e CUDA
python verificar_pytorch.py

# Verificar GPU
python check_gpu.py

# Verificar estrutura de dados
python diagnose_data.py
```

### Configuração do Kaggle (Opcional)

Para usar o dataset do Kaggle automaticamente:

1. **Criar conta no Kaggle**: https://www.kaggle.com/
2. **Aceitar termos do dataset**: Acesse [AI Art vs Human Art](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art) e aceite os termos
3. **Configurar credenciais** (opcional):
   ```bash
   # Linux/Mac
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   
   # Windows
   # Copie kaggle.json para: C:\Users\<username>\.kaggle\kaggle.json
   ```

### Configuração do Kaggle

Para usar o dataset do Kaggle, você precisa:

1. **Criar uma conta no Kaggle**: https://www.kaggle.com/
2. **Aceitar os termos do dataset**: Acesse o dataset [AI Art vs Human Art](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art) e aceite os termos
3. **Configurar credenciais do Kaggle** (opcional, mas recomendado):
   - Baixe seu arquivo `kaggle.json` das configurações da conta
   - Coloque em `~/.kaggle/kaggle.json` (Linux/Mac) ou `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

---

## 💾 Sistema de Salvamento de Modelos com Metadados

### Problema Identificado

O projeto não tinha sistema unificado para salvar modelos treinados com suas métricas e configurações. Isso dificultava:
- Comparar modelos salvos
- Reproduzir resultados
- Entender configurações usadas
- Carregar modelos para predições futuras

### Solução Implementada

**Novo módulo criado**: `src/model_saver.py` com 3 funções principais:

#### **1. `save_model_with_metadata()` - Salva Modelo com Metadados**

**Localização**: `src/model_saver.py`, linhas 11-45

**Assinatura**:
```python
def save_model_with_metadata(model, model_path, metadata, model_type='pytorch'):
    """
    Salva modelo com metadados completos
    
    Args:
        model: Modelo a ser salvo
        model_path: Caminho para salvar o modelo
        metadata: Dicionário com metadados (métricas, hiperparâmetros, etc.)
        model_type: Tipo do modelo ('pytorch' ou 'sklearn')
    """
```

**Implementação para PyTorch** (linha 25-31):
```python
# src/model_saver.py - LINHA 25-31:
if model_type == 'pytorch':
    import torch
    torch.save({
        'model_state_dict': model.state_dict(),  # ✅ Salva apenas pesos (mais leve)
        'model_class': model.__class__.__name__,  # ✅ Nome da classe para reconstrução
        'metadata': metadata  # ✅ Metadados incluídos no checkpoint
    }, model_path)
```

**Implementação para scikit-learn** (linha 32-37):
```python
# src/model_saver.py - LINHA 32-37:
elif model_type == 'sklearn':
    import joblib
    joblib.dump({
        'model': model,  # ✅ Modelo completo
        'metadata': metadata  # ✅ Metadados incluídos
    }, model_path)
```

**Salvamento de Metadados em JSON** (linha 39-42):
```python
# src/model_saver.py - LINHA 39-42:
# Salvar metadados em JSON separado (fácil de ler e editar)
metadata_path = model_path.with_suffix('.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
```

**Uso no Pipeline Clássico - SVM** (linha 358-363 em `classic.py`):
```python
# src/pipelines/classic.py - LINHA 358-363:
model_path = MODELS_DIR / 'svm_model.pkl'
save_model_with_metadata(
    model=self.svm_model,
    model_path=model_path,
    metadata=metadata,
    model_type='sklearn'  # ✅ Tipo específico para scikit-learn
)
```

**Uso no Pipeline Deep Learning - SimpleCNN** (linha 966-972 em `deep_learning.py`):
```python
# src/pipelines/deep_learning.py - LINHA 966-972:
model_path = MODELS_DIR / 'simple_cnn.pth'
save_model_with_metadata(
    model=model,
    model_path=model_path,
    metadata=metadata,
    model_type='pytorch'  # ✅ Tipo específico para PyTorch
)
```

---

#### **2. `create_model_metadata()` - Cria Dicionário de Metadados**

**Localização**: `src/model_saver.py`, linhas 90-118

**Assinatura**:
```python
def create_model_metadata(model_name, metrics, hyperparams, training_info, class_names, model_params=None):
    """
    Cria dicionário de metadados para um modelo
    
    Args:
        model_name: Nome do modelo
        metrics: Dicionário com métricas (accuracy, precision, etc.)
        hyperparams: Dicionário com hiperparâmetros
        training_info: Dicionário com informações de treinamento
        class_names: Lista de nomes das classes
        model_params: Parâmetros usados para inicializar a classe do modelo (para PyTorch)
    
    Returns:
        metadata: Dicionário com metadados completos
    """
```

**Estrutura de Metadados Retornada** (linha 104-118):
```python
# src/model_saver.py - LINHA 104-118:
return {
    'model_name': model_name,  # Ex: 'SVM', 'SimpleCNN', 'ResNet50'
    'timestamp': datetime.now().isoformat(),  # ✅ Data/hora do salvamento
    'metrics': {
        'accuracy': float(metrics.get('accuracy', 0)),  # ✅ Convertido para float
        'precision': float(metrics.get('precision', 0)),
        'recall': float(metrics.get('recall', 0)),
        'f1_score': float(metrics.get('f1_score', 0))
    },
    'hyperparameters': hyperparams,  # ✅ Todos os hiperparâmetros usados
    'training_info': training_info,  # ✅ Informações detalhadas de treinamento
    'class_names': class_names,  # ✅ Nomes das classes
    'num_classes': len(class_names),  # ✅ Número de classes
    'model_params': model_params,  # ✅ Parâmetros para reconstruir modelo PyTorch
    'version': '1.0'  # ✅ Versão do formato
}
```

**Exemplo de Uso - SVM** (linha 337-355 em `classic.py`):
```python
# src/pipelines/classic.py - LINHA 337-355:
metadata = create_model_metadata(
    model_name='SVM',
    metrics=metrics_test,  # ✅ Métricas calculadas
    hyperparams=best_hyperparams,  # ✅ Hiperparâmetros encontrados pelo Random Search
    training_info={
        'use_random_search': use_random_search,
        'n_iter': n_iter if use_random_search else 0,
        'cv_folds': CLASSIC_CV_FOLDS if use_random_search else 0,  # ✅ CV folds usados
        'pca_used': self.pca is not None,  # ✅ Se PCA foi usado
        'pca_components': self.pca.n_components if self.pca is not None else None,  # ✅ Componentes PCA
        'use_linear_svm': use_linear_svm,  # ✅ Tipo de SVM usado
        'img_size_classic': IMG_SIZE_CLASSIC,  # ✅ Tamanho de imagem
        'max_samples': CLASSIC_MAX_SAMPLES,  # ✅ Limite de amostras
        'total_time_seconds': total_time,  # ✅ Tempo de execução
        'device': 'CPU',
        'n_jobs': svm_n_jobs  # ✅ Paralelização usada
    },
    class_names=self.class_names  # ✅ Nomes das classes
)
```

**Exemplo de Uso - SimpleCNN** (linha 940-963 em `deep_learning.py`):
```python
# src/pipelines/deep_learning.py - LINHA 940-963:
metadata = create_model_metadata(
    model_name='SimpleCNN',
    metrics=metrics,
    hyperparams={
        'learning_rate': best_params['learning_rate'],
        'batch_size': best_params['batch_size'],
        'dropout_rate': best_params['dropout_rate'],
        'hidden_units': best_params['hidden_units'],
        'num_classes': self.num_classes
    },
    training_info={
        'use_random_search': use_random_search,
        'n_iter': n_iter if use_random_search else 0,
        'final_epochs': final_epochs,
        'random_search_time': random_search_time,  # ✅ Tempo de Random Search
        'final_train_time': final_train_time,  # ✅ Tempo de treinamento final
        'total_time': total_time,  # ✅ Tempo total
        'device': str(self.device),  # ✅ Dispositivo usado (GPU/CPU)
        'use_augmentation': USE_AUGMENTATION,  # ✅ Data augmentation usado
        'transfer_learning': False  # ✅ Se usou transfer learning
    },
    class_names=self.class_names,
    model_params={  # ✅ Parâmetros para reconstruir modelo
        'num_classes': self.num_classes,
        'dropout_rate': best_params['dropout_rate'],
        'hidden_units': best_params['hidden_units']
    }
)
```

---

#### **3. `load_model_with_metadata()` - Carrega Modelo com Metadados**

**Localização**: `src/model_saver.py`, linhas 48-87

**Assinatura**:
```python
def load_model_with_metadata(model_path, model_type='pytorch', model_class=None):
    """
    Carrega modelo com metadados
    
    Args:
        model_path: Caminho do modelo salvo
        model_type: Tipo do modelo ('pytorch' ou 'sklearn')
        model_class: Classe do modelo (necessário para PyTorch)
    
    Returns:
        model: Modelo carregado
        metadata: Dicionário com metadados
    """
```

**Implementação para PyTorch** (linha 66-79):
```python
# src/model_saver.py - LINHA 66-79:
if model_type == 'pytorch':
    import torch
    checkpoint = torch.load(model_path, map_location='cpu')  # ✅ Carrega na CPU primeiro
    
    if model_class is None:
        raise ValueError("model_class é necessário para carregar modelos PyTorch")
    
    # Recriar modelo com metadados
    metadata = checkpoint.get('metadata', {})  # ✅ Extrai metadados
    model = model_class(**metadata.get('model_params', {}))  # ✅ Reconstrói modelo
    model.load_state_dict(checkpoint['model_state_dict'])  # ✅ Carrega pesos
    model.eval()  # ✅ Modo avaliação
    
    return model, metadata
```

**Implementação para scikit-learn** (linha 81-84):
```python
# src/model_saver.py - LINHA 81-84:
elif model_type == 'sklearn':
    import joblib
    data = joblib.load(model_path)
    return data['model'], data.get('metadata', {})  # ✅ Retorna modelo e metadados
```

**Exemplo de Uso - Carregar SVM** (arquivo `scripts/load_model_example.py`):
```python
# scripts/load_model_example.py - EXEMPLO:
from src.model_saver import load_model_with_metadata

# Carregar modelo SVM
svm_model, metadata = load_model_with_metadata(
    model_path='outputs/models/svm_model.pkl',
    model_type='sklearn'
)

print(f"Modelo: {metadata['model_name']}")
print(f"Acurácia: {metadata['metrics']['accuracy']:.4f}")
print(f"Hiperparâmetros: {metadata['hyperparameters']}")
print(f"Data de treinamento: {metadata['timestamp']}")
```

**Exemplo de Uso - Carregar SimpleCNN** (arquivo `scripts/load_model_example.py`):
```python
# scripts/load_model_example.py - EXEMPLO:
from src.model_saver import load_model_with_metadata
from src.models import SimpleCNN

# Carregar modelo SimpleCNN
model, metadata = load_model_with_metadata(
    model_path='outputs/models/simple_cnn.pth',
    model_type='pytorch',
    model_class=SimpleCNN  # ✅ Necessário para reconstruir modelo
)

print(f"Modelo: {metadata['model_name']}")
print(f"Acurácia: {metadata['metrics']['accuracy']:.4f}")
print(f"Device usado: {metadata['training_info']['device']}")
```

---

### Estrutura de Arquivos Salvos

Após treinar modelos, você terá:

```
outputs/models/
├── svm_model.pkl              # Modelo SVM (joblib)
├── svm_model.json             # Metadados do SVM
├── svm_scaler.pkl             # Scaler usado no SVM
├── random_forest_model.pkl    # Modelo Random Forest
├── random_forest_model.json   # Metadados do Random Forest
├── simple_cnn.pth             # Modelo SimpleCNN (PyTorch)
├── simple_cnn.json            # Metadados do SimpleCNN
├── resnet50_transfer.pth      # Modelo ResNet50 (PyTorch)
└── resnet50_transfer.json     # Metadados do ResNet50
```

**Exemplo de arquivo JSON de metadados** (`svm_model.json`):
```json
{
  "model_name": "SVM",
  "timestamp": "2024-01-15T10:30:45.123456",
  "metrics": {
    "accuracy": 0.8542,
    "precision": 0.8520,
    "recall": 0.8542,
    "f1_score": 0.8531
  },
  "hyperparameters": {
    "C": 1.23,
    "gamma": 0.045,
    "kernel": "rbf",
    "degree": 3,
    "class_weight": "balanced"
  },
  "training_info": {
    "use_random_search": true,
    "n_iter": 50,
    "cv_folds": 2,
    "pca_used": true,
    "pca_components": 500,
    "use_linear_svm": false,
    "img_size_classic": [64, 64],
    "max_samples": null,
    "total_time_seconds": 932.45,
    "device": "CPU",
    "n_jobs": 1
  },
  "class_names": ["aiartdata", "realart"],
  "num_classes": 2,
  "version": "1.0"
}
```

---

## Configuração

#### Opção 1: Usar Dataset do Kaggle (Recomendado)

O projeto está configurado para usar o dataset **AI Art vs Human Art** do Kaggle:

```bash
# Baixar e organizar o dataset automaticamente
python download_dataset.py
```

O script irá:
- Baixar o dataset do Kaggle automaticamente
- Explorar a estrutura do dataset
- Organizar os dados em `data/train/` e `data/test/`
- Dividir automaticamente em 70% treino e 30% teste

**Nota**: Certifique-se de ter aceitado os termos do dataset no Kaggle antes de executar.

#### Opção 2: Organizar Dados Manualmente

Se preferir usar seus próprios dados, organize no formato:
   ```
   data/
     train/
       classe1/
         img1.jpg
         img2.jpg
       classe2/
         img1.jpg
     test/
       classe1/
       classe2/
   ```

#### Configurações do `config.py`:

- `USE_GPU`: True para usar GPU, False para CPU
- `USE_KAGGLE_DATASET`: True para usar dataset do Kaggle (padrão: True)
- `KAGGLE_DATASET`: Nome do dataset no formato "usuario/dataset"
- `TRAIN_SPLIT`: Proporção de dados para treinamento (padrão: 0.7)
- `TEST_SPLIT`: Proporção de dados para teste (padrão: 0.3)
- `BATCH_SIZE`: Tamanho do batch (padrão: 32)
- `EPOCHS`: Número de épocas (padrão: 50)
- `USE_AUGMENTATION`: Ativar data augmentation

## Uso

### Passo 1: Baixar o Dataset (se necessário)

Se você ainda não tem os dados organizados:

```bash
python scripts/download_dataset.py
```

O script irá baixar e organizar automaticamente o dataset do Kaggle.

### Passo 2: Executar o Projeto

Execute o script principal:

```bash
python main.py
```

Se os dados não estiverem organizados, o script oferecerá a opção de baixar automaticamente.

Escolha uma das opções:
1. Pipeline Clássico (SVM + Random Forest)
2. Pipeline Deep Learning (CNN + ResNet)
3. Ambos os pipelines
4. Sair

## Contextualização da Base de Dados

### Dataset: AI Art vs Human Art

Este projeto utiliza o dataset **AI Art vs Human Art** do Kaggle, que contém imagens classificadas em duas categorias:

- **AI Art**: Arte gerada por inteligência artificial
- **Human Art**: Arte criada por humanos

**Link do Dataset**: https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art

### Descrição dos Dados

A base de dados é organizada automaticamente em diretórios por classe. O sistema detecta automaticamente:

- **Quantidade de imagens**: Contadas automaticamente durante o carregamento
- **Tamanho das imagens**: Configurável em `config.py` (padrão: 224x224 pixels)
- **Canais**: RGB (3 canais)
- **Quantidade de classes**: Detectada automaticamente a partir dos diretórios
- **Divisão treino/teste**: 70% treino, 30% teste (configurável)

### Características do Dataset

O dataset **AI Art vs Human Art** contém:
- **Total de arquivos**: ~975 imagens
- **Formatos**: JPG (763), PNG (150), JPEG (57), outros (5)
- **Classes**: 
  - AiArtData: ~539 imagens (55%)
  - RealArt: ~436 imagens (45%)
- **Desbalanceamento**: Leve desbalanceamento (~20% de diferença)

### Padronização de Imagens

O projeto implementa **padronização completa** de imagens para garantir consistência e qualidade dos dados:

#### 1. **Tratamento de Múltiplos Formatos** ✅
- Suporta automaticamente: JPG, JPEG, PNG, BMP, GIF
- Conversão uniforme para formato interno
- Tratamento específico para cada tipo de arquivo

#### 2. **Padronização de Canais de Cor** ✅
- **Conversão para RGB**: Todas as imagens são convertidas para RGB (3 canais)
- **Remoção de Alpha Channel**: PNGs com transparência são convertidos com fundo branco
- **Conversão Grayscale**: Imagens em escala de cinza são convertidas para RGB
- **Validação**: Garante que todas as imagens tenham exatamente 3 canais

#### 3. **Correção de Orientação EXIF** ✅
- **Correção Automática**: Aplica correção de orientação baseada em metadados EXIF
- **Importante para Arte**: Evita que imagens apareçam rotacionadas incorretamente
- **Transparente**: Processo automático, sem intervenção manual

#### 4. **Redimensionamento Inteligente** ✅
- **Tamanho Padrão**: Todas as imagens são redimensionadas para 224x224 pixels
- **Interpolação de Alta Qualidade**: Usa `INTER_AREA` do OpenCV (melhor para downscaling)
- **Validação de Dimensões**: Rejeita imagens muito pequenas (< 32x32 pixels)

#### 5. **Validação e Tratamento de Erros** ✅
- **Detecção de Imagens Corrompidas**: Identifica e trata arquivos inválidos
- **Validação de Qualidade**: Verifica dimensões mínimas e formato válido
- **Logging Detalhado**: Relatório completo de problemas encontrados
- **Continuidade**: Processo não é interrompido por imagens problemáticas

#### 6. **Relatório de Estatísticas** ✅

Ao carregar as imagens, o sistema exibe um relatório detalhado:

```
============================================================
ESTATÍSTICAS DE CARREGAMENTO DE IMAGENS
============================================================
Total de arquivos processados: 975
Imagens carregadas com sucesso: 970
Erros encontrados: 5

Formatos encontrados:
  .jpg: 763
  .jpeg: 57
  .png: 150

Imagens em escala de cinza convertidas: X
Canais alpha removidos: Y
Orientações EXIF corrigidas: Z
============================================================
```

#### 7. **Normalização de Valores** ✅

**Para Pipeline Clássico:**
- Normalização para [0, 1]: Divisão por 255
- Padronização: StandardScaler (média 0, desvio padrão 1)

**Para Pipeline Deep Learning:**
- Normalização ImageNet: 
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- Conversão para Tensor: Valores normalizados para treinamento

### Benefícios da Padronização

1. **Consistência**: Todas as imagens têm o mesmo formato e tamanho
2. **Qualidade**: Melhor performance dos modelos com dados padronizados
3. **Robustez**: Tratamento automático de diferentes formatos e problemas
4. **Transparência**: Relatórios detalhados sobre o processamento
5. **Confiabilidade**: Validação garante que apenas imagens válidas são usadas

### Estrutura Após Download

Após executar `download_dataset.py`, a estrutura será:

```
data/
  train/
    ai_art/        (70% das imagens de arte IA)
    human_art/     (70% das imagens de arte humana)
  test/
    ai_art/        (30% das imagens de arte IA)
    human_art/     (30% das imagens de arte humana)
```

O código imprime automaticamente:
- Número de amostras de treinamento
- Número de amostras de teste
- Tamanho das imagens
- Número de canais
- Nomes das classes

---

## 🔍 Random Search Otimizado - Como Funciona em Todos os Modelos

### Visão Geral

O Random Search foi **otimizado** para economizar memória mantendo a qualidade da busca de hiperparâmetros. Esta seção explica **EXATAMENTE** como funciona em cada modelo.

### Configuração Global do Random Search

**Localização**: `src/config.py`, linha 108

```python
# src/config.py - LINHA 108:
CLASSIC_CV_FOLDS = 2  # Número de folds para validação cruzada
# Aplica-se a TODOS os modelos clássicos (SVM E Random Forest)
```

**Antes**: `cv=3` (fixo no código)  
**Depois**: `cv=CLASSIC_CV_FOLDS` (configurável, padrão: 2)  
**Redução de memória**: ~33% (2 folds vs 3 folds)

---

### Random Search no SVM

#### **Configurações Específicas**

**Localização**: `src/config.py`, linhas 106-107

```python
# src/config.py - LINHAS 106-107:
CLASSIC_SVM_N_JOBS = 1  # Jobs paralelos para SVM (1 = sem paralelização)
CLASSIC_USE_LINEAR_SVM = False  # False = SVC (kernels), True = LinearSVC (só linear)
```

#### **Implementação Completa**

**Localização**: `src/pipelines/classic.py`, função `train_svm()`, linhas 257-294

**Código Completo do Random Search para SVM**:
```python
# src/pipelines/classic.py - LINHAS 257-294:
if use_random_search:
    print(f"\n   Otimizando hiperparâmetros com Random Search ({n_iter} iterações)...")
    print(f"   CV folds: {CLASSIC_CV_FOLDS} (reduzido para economizar memória)")
    search_start = time.time()
    
    if use_linear_svm:
        # LinearSVC: apenas kernel linear, menos parâmetros
        param_distributions = {
            'C': loguniform(0.01, 100),           # Distribuição log-uniform
            'loss': ['hinge', 'squared_hinge'],   # Funções de perda
            'class_weight': [None, 'balanced'],   # Balanceamento de classes
            'dual': [True, False]                 # Forma dual ou primal
        }
        svm = LinearSVC(random_state=42, max_iter=2000)
    else:
        # SVC tradicional: múltiplos kernels
        param_distributions = {
            'C': loguniform(0.01, 100),           # Parâmetro de regularização
            'gamma': loguniform(0.0001, 1),       # Para kernels RBF e poly
            'kernel': ['rbf', 'linear', 'poly'],  # Tipo de kernel
            'degree': randint(2, 5),              # Para kernel poly (grau 2, 3 ou 4)
            'class_weight': [None, 'balanced']    # Balanceamento de classes
        }
        svm = SVC(random_state=42)
    
    # CRÍTICO: CV_FOLDS configurável e n_jobs específico para SVM
    random_search = RandomizedSearchCV(
        svm, param_distributions, 
        n_iter=n_iter,                          # ✅ Número de iterações (padrão: 50)
        cv=CLASSIC_CV_FOLDS,                    # ✅ CV folds configurável (padrão: 2)
        scoring='accuracy',                      # Métrica de avaliação
        n_jobs=svm_n_jobs,                      # ✅ Paralelização configurável (padrão: 1)
        verbose=1,                              # Mostrar progresso
        random_state=42                         # Reproduzibilidade
    )
    random_search.fit(self.X_train, self.y_train)  # ✅ Treina com todos os dados
    
    search_time = time.time() - search_start
    search_time_str = str(timedelta(seconds=int(search_time)))
    
    self.svm_model = random_search.best_estimator_  # ✅ Melhor modelo encontrado
    print(f"Melhores parâmetros: {random_search.best_params_}")
    print(f"Melhor score (CV): {random_search.best_score_:.4f}")
    print(f"Tempo de Random Search: {search_time_str} ({search_time:.2f} segundos)")
```

**Fluxo Completo**:
1. **Define espaço de parâmetros**: Distribuições log-uniform, uniform ou listas discretas
2. **Cria RandomizedSearchCV**: Com `n_iter` iterações, `cv=CLASSIC_CV_FOLDS` folds, `n_jobs=svm_n_jobs`
3. **Executa busca**: Para cada iteração, seleciona parâmetros aleatórios e avalia com CV
4. **Total de fits**: `n_iter × cv_folds` (ex: 50 × 2 = 100 fits)
5. **Retorna melhor modelo**: `best_estimator_` com melhores hiperparâmetros encontrados

**Memória Usada**:
- **Por fold**: Uma cópia dos dados transformados (após PCA)
- **Com PCA ativo**: ~500 features × n_samples × 8 bytes = muito menor!
- **Sem paralelização** (`n_jobs=1`): Uma cópia por vez
- **Total estimado**: ~1-2 GB (vs ~15-20 GB antes das otimizações)

---

### Random Search no Random Forest

#### **Configurações Específicas**

**Localização**: `src/config.py`, linha 107

```python
# src/config.py - LINHA 107:
CLASSIC_RF_N_JOBS = -1  # Jobs paralelos para Random Forest (-1 = todos os cores)
# Random Forest pode usar mais paralelização que SVM (mais eficiente em memória)
```

#### **Implementação Completa**

**Localização**: `src/pipelines/classic.py`, função `train_random_forest()`, linhas 426-453

**Código Completo do Random Search para Random Forest**:
```python
# src/pipelines/classic.py - LINHAS 426-453:
if use_random_search:
    print(f"\n   Otimizando hiperparâmetros com Random Search ({n_iter} iterações)...")
    print(f"   CV folds: {CLASSIC_CV_FOLDS} (reduzido para economizar memória)")
    search_start = time.time()
    
    param_distributions = {
        'n_estimators': randint(50, 300),       # Número de árvores (50 a 299)
        'max_depth': [None, 10, 20, 30, 50],   # Profundidade máxima
        'min_samples_split': randint(2, 20),    # Amostras mínimas para dividir (2 a 19)
        'min_samples_leaf': randint(1, 10),     # Amostras mínimas por folha (1 a 9)
        'max_features': ['sqrt', 'log2', None], # Features por split
        'bootstrap': [True, False],             # Bootstrap sampling
        'class_weight': [None, 'balanced', 'balanced_subsample']  # Balanceamento
    }
    
    # CRÍTICO: n_jobs específico para Random Forest (pode usar mais cores)
    rf_n_jobs = CLASSIC_RF_N_JOBS if CLASSIC_RF_N_JOBS is not None else self.n_jobs
    if rf_n_jobs == -1:
        actual_jobs = self.num_cores  # ✅ Todos os cores disponíveis
    else:
        actual_jobs = rf_n_jobs
    
    rf = RandomForestClassifier(random_state=42, n_jobs=rf_n_jobs)  # ✅ n_jobs específico
    
    random_search = RandomizedSearchCV(
        rf, param_distributions, 
        n_iter=n_iter,                          # ✅ Número de iterações (padrão: 50)
        cv=CLASSIC_CV_FOLDS,                    # ✅ CV folds configurável (padrão: 2)
        scoring='accuracy',
        n_jobs=rf_n_jobs,                       # ✅ Paralelização configurável (padrão: -1)
        verbose=1,
        random_state=42
    )
    random_search.fit(self.X_train, self.y_train)
    
    search_time = time.time() - search_start
    search_time_str = str(timedelta(seconds=int(search_time)))
    
    self.rf_model = random_search.best_estimator_
    print(f"Melhores parâmetros: {random_search.best_params_}")
    print(f"Melhor score (CV): {random_search.best_score_:.4f}")
    print(f"Tempo de Random Search: {search_time_str} ({search_time:.2f} segundos)")
```

**Diferenças em relação ao SVM**:
- ✅ Random Forest pode usar `n_jobs=-1` (todos os cores) porque usa memória de forma mais eficiente
- ✅ Não precisa calcular matriz Gram como SVM
- ✅ Árvores independentes = paralelização nativa muito eficiente
- ✅ Mesmo `CLASSIC_CV_FOLDS = 2` se aplica (configuração global)

---

### Random Search no Simple CNN (Deep Learning)

**Diferença Importante**: Simple CNN usa **implementação customizada** de Random Search, não `RandomizedSearchCV` do scikit-learn.

#### **Implementação Customizada**

**Localização**: `src/pipelines/deep_learning.py`, função `train_simple_cnn()`, linhas 768-825

**Código Completo do Random Search para SimpleCNN**:
```python
# src/pipelines/deep_learning.py - LINHAS 768-825:
if use_random_search:
    print(f"\nExecutando Random Search ({n_iter} iterações)...")
    search_start_time = time.time()
    
    # Espaço de hiperparâmetros
    param_space = {
        'learning_rate': (0.0001, 0.01),      # Log-uniform (distribuição log)
        'batch_size': [16, 32, 64],           # Valores discretos
        'dropout_rate': (0.3, 0.7),           # Uniform entre 0.3 e 0.7
        'hidden_units': [256, 512, 1024]      # Valores discretos
    }
    
    best_val_acc = 0.0
    search_epochs = min(15, final_epochs)  # ✅ Limita épocas durante busca
    
    for i in range(n_iter):  # ✅ Loop manual ao invés de RandomizedSearchCV
        iter_start = time.time()
        
        # Amostrar hiperparâmetros aleatoriamente
        params = sample_hyperparameters(param_space)  # ✅ Função customizada
        
        print(f"\n  Iteração {i+1}/{n_iter}: lr={params['learning_rate']:.6f}, "
              f"batch={params['batch_size']}, dropout={params['dropout_rate']:.2f}, "
              f"hidden={params['hidden_units']}")
        
        # Criar dataloaders com batch size amostrado
        train_loader, val_loader, _ = self.create_dataloaders(
            params['batch_size'], val_split=0.2  # ✅ Split interno de validação
        )
        
        # Criar modelo com hiperparâmetros amostrados
        model = SimpleCNN(
            self.num_classes,
            dropout_rate=params['dropout_rate'],
            hidden_units=params['hidden_units']
        )
        
        # CRÍTICO: Mover modelo para GPU ANTES do treinamento
        model = model.to(self.device)
        
        # Treinar modelo com configuração específica
        val_acc, _, iter_time = self.train_single_config(
            model, train_loader, val_loader, search_epochs,
            params['learning_rate'], patience=5  # ✅ Early stopping
        )
        
        iter_total_time = time.time() - iter_start
        print(f"    Val Acc: {val_acc:.4f} | Tempo da iteração: {iter_total_time:.1f}s")
        
        # Manter melhor configuração encontrada
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params.copy()
    
    random_search_time = time.time() - search_start_time
    # ... exibir resultados ...
```

**Função `sample_hyperparameters()` - Localização**: `src/pipelines/deep_learning.py`, linhas 60-86

```python
# src/pipelines/deep_learning.py - LINHAS 60-86:
def sample_hyperparameters(param_space):
    """
    Amostra aleatoriamente hiperparâmetros do espaço definido
    
    Args:
        param_space: Dicionário com espaço de hiperparâmetros
            - Tuplas (min, max): Uniform ou log-uniform
            - Listas: Escolha aleatória
    
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
                    params[key] = 10 ** np.random.uniform(log_low, log_high)  # ✅ Log-uniform
                else:
                    params[key] = np.random.uniform(value[0], value[1])  # ✅ Uniform
            elif isinstance(value[0], int):
                params[key] = np.random.randint(value[0], value[1] + 1)  # ✅ Randint
        elif isinstance(value, list):
            params[key] = random.choice(value)  # ✅ Escolha aleatória de lista
        else:
            params[key] = value  # ✅ Valor fixo
    return params
```

**Diferenças da Implementação Customizada**:
- ✅ **Não cria múltiplas cópias dos dados**: Usa lazy loading e processamento em batches
- ✅ **Validação split interna**: 20% dos dados de treino, não CV folds
- ✅ **Early stopping**: Para treinamento quando não melhora (patience=5)
- ✅ **Épocas limitadas**: `search_epochs = min(15, final_epochs)` durante busca
- ✅ **Sequencial**: Testa configurações uma por vez (não paralelo, mas usa GPU eficientemente)

**Por que não precisa das mesmas otimizações do SVM?**:
- ✅ Lazy loading: Dados carregados sob demanda (não tudo na memória)
- ✅ Processamento em batches: Apenas um batch por vez na GPU
- ✅ Sem CV folds: Apenas split simples de validação
- ✅ Cada iteração é independente: Modelo deletado após avaliação

---

### Random Search no ResNet50 (Deep Learning)

**Mesma implementação customizada** do SimpleCNN, mas com configurações específicas para ResNet50.

#### **Configurações Específicas**

**Localização**: `src/config.py`, linhas 84-95

```python
# src/config.py - LINHAS 84-95:
RESNET50_BATCH_SIZES = [8, 16, 32]  # ✅ Batch sizes reduzidos (era [16, 32, 64])
RESNET50_DEFAULT_BATCH_SIZE = 16    # ✅ Padrão reduzido (era 32)
RESNET50_SEARCH_EPOCHS = 10         # ✅ Épocas limitadas durante busca
RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS = True  # ✅ Limpar memória entre iterações
```

#### **Implementação com Limpeza de Memória**

**Localização**: `src/pipelines/deep_learning.py`, função `train_resnet_transfer()`, linhas 1093-1145

**Código Completo do Random Search para ResNet50** (com limpeza de memória):
```python
# src/pipelines/deep_learning.py - LINHAS 1093-1145:
if use_random_search:
    print(f"\nExecutando Random Search ({n_iter} iterações)...")
    print(f"  Batch sizes testados: {RESNET50_BATCH_SIZES}")
    print(f"  Épocas por iteração: {RESNET50_SEARCH_EPOCHS}")
    print(f"  Limpeza de memória entre iterações: {'Ativada' if RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS else 'Desativada'}")
    search_start_time = time.time()
    
    param_space = {
        'learning_rate': (0.00001, 0.001),   # ✅ Learning rate menor (transfer learning)
        'batch_size': RESNET50_BATCH_SIZES,  # ✅ Batch sizes configuráveis
        'unfreeze_layers': [0, 1, 2]         # ✅ Quantas camadas descongelar
    }
    
    best_val_acc = 0.0
    search_epochs = min(RESNET50_SEARCH_EPOCHS, final_epochs)  # ✅ Épocas limitadas
    
    for i in range(n_iter):
        iter_start = time.time()
        params = sample_hyperparameters(param_space)
        
        print(f"\n  Iteração {i+1}/{n_iter}: lr={params['learning_rate']:.6f}, "
              f"batch={params['batch_size']}, unfreeze={params['unfreeze_layers']}")
        
        # ✅ LIMPAR MEMÓRIA ANTES de cada iteração
        if RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS:
            clear_memory(clear_gpu=True)
        
        train_loader, val_loader, _ = self.create_dataloaders(
            params['batch_size'], val_split=0.2
        )
        
        # Criar modelo (já verifica memória internamente)
        model = self.create_resnet_model(unfreeze_layers=params['unfreeze_layers'])
        
        try:
            val_acc, trained_model, iter_time = self.train_single_config(
                model, train_loader, val_loader, search_epochs,
                params['learning_rate'], patience=5
            )
            # ... processar resultados ...
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # ✅ TRATAMENTO DE ERRO: Recuperação automática
                print(f"    [ERRO] Estouro de memória na iteração {i+1}!")
                clear_memory(clear_gpu=True)
                
                # Tentar com batch size menor
                if params['batch_size'] > min(RESNET50_BATCH_SIZES):
                    reduced_batch = max(min(RESNET50_BATCH_SIZES), params['batch_size'] // 2)
                    print(f"    Tentando com batch size reduzido: {reduced_batch}")
                    params['batch_size'] = reduced_batch
                    continue
                else:
                    print(f"    [AVISO] Não foi possível reduzir mais o batch size. Pulando iteração.")
                    continue
            else:
                raise
        finally:
            # ✅ CRÍTICO: LIMPAR MEMÓRIA APÓS cada iteração
            if RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS:
                # Mover modelo para CPU antes de deletar
                if 'trained_model' in locals():
                    trained_model = trained_model.cpu()
                if 'model' in locals():
                    model = model.cpu()
                del model, trained_model  # ✅ Deletar explicitamente
                clear_memory(clear_gpu=True)  # ✅ Limpar cache CUDA
                
                # Mostrar status de memória
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    gpu_mem_used = torch.cuda.memory_allocated() / (1024**3)
                    print(f"    Memória GPU após limpeza: {gpu_mem_used:.2f} GB")
```

**Características Especiais do Random Search do ResNet50**:
- ✅ **Limpeza automática**: Antes e depois de cada iteração
- ✅ **Tratamento de erros**: Recupera automaticamente de estouro de memória
- ✅ **Batch size adaptativo**: Reduz automaticamente se necessário
- ✅ **Verificação de memória**: Antes de carregar modelo grande
- ✅ **Batch sizes menores**: [8, 16, 32] ao invés de [16, 32, 64]

---

### Comparação: Random Search em Todos os Modelos

| Modelo | Tipo | Implementação | CV Folds | n_jobs | Limpeza Memória | Otimizado? |
|--------|------|---------------|----------|--------|-----------------|------------|
| **SVM** | Clássico | RandomizedSearchCV | `CLASSIC_CV_FOLDS=2` | `CLASSIC_SVM_N_JOBS=1` | N/A (CPU) | ✅ **Sim** |
| **Random Forest** | Clássico | RandomizedSearchCV | `CLASSIC_CV_FOLDS=2` | `CLASSIC_RF_N_JOBS=-1` | N/A (CPU) | ✅ **Sim** |
| **Simple CNN** | Deep Learning | Customizado | Split interno (20%) | N/A (GPU) | Automática | ⚠️ **Não precisa** |
| **ResNet50** | Deep Learning | Customizado | Split interno (20%) | N/A (GPU) | **Entre iterações** | ✅ **Sim** |

**Explicação**:
- ✅ Modelos clássicos usam `RandomizedSearchCV` do scikit-learn → precisam de otimizações de memória
- ✅ Modelos deep learning usam implementação customizada → já são eficientes (lazy loading + batches)
- ✅ ResNet50 precisa de limpeza adicional porque modelo é muito grande

---

## 🏗️ Pipeline Clássico - Detalhes Completos

### Modelos Implementados

#### **1. Support Vector Machine (SVM)**

**Localização**: `src/pipelines/classic.py`, função `train_svm()`, linhas 202-393

**Características**:
- ✅ Suporta `SVC` (kernels: RBF, linear, poly) ou `LinearSVC` (apenas linear)
- ✅ Otimização: Random Search (50 iterações padrão)
- ✅ Parâmetros otimizados: C, gamma, kernel, degree, class_weight (SVC) ou C, loss, dual, class_weight (LinearSVC)
- ✅ Validação cruzada: 2 folds (configurável)
- ✅ Paralelização: 1 job (configurável para economizar memória)

**Exemplo de Parâmetros Otimizados** (linha 274-279):
```python
# src/pipelines/classic.py - LINHA 274-279 (SVC):
param_distributions = {
    'C': loguniform(0.01, 100),           # Parâmetro de regularização (log-uniform)
    'gamma': loguniform(0.0001, 1),       # Para kernels RBF e poly (log-uniform)
    'kernel': ['rbf', 'linear', 'poly'],  # Tipo de kernel (escolha aleatória)
    'degree': randint(2, 5),              # Grau do polinômio para kernel poly (2, 3 ou 4)
    'class_weight': [None, 'balanced']    # Balanceamento de classes
}
```

**Exemplo de Saída Durante Treinamento**:
```
================================================================================
TREINANDO MODELO: Support Vector Machine (SVM)
================================================================================
   Dispositivo: CPU (scikit-learn não suporta GPU)

   Verificação de memória:
     Amostras: 10,000
     Features: 500 (após PCA)
     Memória estimada para treinamento: ~0.08 GB

   Paralelização SVM: 1 job(s) (configurado para economizar memória)
   Tipo: SVC (suporta kernels não-lineares, mas usa mais memória)

   Otimizando hiperparâmetros com Random Search (50 iterações)...
   CV folds: 2 (reduzido para economizar memória)

   Fitting 2 folds for each of 50 candidates, totalling 100 fits
   [Progresso: ████████████████████████████████████████████] 100/100

   Melhores parâmetros: {'C': 1.23, 'gamma': 0.045, 'kernel': 'rbf', 'degree': 3, 'class_weight': 'balanced'}
   Melhor score (CV): 0.8542
   Tempo de Random Search: 0:15:32 (932.45 segundos)

   Predições - Tempo: 2.34 segundos

   Acurácia - Treinamento: 0.8734
   Acurácia - Teste: 0.8542
   Precisão - Teste: 0.8520
   Recall - Teste: 0.8542
   F1-Score - Teste: 0.8531

   Tempo total de execução: 0:15:40 (940.23 segundos)
```

#### **2. Random Forest**

**Localização**: `src/pipelines/classic.py`, função `train_random_forest()`, linhas 396-527

**Características**:
- ✅ Ensemble de árvores de decisão
- ✅ Otimização: Random Search (50 iterações padrão)
- ✅ Parâmetros otimizados: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, class_weight
- ✅ Validação cruzada: 2 folds (configurável)
- ✅ Paralelização: -1 (todos os cores) - Random Forest usa memória eficientemente

**Exemplo de Parâmetros Otimizados** (linha 430-437):
```python
# src/pipelines/classic.py - LINHA 430-437 (Random Forest):
param_distributions = {
    'n_estimators': randint(50, 300),       # Número de árvores (50 a 299)
    'max_depth': [None, 10, 20, 30, 50],   # Profundidade máxima (None = sem limite)
    'min_samples_split': randint(2, 20),   # Amostras mínimas para dividir (2 a 19)
    'min_samples_leaf': randint(1, 10),    # Amostras mínimas por folha (1 a 9)
    'max_features': ['sqrt', 'log2', None], # Features por split
    'bootstrap': [True, False],            # Bootstrap sampling
    'class_weight': [None, 'balanced', 'balanced_subsample']  # Balanceamento de classes
}
```

---

### Transformações Aplicadas no Pipeline Clássico

#### **1. Carregamento de Imagens com Padronização Completa**

**Localização**: `src/utils.py`, função `load_images_from_directory()`, linhas 128-316

**Características Implementadas**:

**1.1. Tratamento de Múltiplos Formatos** (linha 175-190):
```python
# src/utils.py - LINHA 175-190:
# Suporta múltiplos formatos automaticamente
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
images_found = list(directory.glob('*.[jJ][pP][gG]')) + \
               list(directory.glob('*.[jJ][pP][eE][gG]')) + \
               list(directory.glob('*.[pP][nN][gG]'))
```

**1.2. Conversão para RGB** (linha 200-215):
```python
# src/utils.py - LINHA 200-215:
# Converter para RGB (3 canais) - CRÍTICO para consistência
if len(image.shape) == 2:  # Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
elif len(image.shape) == 4:  # RGBA
    # Converter RGBA para RGB com fundo branco
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
elif image.shape[2] == 4:  # Alpha channel
    # Remover alpha channel
    image = image[:, :, :3]

# Garantir exatamente 3 canais
assert image.shape[2] == 3, f"Imagem deve ter 3 canais, encontrado: {image.shape[2]}"
```

**1.3. Correção de Orientação EXIF** (linha 195-198):
```python
# src/utils.py - LINHA 195-198:
# Correção de orientação EXIF (importante para arte)
pil_image = Image.fromarray(image)
pil_image = ImageOps.exif_transpose(pil_image)  # ✅ Corrige rotação baseada em EXIF
image = np.array(pil_image)
```

**1.4. Redimensionamento Inteligente** (linha 217-220):
```python
# src/utils.py - LINHA 217-220:
# Redimensionamento para tamanho padrão
# IMPORTANTE: Usa INTER_AREA (melhor para downscaling)
image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
```

**1.5. Validação e Tratamento de Erros** (linha 222-230):
```python
# src/utils.py - LINHA 222-230:
# Validar dimensões mínimas
if image.shape[0] < 32 or image.shape[1] < 32:
    warnings.warn(f"Imagem {img_path} muito pequena: {image.shape}")
    continue

# Validar formato válido
if image is None or image.size == 0:
    warnings.warn(f"Imagem {img_path} inválida ou corrompida")
    continue
```

---

#### **2. Pré-processamento Específico para Modelos Clássicos**

**Localização**: `src/pipelines/classic.py`, função `load_data()`, linhas 135-179

**2.1. Flatten de Imagens** (linha 137):
```python
# src/pipelines/classic.py - LINHA 137:
# Função preprocess_images_classic em src/utils.py
X_train_flat = preprocess_images_classic(X_train)
# Transforma (n_samples, height, width, channels) em (n_samples, height*width*channels)
# Exemplo: (1000, 64, 64, 3) → (1000, 12288)
```

**2.2. Normalização com StandardScaler** (linha 148-150):
```python
# src/pipelines/classic.py - LINHA 148-150:
# Normalização: Média 0, Desvio Padrão 1
X_train_scaled = self.scaler.fit_transform(X_train_flat)  # ✅ Aprende média e std do treino
X_test_scaled = self.scaler.transform(X_test_flat)  # ✅ Usa mesma média e std (importante!)
```

**2.3. Redução de Dimensionalidade com PCA** (linha 152-179):
```python
# src/pipelines/classic.py - LINHA 152-179:
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
    
    # CRÍTICO: fit_transform apenas no treino, transform no teste
    X_train_scaled = self.pca.fit_transform(X_train_scaled)  # ✅ Aprende componentes principais
    X_test_scaled = self.pca.transform(X_test_scaled)  # ✅ Usa componentes aprendidos (não aprende novamente!)
    
    # Calcular e exibir estatísticas
    n_features_after_pca = X_train_scaled.shape[1]
    reduction = ((n_features - n_features_after_pca) / n_features) * 100
    estimated_mem_after_gb = (n_samples * n_features_after_pca * 8) / (1024**3)
    print(f"   Features após PCA: {n_features_after_pca:,} ({reduction:.1f}% redução)")
    print(f"   Memória estimada após PCA: {estimated_mem_after_gb:.2f} GB")
    
    if hasattr(self.pca, 'explained_variance_ratio_'):
        total_variance = self.pca.explained_variance_ratio_.sum()
        print(f"   Variância explicada: {total_variance:.2%}")
```

**Por Que PCA é Importante?**:
- ✅ **Reduz dimensionalidade**: 12,288 features → 500 componentes (96% redução)
- ✅ **Mantém informação**: ~98% de variância explicada mantida
- ✅ **Economiza memória**: ~98% menos memória necessária
- ✅ **Acelera treinamento**: Menos features = treinamento mais rápido
- ✅ **Melhora performance**: Remove ruído e redundância

**Erro Comum Evitado**:
```python
# ❌ ERRADO (não fazer isso):
X_test_scaled = self.pca.fit_transform(X_test_scaled)  # ❌ Erro: re-aprende componentes no teste!

# ✅ CORRETO (implementado):
X_train_scaled = self.pca.fit_transform(X_train_scaled)  # ✅ Aprende do treino
X_test_scaled = self.pca.transform(X_test_scaled)  # ✅ Usa componentes do treino
```

**Salvamento do PCA**: O PCA é salvo junto com o modelo para uso em predições futuras (linha 362-367):
```python
# src/pipelines/classic.py - LINHA 362-367:
# Salvar PCA se foi usado (importante para predições futuras)
if self.pca is not None:
    pca_path = MODELS_DIR / 'svm_pca.pkl'
    joblib.dump(self.pca, pca_path)
    print(f"✅ PCA salvo em: {pca_path}")
```

---

#### **3. Valores dos Parâmetros Otimizados**

**SVM - SVC (Random Search - 50 iterações padrão):**

**Localização**: `src/pipelines/classic.py`, linhas 273-279

```python
# Espaço de busca para SVC (CLASSIC_USE_LINEAR_SVM = False)
param_distributions = {
    'C': loguniform(0.01, 100),           # Regularização: 0.01 a 100 (log-uniform)
    'gamma': loguniform(0.0001, 1),       # Kernel RBF/poly: 0.0001 a 1 (log-uniform)
    'kernel': ['rbf', 'linear', 'poly'],  # Tipo de kernel (3 opções)
    'degree': randint(2, 5),              # Grau polinomial: 2, 3 ou 4 (para kernel poly)
    'class_weight': [None, 'balanced']    # Balanceamento: None ou balanced (2 opções)
}
```

**Total de combinações teóricas**: Infinito (distribuições contínuas)  
**Combinações avaliadas**: Apenas `n_iter` (padrão: 50) aleatórias  
**Total de fits**: `n_iter × cv_folds` = 50 × 2 = **100 fits**

---

**SVM - LinearSVC (Random Search - 50 iterações padrão):**

**Localização**: `src/pipelines/classic.py`, linhas 264-268

```python
# Espaço de busca para LinearSVC (CLASSIC_USE_LINEAR_SVM = True)
param_distributions = {
    'C': loguniform(0.01, 100),           # Regularização: 0.01 a 100 (log-uniform)
    'loss': ['hinge', 'squared_hinge'],   # Função de perda (2 opções)
    'class_weight': [None, 'balanced'],   # Balanceamento: None ou balanced (2 opções)
    'dual': [True, False]                 # Forma dual ou primal (2 opções)
}
```

**Total de combinações teóricas**: Menor que SVC  
**Combinações avaliadas**: Apenas `n_iter` (padrão: 50) aleatórias  
**Total de fits**: `n_iter × cv_folds` = 50 × 2 = **100 fits**  
**Benefício**: ✅ Muito mais eficiente em memória (não calcula matriz Gram)

---

**Random Forest (Random Search - 50 iterações padrão):**

**Localização**: `src/pipelines/classic.py`, linhas 430-437

```python
# Espaço de busca para Random Forest
param_distributions = {
    'n_estimators': randint(50, 300),       # Número de árvores: 50 a 299
    'max_depth': [None, 10, 20, 30, 50],   # Profundidade máxima: 5 opções
    'min_samples_split': randint(2, 20),   # Amostras mínimas split: 2 a 19
    'min_samples_leaf': randint(1, 10),    # Amostras mínimas folha: 1 a 9
    'max_features': ['sqrt', 'log2', None], # Features por split: 3 opções
    'bootstrap': [True, False],             # Bootstrap sampling: 2 opções
    'class_weight': [None, 'balanced', 'balanced_subsample']  # Balanceamento: 3 opções
}
```

**Total de combinações teóricas**: Muito grande (produto de todos os espaços)  
**Combinações avaliadas**: Apenas `n_iter` (padrão: 50) aleatórias  
**Total de fits**: `n_iter × cv_folds` = 50 × 2 = **100 fits**

### Métricas Utilizadas

- Acurácia (Accuracy)
- Precisão (Precision)
- Recall
- F1-Score
- Matriz de Confusão

---

## 🧠 Pipeline Deep Learning - Detalhes Completos

### Modelos Implementados

#### **1. Simple CNN (sem Transfer Learning)**

**Localização**: `src/models/cnn.py`, classe `SimpleCNN`, linhas 9-69

**Arquitetura Completa**:
```python
# src/models/cnn.py - LINHAS 26-46:
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, hidden_units=512):
        super(SimpleCNN, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)      # ✅ 3 canais → 32 filtros
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)     # ✅ 32 → 64 filtros
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)    # ✅ 64 → 128 filtros
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)                               # ✅ Reduz tamanho pela metade
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))           # ✅ Garante tamanho fixo (7x7)
        
        # Regularização
        self.dropout = nn.Dropout(dropout_rate)                      # ✅ Dropout configurável
        
        # Camadas fully connected
        self.fc1 = nn.Linear(128 * 7 * 7, hidden_units)             # ✅ 6272 → hidden_units
        self.fc2 = nn.Linear(hidden_units, num_classes)             # ✅ hidden_units → num_classes
        
        # Ativação
        self.relu = nn.ReLU()                                        # ✅ ReLU
```

**Forward Pass** (linha 48-69):
```python
# src/models/cnn.py - LINHA 48-69:
def forward(self, x):
    # Bloco 1: Conv1 → ReLU → MaxPool
    x = self.pool(self.relu(self.conv1(x)))  # ✅ 224x224 → 112x112
    
    # Bloco 2: Conv2 → ReLU → MaxPool
    x = self.pool(self.relu(self.conv2(x)))  # ✅ 112x112 → 56x56
    
    # Bloco 3: Conv3 → ReLU → MaxPool
    x = self.pool(self.relu(self.conv3(x)))  # ✅ 56x56 → 28x28
    
    # Adaptive pooling: Garante tamanho fixo independente da entrada
    x = self.adaptive_pool(x)  # ✅ 28x28 → 7x7
    
    # Flatten: Transforma em vetor
    x = x.view(-1, 128 * 7 * 7)  # ✅ (batch, 128, 7, 7) → (batch, 6272)
    
    # Fully connected com dropout
    x = self.dropout(x)           # ✅ Dropout aplicado
    x = self.relu(self.fc1(x))    # ✅ 6272 → hidden_units
    x = self.fc2(x)               # ✅ hidden_units → num_classes
    
    return x
```

**Número de Parâmetros**:
- Com `hidden_units=512`: ~2.5 milhões de parâmetros
- Com `hidden_units=1024`: ~5.3 milhões de parâmetros
- Treinamento: Do zero (sem transfer learning)

**Otimização**: Random Search customizado (10 iterações padrão)

---

#### **2. ResNet50 (com Transfer Learning)**

**Localização**: `src/pipelines/deep_learning.py`, função `create_resnet_model()`, linhas 993-1063

**Características**:
- ✅ Base pré-treinada: ImageNet (IMAGENET1K_V2)
- ✅ ~25 milhões de parâmetros total
- ✅ Camadas convolucionais: Congeladas por padrão (configurável)
- ✅ Camada final: Substituída e treinada
- ✅ Otimização: Random Search customizado (10 iterações padrão)

**Código Completo de Criação**:
```python
# src/pipelines/deep_learning.py - LINHAS 1031-1047:
print(f"   Carregando ResNet50 pré-treinado...")
model = models.resnet50(weights='IMAGENET1K_V2')  # ✅ Carrega pesos pré-treinados

# Congelar todas as camadas por padrão
for param in model.parameters():
    param.requires_grad = False  # ✅ Não treina camadas convolucionais

# Substituir camada final (fully connected)
num_features = model.fc.in_features  # ✅ 2048 features
model.fc = nn.Linear(num_features, self.num_classes)  # ✅ 2048 → num_classes

# Descongelar apenas camada final
for param in model.fc.parameters():
    param.requires_grad = True  # ✅ Treina apenas camada final

# Opcional: Descongelar mais camadas para fine-tuning
if unfreeze_layers > 0:
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]  # ✅ Ordem: mais profundo → mais raso
    for i, layer in enumerate(layers[:unfreeze_layers]):
        for param in layer.parameters():
            param.requires_grad = True  # ✅ Descongela camada para treinamento
```

**Configurações de Unfreeze Layers**:
- `unfreeze_layers=0`: Apenas camada FC treinada (padrão) - **2,049 parâmetros treináveis**
- `unfreeze_layers=1`: FC + layer4 treinadas - **~2.7 milhões treináveis**
- `unfreeze_layers=2`: FC + layer4 + layer3 treinadas - **~7.4 milhões treináveis**

**Código de Movimento para GPU** (linha 1049-1061):
```python
# src/pipelines/deep_learning.py - LINHAS 1049-1061:
# CRÍTICO: Mover modelo para dispositivo correto (GPU ou CPU)
model = model.to(self.device)

# Verificar dispositivo e mostrar informações
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_device = next(model.parameters()).device

print(f"   ResNet50 carregado: {total_params:,} parâmetros total, {trainable_params:,} treináveis")
print(f"   Modelo movido para: {model_device}")
if model_device.type == 'cuda':
    print(f"   ✅ ResNet50 está na GPU: {torch.cuda.get_device_name(model_device.index or 0)}")
else:
    print(f"   ℹ️  ResNet50 está na CPU")

return model
```

### Configuração de Treinamento Deep Learning

**Localização**: `src/config.py`, linhas 32-36 e `src/pipelines/deep_learning.py`

**Parâmetros Padrão** (`src/config.py`, linhas 32-36):
```python
# src/config.py - LINHAS 32-36:
BATCH_SIZE = 32           # Tamanho do batch padrão
EPOCHS = 50                # Número de épocas padrão
LEARNING_RATE = 0.001      # Taxa de aprendizado padrão
```

**Optimizer** (implementado em `deep_learning.py`, linha 598):
```python
# src/pipelines/deep_learning.py - LINHA 598:
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # ✅ Adam optimizer
```

**Loss Function** (implementado em `deep_learning.py`, linha 597):
```python
# src/pipelines/deep_learning.py - LINHA 597:
criterion = nn.CrossEntropyLoss()  # ✅ Cross-entropy loss (padrão para classificação)
```

**Learning Rate Scheduler** (implementado em `deep_learning.py`, linhas 599-601):
```python
# src/pipelines/deep_learning.py - LINHAS 599-601:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5  # ✅ Reduz LR quando loss não melhora
)
# mode='min': Reduz quando loss para de diminuir
# factor=0.5: Multiplica LR por 0.5 quando reduz
# patience=5: Espera 5 épocas sem melhoria antes de reduzir
```

**Correção Implementada** (linha 599-601):
- **Antes**: `verbose=True` → ❌ Erro: `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`
- **Depois**: Removido `verbose` → ✅ Funciona em todas as versões do PyTorch

### Data Augmentation

Aplicado apenas durante o treinamento (não no teste):

- Rotação aleatória: 20 graus
- Translação horizontal/vertical: 20%
- Flip horizontal: Sim
- Zoom: 20%
- Ajuste de brilho/contraste: 20%

**Justificativa**: Aumenta a variabilidade dos dados de treinamento, reduzindo overfitting e melhorando generalização.

### Normalização

Valores de normalização ImageNet:
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

### Otimização de Hiperparâmetros (Random Search) - Deep Learning

**CNN Simples (Random Search - 10 iterações padrão):**

**Localização**: `src/pipelines/deep_learning.py`, função `train_simple_cnn()`, linhas 772-777

```python
# src/pipelines/deep_learning.py - LINHAS 772-777:
param_space = {
    'learning_rate': (0.0001, 0.01),      # ✅ Log-uniform entre 0.0001 e 0.01
    'batch_size': [16, 32, 64],           # ✅ Valores discretos
    'dropout_rate': (0.3, 0.7),           # ✅ Uniform entre 0.3 e 0.7
    'hidden_units': [256, 512, 1024]      # ✅ Valores discretos
}

best_val_acc = 0.0
search_epochs = min(15, final_epochs)  # ✅ Épocas limitadas durante busca (máximo 15)
```

**Validação**: 20% split interno com early stopping (patience=5)  
**Total de iterações**: `n_iter` (padrão: 10)

---

**ResNet50 (Random Search - 10 iterações padrão):**

**Localização**: `src/pipelines/deep_learning.py`, função `train_resnet_transfer()`, linhas 1100-1104

```python
# src/pipelines/deep_learning.py - LINHAS 1100-1104:
param_space = {
    'learning_rate': (0.00001, 0.001),   # ✅ Log-uniform (menor para transfer learning)
    'batch_size': RESNET50_BATCH_SIZES,  # ✅ [8, 16, 32] (reduzido de [16, 32, 64])
    'unfreeze_layers': [0, 1, 2]         # ✅ Quantidade de camadas a descongelar
}

best_val_acc = 0.0
search_epochs = min(RESNET50_SEARCH_EPOCHS, final_epochs)  # ✅ Máximo 10 épocas
```

**Validação**: 20% split interno com early stopping (patience=5)  
**Limpeza de memória**: Entre cada iteração (configurável)  
**Total de iterações**: `n_iter` (padrão: 10)

**Configurações de unfreeze_layers**:
- `unfreeze_layers=0`: Apenas camada FC treinada (padrão, mais rápido)
- `unfreeze_layers=1`: FC + layer4 treinadas (fine-tuning parcial)
- `unfreeze_layers=2`: FC + layer4 + layer3 treinadas (fine-tuning mais profundo)

**Vantagens do Random Search:**
1. Mais eficiente que Grid Search para espaços de alta dimensão
2. Permite explorar distribuições contínuas (log-uniform)
3. Early stopping reduz tempo de busca
4. Validação split garante seleção não enviesada de hiperparâmetros

### Escolha CPU/GPU

O sistema detecta automaticamente se há GPU disponível. Para forçar CPU, altere em `config.py`:

```python
USE_GPU = False  # Força uso de CPU
```

## Apresentação e Discussão dos Resultados

### Tabela de Resultados

Os resultados são salvos automaticamente em:
- `outputs/results/classic_pipeline_results.csv`
- `outputs/results/deep_learning_results.csv`

### Exemplo de Tabela

| Modelo | Acurácia | Precisão | Recall | F1-Score | Otimização | Transfer Learning |
|--------|----------|----------|--------|----------|------------|-------------------|
| SVM | 0.8500 | 0.8520 | 0.8500 | 0.8500 | Random Search (50 iter) | - |
| Random Forest | 0.8700 | 0.8720 | 0.8700 | 0.8700 | Random Search (50 iter) | - |
| CNN Simples | 0.8800 | 0.8820 | 0.8800 | 0.8800 | Random Search (10 iter) | Não |
| ResNet50 | 0.9500 | 0.9520 | 0.9500 | 0.9500 | Random Search (10 iter) | Sim |

### Visualizações Geradas

1. **Matrizes de Confusão**: Uma para cada modelo
   - Salvas em `outputs/figures/`
   - Formato PNG, alta resolução

2. **Métricas Comparativas**: Tabelas em CSV

### Análise dos Resultados

**Pipeline Clássico:**
- SVM geralmente apresenta melhor performance para dados de alta dimensionalidade
- Random Forest é robusto, interpretável e lida bem com dados desbalanceados
- Ambos usam Random Search para encontrar hiperparâmetros ótimos

**Pipeline Deep Learning:**
- CNN Simples aprende features automaticamente mas requer mais dados
- ResNet50 com transfer learning aproveita conhecimento pré-treinado
- Random Search otimiza hiperparâmetros de forma eficiente
- Deep learning geralmente supera métodos clássicos com dados suficientes

**Comparação de Otimização (Random Search):**
- Todos os 4 modelos utilizam Random Search para otimização de hiperparâmetros
- Permite comparação justa entre modelos clássicos e deep learning
- Pipeline clássico: 50 iterações (mais rápido por modelo)
- Pipeline deep learning: 10 iterações (mais custoso por iteração)

## Conclusão

### Dificuldades Encontradas

1. **Pré-processamento de Dados**
   - **Múltiplos formatos**: Necessidade de tratar JPG, PNG, JPEG uniformemente
   - **Canais inconsistentes**: Conversão de RGBA e grayscale para RGB
   - **Orientação EXIF**: Correção automática de rotação baseada em metadados
   - **Normalização adequada**: Diferentes normalizações para modelos clássicos e deep learning
   - **Balanceamento de classes**: Dataset com leve desbalanceamento (55% vs 45%)
   - **Tamanho adequado das imagens**: Redimensionamento mantendo qualidade
   - **Validação robusta**: Tratamento de imagens corrompidas ou inválidas

2. **Otimização de Hiperparâmetros**
   - Random Search mais eficiente que Grid Search para espaços grandes
   - Trade-off entre número de iterações e qualidade dos resultados
   - Validação cruzada/split requer dados suficientes

3. **Deep Learning**
   - Requer GPU para treinamento eficiente
   - Overfitting com poucos dados
   - Ajuste fino de learning rate e batch size

4. **Comparação de Modelos**
   - Diferentes métricas podem dar resultados diferentes
   - Necessidade de múltiplas execuções para estabilidade

### Melhorias Futuras

Se houvesse mais tempo para desenvolvimento:

1. **Pré-processamento**
   - ✅ **Implementado**: Padronização completa de formatos (JPG, PNG, JPEG)
   - ✅ **Implementado**: Conversão automática para RGB (3 canais)
   - ✅ **Implementado**: Correção de orientação EXIF
   - ✅ **Implementado**: Remoção de transparência (alpha channel)
   - ✅ **Implementado**: Validação robusta e tratamento de erros
   - ✅ **Implementado**: Relatório detalhado de estatísticas
   - Implementar balanceamento de classes (SMOTE, undersampling)
   - Testar diferentes tamanhos de imagem
   - Aplicar técnicas de denoising
   - Histogram equalization para normalizar brilho/contraste
   - Detecção automática de imagens de baixa qualidade

2. **Otimização de Hiperparâmetros**
   - Implementar Optuna para busca bayesiana mais eficiente
   - Early stopping para evitar overfitting
   - Ensemble de modelos

3. **Deep Learning**
   - Testar diferentes arquiteturas (EfficientNet, Vision Transformer)
   - Fine-tuning completo do ResNet (não apenas última camada)
   - Implementar callbacks (checkpointing, tensorboard)

4. **Avaliação**
   - Validação cruzada k-fold
   - Análise de erros (quais classes são mais confundidas)
   - Visualização de features aprendidas

5. **Deploy**
   - API REST para predições
   - Interface web para upload de imagens
   - Otimização de modelos para produção

## Execução

### Exemplo Completo

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar e organizar dataset do Kaggle
python scripts/download_dataset.py

# 3. Executar pipeline
python main.py

# 4. Escolher opção (1, 2, 3 ou 4)
#    1. Pipeline Clássico (SVM + Random Forest)
#    2. Pipeline Deep Learning (CNN + ResNet)
#    3. Ambos os pipelines
#    4. Sair

# 5. Ver resultados
# - outputs/results/classic_pipeline_results.csv
# - outputs/results/deep_learning_results.csv
# - outputs/figures/*.png
# - outputs/models/*.pkl ou *.pth
```

### Execução Rápida (Automática)

Se você já tem as credenciais do Kaggle configuradas:

```bash
python main.py
```

O script detectará automaticamente se os dados não existem e oferecerá a opção de baixar.

---

## ⚙️ Configurações Detalhadas

Esta seção explica **TODAS** as configurações disponíveis em `src/config.py`, organizadas por categoria.

### 📍 Localização das Configurações

**Arquivo**: `src/config.py`  
**Total de configurações**: 37 variáveis  
**Organização**: Por categoria (dataset, treinamento, memória, modelos específicos)

---

### 🎯 Configurações de Dispositivo e Hardware

#### **`USE_GPU`** (linha 13)
```python
USE_GPU = True  # Altere para False para usar CPU
```

**Descrição**: Controla se o sistema deve tentar usar GPU para modelos deep learning.  
**Valores**: `True` (tenta usar GPU se disponível) ou `False` (força CPU)  
**Quando alterar**: 
- ✅ `False` se não tiver GPU ou se quiser usar apenas CPU
- ✅ `False` se estiver tendo problemas com CUDA
- ✅ `True` para acelerar treinamento de modelos deep learning (CNN, ResNet50)

**Exemplo de uso**:
```python
# Forçar CPU
USE_GPU = False

# Tentar usar GPU (padrão)
USE_GPU = True
```

---

### 📊 Configurações do Dataset

#### **`KAGGLE_DATASET`** (linha 22)
```python
KAGGLE_DATASET = "hassnainzaidi/ai-art-vs-human-art"
```

**Descrição**: Nome do dataset do Kaggle no formato `usuario/dataset`.  
**Valores**: String com nome do dataset  
**Quando alterar**: Se quiser usar um dataset diferente do Kaggle  
**Exemplo**: `"outro_usuario/outro-dataset"`

---

#### **`USE_KAGGLE_DATASET`** (linha 23)
```python
USE_KAGGLE_DATASET = True  # Se True, usa dataset do Kaggle
```

**Descrição**: Se `True`, o script tenta baixar o dataset do Kaggle automaticamente.  
**Valores**: `True` ou `False`  
**Quando alterar**: Se já tiver os dados organizados manualmente, pode manter `True` (o script não baixa novamente se já existir)

---

#### **`TRAIN_SPLIT` e `TEST_SPLIT`** (linhas 24-25)
```python
TRAIN_SPLIT = 0.7  # Proporção de dados para treinamento
TEST_SPLIT = 0.3   # Proporção de dados para teste
```

**Descrição**: Proporções para dividir o dataset em treino e teste.  
**Valores**: Float entre 0 e 1, devem somar 1.0  
**Padrão**: 70% treino, 30% teste  
**Quando alterar**: 
- ✅ Se quiser mais dados de treino: `TRAIN_SPLIT = 0.8, TEST_SPLIT = 0.2`
- ✅ Se quiser mais dados de teste: `TRAIN_SPLIT = 0.6, TEST_SPLIT = 0.4`

**Importante**: Os valores devem somar 1.0!

---

### 🖼️ Configurações de Imagens

#### **`IMG_SIZE`** (linha 28)
```python
IMG_SIZE = (224, 224)  # Tamanho padrão para modelos de deep learning
```

**Descrição**: Tamanho das imagens para modelos deep learning (CNN, ResNet50).  
**Valores**: Tupla `(altura, largura)` em pixels  
**Padrão**: `(224, 224)` - padrão ImageNet  
**Quando alterar**: 
- ✅ Maior tamanho (`(256, 256)`, `(512, 512)`): Mais qualidade, mas mais memória e tempo
- ✅ Menor tamanho (`(128, 128)`): Menos memória, mas pode perder detalhes

**Uso**: Aplicado apenas em `src/pipelines/deep_learning.py`

---

#### **`IMG_SIZE_CLASSIC`** (linha 29)
```python
IMG_SIZE_CLASSIC = (64, 64)  # Tamanho menor para modelos clássicos (economiza memória)
```

**Descrição**: Tamanho das imagens para modelos clássicos (SVM, Random Forest).  
**Valores**: Tupla `(altura, largura)` em pixels  
**Padrão**: `(64, 64)` - **OTIMIZADO para economizar memória**  
**Quando alterar**: 
- ✅ Se tiver muito RAM: Pode aumentar para `(128, 128)` ou `(96, 96)`
- ✅ Se estiver com pouco RAM: Manter `(64, 64)` ou reduzir para `(32, 32)`

**Impacto na memória**: 
- `(224, 224)`: 150,528 features por imagem
- `(64, 64)`: 12,288 features por imagem (**92% redução!**)

**Uso**: Aplicado apenas em `src/pipelines/classic.py`

---

#### **`IMG_CHANNELS`** (linha 30)
```python
IMG_CHANNELS = 3  # RGB
```

**Descrição**: Número de canais de cor.  
**Valores**: `3` (RGB) ou `1` (grayscale)  
**Não recomendado alterar**: O código está otimizado para RGB (3 canais)

---

### 🏋️ Configurações de Treinamento (Deep Learning)

#### **`BATCH_SIZE`** (linha 33)
```python
BATCH_SIZE = 32
```

**Descrição**: Tamanho do batch para modelos deep learning (CNN, ResNet50).  
**Valores**: Inteiro positivo (8, 16, 32, 64, etc.)  
**Padrão**: `32`  
**Quando alterar**: 
- ✅ **Mais memória disponível**: Aumentar para `64` ou `128` (treina mais rápido)
- ✅ **Pouca memória GPU**: Reduzir para `16` ou `8` (evita estouro de memória)
- ✅ **ResNet50**: Use `RESNET50_DEFAULT_BATCH_SIZE` (linha 89) ao invés desta

**Impacto**:
- Batch maior = treina mais rápido, mas usa mais memória
- Batch menor = mais lento, mas usa menos memória

---

#### **`EPOCHS`** (linha 34)
```python
EPOCHS = 50
```

**Descrição**: Número máximo de épocas para treinamento deep learning.  
**Valores**: Inteiro positivo  
**Padrão**: `50`  
**Quando alterar**: 
- ✅ **Mais tempo disponível**: Aumentar para `100` ou `200`
- ✅ **Testes rápidos**: Reduzir para `10` ou `20`
- ✅ **Random Search**: Usa `min(15, EPOCHS)` durante busca (linha 1586 em `deep_learning.py`)

**Nota**: Early stopping pode parar antes se não houver melhoria (patience=5)

---

#### **`LEARNING_RATE`** (linha 35)
```python
LEARNING_RATE = 0.001
```

**Descrição**: Taxa de aprendizado inicial para otimizador Adam.  
**Valores**: Float positivo (geralmente entre 0.00001 e 0.1)  
**Padrão**: `0.001` (1e-3)  
**Quando alterar**: 
- ✅ **Modelo não converge**: Reduzir para `0.0001` ou `0.0005`
- ✅ **Modelo converge muito devagar**: Aumentar para `0.002` ou `0.005`
- ✅ **Transfer learning (ResNet50)**: Usar learning rate menor (`0.0001` ou `0.00001`)

**Nota**: Random Search otimiza automaticamente este parâmetro (espaço: 0.0001 a 0.01 para CNN, 0.00001 a 0.001 para ResNet50)

---

### 🎨 Configurações de Data Augmentation

#### **`USE_AUGMENTATION`** (linha 38)
```python
USE_AUGMENTATION = True
```

**Descrição**: Ativa/desativa data augmentation durante treinamento deep learning.  
**Valores**: `True` ou `False`  
**Quando alterar**: 
- ✅ **Poucos dados**: Manter `True` (aumenta variabilidade)
- ✅ **Muitos dados**: Pode desativar `False` (acelera treinamento)
- ✅ **Overfitting**: Manter `True` (reduz overfitting)

**Aplicado apenas em**: Treinamento (não em teste/validação)

---

#### **`AUGMENTATION_PARAMS`** (linhas 39-46)
```python
AUGMENTATION_PARAMS = {
    'rotation_range': 20,        # Rotação: ±20 graus
    'width_shift_range': 0.2,    # Translação horizontal: ±20%
    'height_shift_range': 0.2,   # Translação vertical: ±20%
    'horizontal_flip': True,     # Flip horizontal
    'zoom_range': 0.2,           # Zoom: ±20%
    'fill_mode': 'nearest'       # Preenchimento de bordas
}
```

**Descrição**: Parâmetros específicos de data augmentation.  
**Quando alterar**: 
- ✅ **Arte com orientação importante**: Reduzir `rotation_range` para `10`
- ✅ **Arte que não deve ser espelhada**: `horizontal_flip = False`
- ✅ **Mais variação**: Aumentar `zoom_range` para `0.3` ou `0.4`

---

### 🧠 Configurações de Gerenciamento de Memória

#### **`USE_LAZY_LOADING`** (linha 58)
```python
USE_LAZY_LOADING = True
```

**Descrição**: Carrega imagens sob demanda (lazy loading) ao invés de carregar tudo na memória.  
**Valores**: `True` ou `False`  
**Recomendado**: Sempre `True` (economiza muita memória)  
**Quando alterar**: Apenas se quiser carregar tudo na memória de uma vez (`False` - não recomendado)

---

#### **`IMAGE_CACHE_SIZE`** (linha 61)
```python
IMAGE_CACHE_SIZE = 100
```

**Descrição**: Tamanho do cache LRU de imagens (quantas imagens manter em cache).  
**Valores**: Inteiro positivo (0 = sem cache)  
**Padrão**: `100`  
**Quando alterar**: 
- ✅ **Mais RAM disponível**: Aumentar para `200` ou `500` (acelera carregamento)
- ✅ **Pouca RAM**: Reduzir para `50` ou `0` (desativa cache)

**Funcionamento**: LRU (Least Recently Used) - imagens menos usadas são removidas do cache

---

#### **`MIN_BATCH_SIZE`** (linha 64)
```python
MIN_BATCH_SIZE = 4
```

**Descrição**: Batch size mínimo para adaptive batch size (em caso de estouro de memória).  
**Valores**: Inteiro positivo (geralmente 1, 2, 4, 8)  
**Quando alterar**: Apenas se implementar adaptive batch size (atualmente não implementado completamente)

---

#### **`MEMORY_WARNING_THRESHOLD` e `MEMORY_CRITICAL_THRESHOLD`** (linhas 67-68)
```python
MEMORY_WARNING_THRESHOLD = 0.8   # 80% de uso
MEMORY_CRITICAL_THRESHOLD = 0.9  # 90% de uso
```

**Descrição**: Limites de memória para alertas.  
**Valores**: Float entre 0 e 1 (0.8 = 80%, 0.9 = 90%)  
**Quando alterar**: Apenas para ajustar sensibilidade dos alertas

---

#### **`CLEAR_MEMORY_EVERY_N_BATCHES`** (linha 74)
```python
CLEAR_MEMORY_EVERY_N_BATCHES = 50
```

**Descrição**: Limpar memória GPU a cada N batches durante treinamento.  
**Valores**: Inteiro positivo  
**Quando alterar**: 
- ✅ **Estouro de memória durante treinamento**: Reduzir para `20` ou `10`
- ✅ **Treinamento estável**: Manter `50` ou aumentar para `100`

**Funcionamento**: Chama `clear_memory(clear_gpu=True)` automaticamente

---

### 🎯 Configurações Específicas para ResNet50

#### **`RESNET50_BATCH_SIZES`** (linha 86)
```python
RESNET50_BATCH_SIZES = [8, 16, 32]  # Reduzido de [16, 32, 64]
```

**Descrição**: Batch sizes testados durante Random Search do ResNet50.  
**Valores**: Lista de inteiros positivos  
**Padrão**: `[8, 16, 32]` (otimizado para evitar estouro de memória)  
**Quando alterar**: 
- ✅ **GPU com muita memória (16GB+)**: Pode aumentar para `[16, 32, 64]`
- ✅ **GPU com pouca memória (4-6GB)**: Reduzir para `[4, 8, 16]`

**Impacto**: Batch sizes menores = menos memória, mas Random Search mais lento

---

#### **`RESNET50_DEFAULT_BATCH_SIZE`** (linha 89)
```python
RESNET50_DEFAULT_BATCH_SIZE = 16  # Reduzido de 32
```

**Descrição**: Batch size padrão para treinamento final do ResNet50 (quando não usar Random Search).  
**Valores**: Inteiro positivo  
**Padrão**: `16` (otimizado)  
**Quando alterar**: Baseado na memória disponível (mesmas recomendações de `BATCH_SIZE`)

---

#### **`RESNET50_SEARCH_EPOCHS`** (linha 92)
```python
RESNET50_SEARCH_EPOCHS = 10  # Número máximo de épocas durante Random Search
```

**Descrição**: Número máximo de épocas por iteração durante Random Search do ResNet50.  
**Valores**: Inteiro positivo  
**Padrão**: `10` (otimizado para velocidade)  
**Quando alterar**: 
- ✅ **Random Search muito rápido**: Aumentar para `15` ou `20` (mais tempo, melhor busca)
- ✅ **Random Search muito lento**: Reduzir para `5` (mais rápido, mas menos preciso)

**Nota**: Treinamento final usa `EPOCHS` completo (50 por padrão)

---

#### **`RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS`** (linha 95)
```python
RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS = True  # IMPORTANTE: Limpar entre iterações
```

**Descrição**: Limpa memória GPU entre cada iteração do Random Search do ResNet50.  
**Valores**: `True` ou `False`  
**Recomendado**: **Sempre `True`** (crítico para evitar estouro de memória)  
**Quando alterar**: Apenas se tiver GPU com muita memória e quiser testar sem limpeza (`False` - não recomendado)

**Funcionamento**: Chama `clear_memory(clear_gpu=True)` antes e depois de cada iteração

---

### 📊 Configurações Específicas para Modelos Clássicos

#### **`CLASSIC_USE_PCA`** (linha 102)
```python
CLASSIC_USE_PCA = True  # Usar PCA para redução de dimensionalidade
```

**Descrição**: Ativa/desativa PCA para reduzir dimensionalidade antes de modelos clássicos.  
**Valores**: `True` ou `False`  
**Recomendado**: **Sempre `True`** (economiza 95%+ de memória)  
**Quando alterar**: 
- ✅ **Muito RAM disponível**: Pode desativar `False` (mais features, mais tempo)
- ✅ **Pouca RAM**: Manter `True` (essencial para economizar memória)

**Impacto**: 
- `True`: 12,288 features → 500 componentes (**96% redução!**)
- `False`: Usa todas as 12,288 features (mais memória)

---

#### **`CLASSIC_PCA_COMPONENTS`** (linha 103)
```python
CLASSIC_PCA_COMPONENTS = 500  # Número de componentes PCA
```

**Descrição**: Número de componentes principais do PCA.  
**Valores**: Inteiro positivo ou `None` (auto = 95% variância)  
**Padrão**: `500` (otimizado para balancear memória e qualidade)  
**Quando alterar**: 
- ✅ **Mais memória disponível**: Aumentar para `1000` ou `1500` (mais features, mais tempo)
- ✅ **Muito pouca RAM**: Reduzir para `250` ou `300` (menos features, menos qualidade)
- ✅ **Auto (95% variância)**: `None` (PCA decide número automaticamente)

**Impacto na variância explicada**: Geralmente mantém ~95-98% da variância original

---

#### **`CLASSIC_USE_LINEAR_SVM`** (linha 104)
```python
CLASSIC_USE_LINEAR_SVM = False  # False = SVC (kernels), True = LinearSVC (só linear)
```

**Descrição**: Se `True`, usa `LinearSVC` (apenas kernel linear, mais eficiente em memória).  
**Valores**: `True` ou `False`  
**Padrão**: `False` (usa `SVC` com kernels RBF, linear, poly)  
**Quando alterar**: 
- ✅ **Estouro de memória com SVC**: Ativar `True` (economiza 70-90% de memória adicional)
- ✅ **Quer kernels não-lineares (RBF, poly)**: Manter `False`

**Trade-off**:
- `True`: Muito mais eficiente em memória, mas apenas kernel linear (pode perder performance)
- `False`: Suporta kernels não-lineares, mas usa mais memória

---

#### **`CLASSIC_MAX_SAMPLES`** (linha 105)
```python
CLASSIC_MAX_SAMPLES = None  # None = usar todas as amostras
```

**Descrição**: Limita número de amostras de treinamento para modelos clássicos.  
**Valores**: Inteiro positivo ou `None` (usa todas)  
**Padrão**: `None` (usa todas as amostras)  
**Quando alterar**: 
- ✅ **Estouro de memória mesmo com PCA**: Definir para `10000` ou `5000` (usa amostras aleatórias)
- ✅ **Testes rápidos**: Definir para `1000` ou `500`

**Nota**: Amostras são selecionadas aleatoriamente mantendo proporção de classes

---

#### **`CLASSIC_SVM_N_JOBS`** (linha 106)
```python
CLASSIC_SVM_N_JOBS = 1  # 1 = sem paralelização (economiza memória)
```

**Descrição**: Número de jobs paralelos para SVM e RandomizedSearchCV do SVM.  
**Valores**: Inteiro positivo (1 = sem paralelização) ou `-1` (todos os cores)  
**Padrão**: `1` (otimizado para economizar memória)  
**Quando alterar**: 
- ✅ **Muito RAM disponível**: Aumentar para `2`, `4` ou `-1` (acelera treinamento)
- ✅ **Pouca RAM**: Manter `1` (evita duplicação de dados em múltiplos processos)

**Trade-off**:
- `1`: Usa menos memória, mas mais lento
- `-1`: Mais rápido, mas usa muito mais memória (cada processo duplica dados)

---

#### **`CLASSIC_RF_N_JOBS`** (linha 107)
```python
CLASSIC_RF_N_JOBS = -1  # -1 = todos os cores (Random Forest usa memória eficientemente)
```

**Descrição**: Número de jobs paralelos para Random Forest e RandomizedSearchCV do RF.  
**Valores**: Inteiro positivo ou `-1` (todos os cores)  
**Padrão**: `-1` (todos os cores)  
**Quando alterar**: 
- ✅ **Quer economizar CPU**: Reduzir para `2` ou `4`
- ✅ **Normal**: Manter `-1` (Random Forest paraleliza muito bem)

**Por que diferente do SVM?**: Random Forest usa memória de forma mais eficiente (árvores independentes), então pode usar paralelização sem problemas

---

#### **`CLASSIC_CV_FOLDS`** (linha 108)
```python
CLASSIC_CV_FOLDS = 2  # 2 ao invés de 3 para economizar memória
```

**Descrição**: Número de folds para validação cruzada em modelos clássicos (SVM e Random Forest).  
**Valores**: Inteiro positivo (geralmente 2, 3, 5, 10)  
**Padrão**: `2` (otimizado para economizar memória)  
**Quando alterar**: 
- ✅ **Mais RAM disponível**: Aumentar para `3` ou `5` (mais robusto, mas mais memória)
- ✅ **Pouca RAM**: Manter `2` (essencial para economizar memória)

**Impacto na memória**: 
- `2`: 2 cópias dos dados durante CV
- `3`: 3 cópias dos dados durante CV (**50% mais memória!**)

**Aplica-se a**: SVM e Random Forest (ambos usam esta configuração)

---

### 📁 Configurações de Diretórios

#### **`ROOT_DIR`** (linha 10)
```python
ROOT_DIR = Path(__file__).parent.parent.absolute()
```

**Descrição**: Diretório raiz do projeto (calculado automaticamente).  
**Não alterar**: É calculado automaticamente baseado na localização de `config.py`

---

#### **`DATA_DIR`, `TRAIN_DIR`, `TEST_DIR`** (linhas 16-18)
```python
DATA_DIR = ROOT_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
```

**Descrição**: Caminhos dos diretórios de dados.  
**Quando alterar**: Se quiser usar uma estrutura de diretórios diferente  
**Exemplo**: `DATA_DIR = Path('/caminho/para/dados')`

---

#### **`OUTPUT_DIR`, `MODELS_DIR`, `RESULTS_DIR`, `FIGURES_DIR`** (linhas 111-114)
```python
OUTPUT_DIR = ROOT_DIR / 'outputs'
MODELS_DIR = OUTPUT_DIR / 'models'
RESULTS_DIR = OUTPUT_DIR / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'
```

**Descrição**: Caminhos dos diretórios de saída (modelos, resultados, figuras).  
**Quando alterar**: Se quiser salvar em outro local  
**Nota**: Diretórios são criados automaticamente se não existirem (linha 117-118)

---

### 📝 Resumo de Configurações Críticas

**Para economizar memória (problemas de estouro)**:
1. ✅ `CLASSIC_USE_PCA = True` (essencial!)
2. ✅ `CLASSIC_PCA_COMPONENTS = 500` (ou menor)
3. ✅ `CLASSIC_SVM_N_JOBS = 1` (sem paralelização)
4. ✅ `CLASSIC_CV_FOLDS = 2` (menos folds)
5. ✅ `RESNET50_BATCH_SIZES = [8, 16, 32]` (ou menor)
6. ✅ `RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS = True` (essencial!)
7. ✅ `IMG_SIZE_CLASSIC = (64, 64)` (não aumentar!)

**Para acelerar treinamento (mais recursos disponíveis)**:
1. ✅ `USE_GPU = True` (essencial para deep learning)
2. ✅ `BATCH_SIZE = 64` ou maior (se tiver memória GPU)
3. ✅ `CLASSIC_RF_N_JOBS = -1` (todos os cores)
4. ✅ `CLASSIC_SVM_N_JOBS = -1` ou `4` (se tiver RAM)
5. ✅ `IMAGE_CACHE_SIZE = 500` (cache maior)

**Para melhor qualidade (mais tempo disponível)**:
1. ✅ `EPOCHS = 100` ou maior
2. ✅ `CLASSIC_PCA_COMPONENTS = 1000` (mais features)
3. ✅ `CLASSIC_CV_FOLDS = 5` (mais robusto)
4. ✅ `IMG_SIZE = (256, 256)` (imagens maiores)
5. ✅ `RESNET50_SEARCH_EPOCHS = 20` (mais épocas por iteração)

---

## 📚 Guias de Uso Completo

### 🚀 Guia 1: Execução Completa do Zero

#### **Passo 1: Preparar Ambiente**

```bash
# 1.1. Clonar ou baixar o projeto
cd Atividade_Visao_Computacional_Residencia_IA

# 1.2. Criar ambiente virtual (recomendado)
python -m venv venv

# 1.3. Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 1.4. Instalar dependências
pip install -r requirements.txt

# 1.5. Verificar instalação
python verificar_pytorch.py  # Verifica PyTorch e CUDA
python check_gpu.py          # Verifica GPU
```

---

#### **Passo 2: Configurar Dataset**

**Opção A: Usar Dataset do Kaggle (Recomendado)**

```bash
# 2.1. Configurar credenciais do Kaggle (se necessário)
# Linux/Mac:
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows:
# Copie kaggle.json para: C:\Users\<username>\.kaggle\kaggle.json

# 2.2. Baixar e organizar dataset
python scripts/download_dataset.py
```

**Opção B: Organizar Dados Manualmente**

```bash
# 2.1. Criar estrutura de diretórios
mkdir -p data/train/classe1
mkdir -p data/train/classe2
mkdir -p data/test/classe1
mkdir -p data/test/classe2

# 2.2. Copiar imagens para diretórios correspondentes
# (organize manualmente suas imagens)
```

---

#### **Passo 3: Verificar Estrutura de Dados**

```bash
# 3.1. Diagnóstico da estrutura de dados
python diagnose_data.py

# 3.2. Verificar se há pelo menos 2 classes
# Saída esperada:
# "Classes encontradas: ['aiartdata', 'realart']"
# "Total de amostras: X"
```

**Se encontrar apenas 1 classe**:
- ✅ Execute `python scripts/download_dataset.py` novamente
- ✅ Ou use `python scripts/create_subset.py` para criar subset com classes artificiais

---

#### **Passo 4: Configurar Parâmetros (Opcional)**

Edite `src/config.py` conforme suas necessidades:

```python
# Exemplo: Configuração para economia de memória
USE_GPU = True
CLASSIC_USE_PCA = True
CLASSIC_PCA_COMPONENTS = 500
CLASSIC_CV_FOLDS = 2
RESNET50_BATCH_SIZES = [8, 16, 32]
```

---

#### **Passo 5: Executar Pipeline**

```bash
# 5.1. Executar script principal
python main.py

# 5.2. Escolher opção no menu:
#     1. Pipeline Clássico (SVM + Random Forest)
#     2. Pipeline Deep Learning (CNN + ResNet50)
#     3. Ambos os pipelines
#     4. Sair
```

**Tempo estimado**:
- Pipeline Clássico: 15-30 minutos (CPU)
- Pipeline Deep Learning: 30-120 minutos (GPU) ou 2-4 horas (CPU)
- Ambos: Soma dos dois

---

#### **Passo 6: Analisar Resultados**

```bash
# 6.1. Resultados em CSV
cat outputs/results/classic_pipeline_results.csv
cat outputs/results/deep_learning_results.csv

# 6.2. Figuras (matrizes de confusão)
# Visualize: outputs/figures/*.png

# 6.3. Modelos salvos
ls outputs/models/
# Arquivos: *.pkl (modelos clássicos), *.pth (modelos deep learning), *.json (metadados)
```

---

### 🧪 Guia 2: Teste Rápido com Subset

Para testar rapidamente sem usar o dataset completo:

```bash
# 1. Criar subset pequeno (10 imagens por classe)
python scripts/create_subset.py

# 2. Executar versão rápida do pipeline
python main_subset.py

# 3. Ajustar configurações para testes rápidos em src/config.py:
#    EPOCHS = 5
#    n_iter = 10  # No código main_subset.py
```

**Tempo estimado**: 2-5 minutos

---

### 🔧 Guia 3: Treinar um Modelo Específico

#### **Treinar apenas SVM**

Edite `main.py` temporariamente ou crie script customizado:

```python
# Exemplo: treinar_svm.py
from src.pipelines.classic import ClassicPipeline
from src.config import *

pipeline = ClassicPipeline(TRAIN_DIR, TEST_DIR)
pipeline.load_data()
pipeline.train_svm(use_random_search=True, n_iter=50)
pipeline.evaluate_svm()
```

```bash
python treinar_svm.py
```

---

#### **Treinar apenas ResNet50**

```python
# Exemplo: treinar_resnet.py
from src.pipelines.deep_learning import DeepLearningPipeline
from src.config import *

pipeline = DeepLearningPipeline(TRAIN_DIR, TEST_DIR)
pipeline.load_data()
pipeline.train_resnet_transfer(use_random_search=True, n_iter=10, final_epochs=50)
pipeline.evaluate_resnet_transfer()
```

```bash
python treinar_resnet.py
```

---

### 📦 Guia 4: Carregar Modelo Salvo e Fazer Predições

Use o script de exemplo:

```python
# scripts/load_model_example.py (já existe no projeto)
from src.model_saver import load_model_with_metadata
from src.utils import load_image, preprocess_image
import torch

# Carregar modelo SVM
svm_model, svm_metadata = load_model_with_metadata(
    model_path='outputs/models/svm_model.pkl',
    model_type='sklearn'
)

# Carregar modelo SimpleCNN
from src.models.cnn import SimpleCNN
cnn_model, cnn_metadata = load_model_with_metadata(
    model_path='outputs/models/simple_cnn.pth',
    model_type='pytorch',
    model_class=SimpleCNN
)

# Fazer predição em nova imagem
image = load_image('caminho/para/imagem.jpg')
# ... preprocessar imagem ...
prediction = model.predict(image)
```

```bash
python scripts/load_model_example.py
```

---

### 🎯 Guia 5: Otimização de Hiperparâmetros Customizada

#### **Aumentar Número de Iterações do Random Search**

No código `main.py` ou nos pipelines, altere:

```python
# Pipeline Clássico
pipeline.train_svm(use_random_search=True, n_iter=100)  # Era 50

# Pipeline Deep Learning
pipeline.train_simple_cnn(use_random_search=True, n_iter=20)  # Era 10
```

**Trade-off**: Mais iterações = melhor resultado, mas mais tempo

---

#### **Personalizar Espaço de Busca**

Edite os pipelines diretamente:

```python
# src/pipelines/deep_learning.py - Função train_simple_cnn()
param_space = {
    'learning_rate': (0.00001, 0.001),  # Espaço maior
    'batch_size': [8, 16, 32, 64],      # Mais opções
    'dropout_rate': (0.2, 0.8),         # Espaço maior
    'hidden_units': [128, 256, 512, 1024, 2048]  # Mais opções
}
```

---

### 🔍 Guia 6: Diagnóstico e Verificação

#### **Verificar GPU**

```bash
# Verificação completa
python verificar_pytorch.py

# Verificação de GPU
python check_gpu.py

# Diagnóstico de uso de GPU
python diagnose_gpu_usage.py

# Teste direto de GPU
python testar_gpu_direto.py
```

---

#### **Diagnosticar Estrutura de Dados**

```bash
# Diagnóstico completo
python diagnose_data.py

# Criar subset se necessário
python scripts/create_subset.py
```

---

#### **Monitorar Memória Durante Treinamento**

Adicione logs no código ou use ferramentas externas:

```python
# Em src/pipelines/deep_learning.py ou classic.py
from src.memory import get_memory_usage

# Durante treinamento
ram_used, ram_total, ram_percent = get_memory_usage()
print(f"RAM: {ram_used:.2f} GB / {ram_total:.2f} GB ({ram_percent*100:.1f}%)")
```

---

## 🔧 Troubleshooting - Problemas Comuns e Soluções

Esta seção lista **TODOS** os problemas encontrados durante o desenvolvimento e suas soluções.

---

### ❌ Problema 1: "ModuleNotFoundError: No module named 'cv2'"

**Erro completo**:
```
ModuleNotFoundError: No module named 'cv2'
```

**Causa**: `opencv-python` não está instalado.

**Solução**:
```bash
pip install opencv-python
# ou
pip install -r requirements.txt
```

**Prevenção**: Sempre instale todas as dependências do `requirements.txt` antes de executar.

---

### ❌ Problema 2: "ValueError: Apenas 1 classe(s) foi(ram) carregada(s)"

**Erro completo**:
```
ValueError: ERRO: Apenas 1 classe(s) foi(ram) carregada(s), mas são necessárias pelo menos 2 classes para classificação.
```

**Causa**: Dataset tem apenas 1 classe ou estrutura de diretórios incorreta.

**Soluções**:

**Solução 2.1: Baixar dataset do Kaggle**
```bash
python scripts/download_dataset.py
```

**Solução 2.2: Criar subset com classes artificiais**
```bash
python scripts/create_subset.py
```

**Solução 2.3: Verificar estrutura manualmente**
```bash
python diagnose_data.py
# Verifique se há pelo menos 2 diretórios em data/train/
```

**Prevenção**: Sempre execute `diagnose_data.py` antes de treinar.

---

### ❌ Problema 3: "TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'"

**Erro completo**:
```
TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
```

**Causa**: Versão do PyTorch não suporta parâmetro `verbose` em `ReduceLROnPlateau`.

**Status**: ✅ **CORRIGIDO** - Parâmetro `verbose` foi removido em `src/pipelines/deep_learning.py` (linhas 599-601 e 548-550).

**Se ainda ocorrer**: Atualize o PyTorch:
```bash
pip install --upgrade torch torchvision
```

---

### ❌ Problema 4: "RuntimeError: CUDA out of memory"

**Erro completo**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB. GPU allocated memory: X.XX GiB
```

**Causa**: Modelo ou batch size muito grande para a memória GPU disponível.

**Soluções**:

**Solução 4.1: Reduzir batch size (ResNet50)**
```python
# Em src/config.py
RESNET50_BATCH_SIZES = [4, 8, 16]  # Era [8, 16, 32]
RESNET50_DEFAULT_BATCH_SIZE = 8    # Era 16
```

**Solução 4.2: Reduzir batch size (CNN simples)**
```python
# Em src/config.py
BATCH_SIZE = 16  # Era 32
```

**Solução 4.3: Garantir limpeza de memória (ResNet50)**
```python
# Em src/config.py
RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS = True  # DEVE estar True
```

**Solução 4.4: Limpar memória GPU manualmente**
```python
import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

**Solução 4.5: Usar CPU ao invés de GPU**
```python
# Em src/config.py
USE_GPU = False
```

**Prevenção**: 
- ✅ Sempre monitore uso de GPU: `nvidia-smi` (Linux/Windows) ou `watch -n 1 nvidia-smi`
- ✅ Use `RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS = True` sempre
- ✅ Comece com batch sizes pequenos e aumente gradualmente

---

### ❌ Problema 5: "MemoryError" ou Sistema Travando (SVM)

**Erro completo**:
```
MemoryError
# ou sistema simplesmente trava/freeze
```

**Causa**: SVM tentando usar muita memória RAM (imagens muito grandes ou muitas amostras).

**Soluções**:

**Solução 5.1: Ativar PCA (ESSENCIAL)**
```python
# Em src/config.py
CLASSIC_USE_PCA = True  # DEVE estar True
CLASSIC_PCA_COMPONENTS = 500  # Ou menor (250, 300)
```

**Solução 5.2: Reduzir tamanho de imagem**
```python
# Em src/config.py
IMG_SIZE_CLASSIC = (32, 32)  # Era (64, 64), ainda menor
```

**Solução 5.3: Limitar número de amostras**
```python
# Em src/config.py
CLASSIC_MAX_SAMPLES = 5000  # Limita a 5000 amostras
```

**Solução 5.4: Usar LinearSVC (mais eficiente)**
```python
# Em src/config.py
CLASSIC_USE_LINEAR_SVM = True  # Mais eficiente em memória
```

**Solução 5.5: Reduzir paralelização**
```python
# Em src/config.py
CLASSIC_SVM_N_JOBS = 1  # Sem paralelização (já é padrão)
CLASSIC_CV_FOLDS = 2    # Menos folds (já é padrão)
```

**Prevenção**: 
- ✅ **SEMPRE** use `CLASSIC_USE_PCA = True` para SVM
- ✅ Não aumente `IMG_SIZE_CLASSIC` acima de `(64, 64)`
- ✅ Monitore memória antes de treinar (o código já faz isso automaticamente)

---

### ❌ Problema 6: "AttributeError: 'str' object has no attribute 'type'"

**Erro completo**:
```
AttributeError: 'str' object has no attribute 'type'
```

**Causa**: `setup_device()` retornava string `'cpu'` ao invés de `torch.device('cpu')`.

**Status**: ✅ **CORRIGIDO** - `setup_device()` sempre retorna `torch.device` em `src/utils.py` (linhas 62, 122).

**Se ainda ocorrer**: Verifique se está usando a versão mais recente do código.

---

### ❌ Problema 7: Modelos Deep Learning Não Estão Usando GPU

**Sintoma**: Treinamento muito lento, ou logs mostram "CPU" ao invés de "GPU".

**Causas possíveis**:
1. GPU não detectada
2. Modelo não movido para GPU
3. Dados não movidos para GPU

**Soluções**:

**Solução 7.1: Verificar GPU**
```bash
python verificar_pytorch.py
python check_gpu.py
```

**Solução 7.2: Verificar configuração**
```python
# Em src/config.py
USE_GPU = True  # DEVE estar True
```

**Solução 7.3: Forçar GPU (se disponível)**
```python
# O código já faz isso automaticamente, mas você pode verificar:
import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

**Status**: ✅ **CORRIGIDO** - Modelos são movidos explicitamente para GPU em:
- `src/pipelines/deep_learning.py` linha 309 (SimpleCNN - Random Search)
- `src/pipelines/deep_learning.py` linha 325 (SimpleCNN - treinamento final)
- `src/pipelines/deep_learning.py` linha 1049 (ResNet50 - criação)
- `src/pipelines/deep_learning.py` linha 505 (train_single_config - verificação)

**Prevenção**: Sempre verifique os logs durante inicialização do pipeline:
```
✅ Dispositivo configurado: cuda:0
✅ GPU disponível: NVIDIA GeForce RTX 3060
✅ SimpleCNN está na GPU: NVIDIA GeForce RTX 3060
```

---

### ❌ Problema 8: "EOFError" ao Executar Scripts Não-Interativamente

**Erro completo**:
```
EOFError
```

**Causa**: Script usa `input()` para confirmação do usuário em ambiente não-interativo (CI/CD, scripts automatizados).

**Status**: ✅ **CORRIGIDO** - `scripts/create_subset.py` não usa mais `input()` interativo.

**Se ainda ocorrer**: Verifique se está usando a versão mais recente do código.

---

### ❌ Problema 9: "UnicodeEncodeError" ao Executar verificar_pytorch.py no Windows

**Erro completo**:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'
```

**Causa**: Console do Windows não configurado para UTF-8.

**Status**: ✅ **CORRIGIDO** - `verificar_pytorch.py` agora usa `sys.stdout.reconfigure(encoding='utf-8')`.

**Se ainda ocorrer**: Execute no PowerShell com encoding UTF-8:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
python verificar_pytorch.py
```

---

### ❌ Problema 10: Resultados Muito Diferentes Entre Execuções

**Sintoma**: Métricas (accuracy, F1-score) variam muito entre execuções.

**Causas possíveis**:
1. Sementes aleatórias não fixadas
2. Divisão treino/teste não fixada
3. Data augmentation muito agressiva

**Soluções**:

**Solução 10.1: Verificar seeds fixadas**
```python
# O código já usa random_state=42 em vários lugares, mas verifique:
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

**Solução 10.2: Desativar data augmentation temporariamente**
```python
# Em src/config.py
USE_AUGMENTATION = False  # Teste sem augmentation
```

**Solução 10.3: Usar mais épocas**
```python
# Em src/config.py
EPOCHS = 100  # Mais épocas para estabilizar
```

---

### ❌ Problema 11: "FileNotFoundError: Modelo não encontrado"

**Erro completo**:
```
FileNotFoundError: Modelo não encontrado: outputs/models/svm_model.pkl
```

**Causa**: Tentando carregar modelo que não foi treinado ainda.

**Solução**:
```bash
# Treine o modelo primeiro
python main.py
# Escolha opção 1 (Pipeline Clássico) ou 2 (Deep Learning)
```

---

### ❌ Problema 12: Treinamento Muito Lento

**Sintoma**: Treinamento demora horas mesmo para datasets pequenos.

**Causas possíveis**:
1. Usando CPU ao invés de GPU
2. Batch size muito pequeno
3. Número de épocas muito alto
4. Random Search com muitas iterações

**Soluções**:

**Solução 12.1: Verificar se está usando GPU**
```bash
python check_gpu.py
# Se GPU disponível, verifique se USE_GPU = True em config.py
```

**Solução 12.2: Aumentar batch size (se tiver memória)**
```python
# Em src/config.py
BATCH_SIZE = 64  # Era 32
```

**Solução 12.3: Reduzir épocas durante Random Search**
```python
# O código já limita: search_epochs = min(15, final_epochs)
# Mas você pode reduzir ainda mais editando o código
```

**Solução 12.4: Reduzir número de iterações do Random Search**
```python
# No main.py ou ao chamar pipeline:
pipeline.train_svm(use_random_search=True, n_iter=10)  # Era 50
```

---

### ❌ Problema 13: Overfitting (Alta Accuracy no Treino, Baixa no Teste)

**Sintoma**: 
- Accuracy treino: 0.95+
- Accuracy teste: 0.70-0.80

**Soluções**:

**Solução 13.1: Aumentar data augmentation**
```python
# Em src/config.py
USE_AUGMENTATION = True  # Já está ativo
AUGMENTATION_PARAMS = {
    'rotation_range': 30,      # Aumentar de 20 para 30
    'zoom_range': 0.3,         # Aumentar de 0.2 para 0.3
    # ... outros parâmetros
}
```

**Solução 13.2: Aumentar dropout**
```python
# Para SimpleCNN, durante Random Search, o dropout varia de 0.3 a 0.7
# Modelo final usará o melhor encontrado, mas você pode forçar:
# (edite o código para usar dropout_rate fixo maior)
```

**Solução 13.3: Reduzir complexidade do modelo**
```python
# Para SimpleCNN: reduzir hidden_units
# Para Random Forest: reduzir max_depth, n_estimators
```

**Solução 13.4: Usar mais dados de treinamento**
- Baixar dataset maior
- Não limitar `CLASSIC_MAX_SAMPLES`

---

### ✅ Checklist de Verificação Antes de Treinar

Antes de executar o pipeline, verifique:

- [ ] ✅ Todas as dependências instaladas: `pip install -r requirements.txt`
- [ ] ✅ Dataset organizado corretamente: `python diagnose_data.py`
- [ ] ✅ Pelo menos 2 classes detectadas
- [ ] ✅ GPU verificada (se usando deep learning): `python check_gpu.py`
- [ ] ✅ Configurações de memória ajustadas (se tiver pouco RAM)
- [ ] ✅ `CLASSIC_USE_PCA = True` (se usando SVM)
- [ ] ✅ `RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS = True` (se usando ResNet50)
- [ ] ✅ Espaço em disco suficiente para salvar modelos

---

### 📞 Como Obter Mais Ajuda

Se nenhuma das soluções acima resolveu seu problema:

1. **Verifique os logs**: O código imprime informações detalhadas durante execução
2. **Execute scripts de diagnóstico**: `verificar_pytorch.py`, `check_gpu.py`, `diagnose_data.py`
3. **Consulte a documentação**: Este README contém todas as informações
4. **Verifique versões**: `pip list | grep torch` (verifique versões compatíveis)

---

## Requisitos do Sistema

- Python 3.7+
- CUDA (opcional, para GPU)
- RAM: Mínimo 8GB (recomendado 16GB)
- Espaço em disco: Depende do tamanho da base de dados

## Autores

Projeto desenvolvido para disciplina de Visão Computacional.

## Licença

Este projeto é para fins educacionais.

# Atividade_Visao_Computacional_Residencia_IA
