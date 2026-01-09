# Projeto de Classificação de Imagens - Visão Computacional

Projeto completo de classificação de imagens utilizando pipelines clássico e deep learning, com comparação de modelos e otimização de hiperparâmetros.

## Estrutura do Projeto

```
.
├── main.py                      # Ponto de entrada principal
├── requirements.txt             # Dependências
├── README.md                    # Documentação
├── .gitignore                   # Arquivos ignorados pelo git
│
├── src/                         # Código fonte principal
│   ├── __init__.py
│   ├── config.py                # Configurações do projeto
│   ├── utils.py                 # Funções utilitárias
│   │
│   ├── models/                  # Definições de modelos
│   │   ├── __init__.py
│   │   └── cnn.py               # Arquitetura CNN
│   │
│   └── pipelines/               # Pipelines de treinamento
│       ├── __init__.py
│       ├── classic.py           # Pipeline clássico (SVM, Random Forest)
│       └── deep_learning.py     # Pipeline deep learning (CNN, ResNet)
│
├── scripts/                     # Scripts auxiliares
│   ├── __init__.py
│   └── download_dataset.py      # Download do dataset Kaggle
│
├── notebooks/                   # Jupyter notebooks (exploração)
├── tests/                       # Testes unitários
├── docs/                        # Documentação adicional
│
├── data/                        # Dados (ignorado pelo git)
│   ├── train/                   # Imagens de treinamento
│   │   ├── classe1/
│   │   └── classe2/
│   └── test/                    # Imagens de teste
│       ├── classe1/
│       └── classe2/
│
└── outputs/                     # Resultados (ignorado pelo git)
    ├── models/                  # Modelos treinados (.pkl, .pth)
    ├── results/                 # Resultados em CSV
    └── figures/                 # Gráficos e visualizações
```

## Instalação

### Requisitos

```bash
pip install -r requirements.txt
```

Ou instale manualmente:

```bash
pip install torch torchvision
pip install scikit-learn
pip install opencv-python
pip install matplotlib seaborn
pip install pandas numpy
pip install joblib
pip install kagglehub
```

### Configuração do Kaggle

Para usar o dataset do Kaggle, você precisa:

1. **Criar uma conta no Kaggle**: https://www.kaggle.com/
2. **Aceitar os termos do dataset**: Acesse o dataset [AI Art vs Human Art](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art) e aceite os termos
3. **Configurar credenciais do Kaggle** (opcional, mas recomendado):
   - Baixe seu arquivo `kaggle.json` das configurações da conta
   - Coloque em `~/.kaggle/kaggle.json` (Linux/Mac) ou `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

### Configuração

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

## Pipeline Clássico

### Modelos Implementados

1. **Support Vector Machine (SVM)**
   - Kernel: RBF, Linear, Polinomial
   - Otimização: Random Search (50 iterações)
   - Parâmetros otimizados: C, gamma, kernel, degree, class_weight

2. **Random Forest**
   - Ensemble de árvores de decisão
   - Otimização: Random Search (50 iterações)
   - Parâmetros otimizados: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, class_weight

### Transformações Aplicadas

1. **Carregamento de Imagens** (com Padronização Completa)
   - Leitura com PIL/OpenCV (suporta múltiplos formatos)
   - Correção de orientação EXIF
   - Conversão para RGB (3 canais)
   - Remoção de transparência (alpha channel)
   - Conversão de grayscale para RGB
   - Redimensionamento para tamanho padrão (224x224)
   - Validação de qualidade e dimensões

2. **Pré-processamento**
   - Flatten das imagens (transformação em vetor 1D)
   - Normalização para [0, 1] (divisão por 255)
   - Padronização com StandardScaler (média 0, desvio padrão 1)

3. **Valores dos Parâmetros**

**SVM (Random Search - 50 iterações):**
- C: log-uniform [0.01, 100]
- gamma: log-uniform [0.0001, 1]
- kernel: ['rbf', 'linear', 'poly']
- degree: randint [2, 5] (para kernel poly)
- class_weight: [None, 'balanced']
- Validação cruzada: 3 folds

**Random Forest (Random Search - 50 iterações):**
- n_estimators: randint [50, 300]
- max_depth: [None, 10, 20, 30, 50]
- min_samples_split: randint [2, 20]
- min_samples_leaf: randint [1, 10]
- max_features: ['sqrt', 'log2', None]
- bootstrap: [True, False]
- class_weight: [None, 'balanced', 'balanced_subsample']
- Validação cruzada: 3 folds

### Métricas Utilizadas

- Acurácia (Accuracy)
- Precisão (Precision)
- Recall
- F1-Score
- Matriz de Confusão

## Pipeline Deep Learning

### Modelos Implementados

1. **CNN Simples (sem Transfer Learning)**
   - Arquitetura: 3 camadas convolucionais + 2 camadas fully connected
   - Parâmetros: ~2.5 milhões (variável conforme hidden_units)
   - Treinamento do zero
   - Otimização: Random Search (10 iterações)

2. **ResNet50 (com Transfer Learning)**
   - Base pré-treinada: ImageNet (IMAGENET1K_V2)
   - Camadas convolucionais: Congeladas (configurável)
   - Camada final: Substituída e treinada
   - Otimização: Random Search (10 iterações)

### Configuração de Treinamento

**Parâmetros Padrão:**
- Batch Size: 32
- Número de Épocas: 50
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Learning Rate Scheduler: ReduceLROnPlateau (reduz LR quando loss para de melhorar)

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

### Otimização de Hiperparâmetros (Random Search)

**CNN Simples (Random Search - 10 iterações):**
- learning_rate: log-uniform [0.0001, 0.01]
- batch_size: [16, 32, 64]
- dropout_rate: uniform [0.3, 0.7]
- hidden_units: [256, 512, 1024]
- Validação: 20% split com early stopping (patience=5)
- Épocas de busca: 15 (reduzidas para eficiência)

**ResNet50 (Random Search - 10 iterações):**
- learning_rate: log-uniform [0.00001, 0.001]
- batch_size: [16, 32, 64]
- unfreeze_layers: [0, 1, 2] (quantidade de camadas a descongelar)
  - 0: apenas camada FC
  - 1: FC + layer4
  - 2: FC + layer4 + layer3
- Validação: 20% split com early stopping (patience=5)
- Épocas de busca: 10 (reduzidas para eficiência)

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
