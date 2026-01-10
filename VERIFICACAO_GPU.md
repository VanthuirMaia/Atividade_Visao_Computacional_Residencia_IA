# Guia de Verificação de GPU/CPU

## 📋 Resumo

**GPU Detectada**: ✅ NVIDIA GeForce RTX 3050 (8GB)  
**PyTorch com CUDA**: ✅ Versão 2.7.1+cu118  
**CUDA**: ✅ Versão 11.8  
**Status**: ✅ **GPU ESTÁ SENDO USADA**

---

## 🔍 Como Verificar se GPU/CPU Está Sendo Usado

### Método 1: Script de Verificação (Recomendado)

Execute o script de verificação:

```bash
python check_gpu.py
```

Este script verifica:
- ✅ Instalação do PyTorch
- ✅ Disponibilidade de CUDA
- ✅ Detalhes da GPU (nome, memória, versão)
- ✅ Configuração do projeto (USE_GPU)
- ✅ Teste prático de criação de tensores
- ✅ Status final (GPU ou CPU)

**Saída esperada se GPU estiver disponível:**
```
[OK] STATUS: GPU ESTA SENDO USADA
   GPU: NVIDIA GeForce RTX 3050
   Config USE_GPU: True
   Memoria Total: 8.00 GB
```

---

### Método 2: Durante o Treinamento

Ao executar o pipeline de Deep Learning, o sistema mostra automaticamente:

1. **Na inicialização do pipeline:**
```
============================================================
CONFIGURAÇÃO DE DISPOSITIVO
============================================================
✅ Usando GPU: NVIDIA GeForce RTX 3050
   Número de GPUs disponíveis: 1
   Memória GPU:
     Total: 8.00 GB
     Livre: 7.98 GB
     Usada: 0.02 GB
============================================================
```

2. **No primeiro batch do treinamento:**
```
============================================================
   VERIFICACAO DE DISPOSITIVO (Batch 1, Epoca 1)
============================================================
   Modelo esta em: cuda:0
   Imagens estao em: cuda:0
   Labels estao em: cuda:0
   [OK] CONFIRMADO: Dados estao na GPU!
   GPU: NVIDIA GeForce RTX 3050
   Memoria GPU em uso: 150.25 MB
============================================================
```

---

### Método 3: Verificação Manual no Código

Você pode verificar programaticamente:

```python
import torch
from src.utils import setup_device
from src.config import USE_GPU

# Verificar disponibilidade
print(f"CUDA disponivel: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# Verificar dispositivo configurado
device = setup_device(use_gpu=USE_GPU)
print(f"Dispositivo usado: {device}")

# Teste prático
tensor = torch.randn(100, 100).to(device)
print(f"Tensor criado em: {tensor.device}")
```

---

## ⚙️ Configuração

### Alterar entre GPU e CPU

Edite `src/config.py`:

```python
# Para usar GPU (padrão)
USE_GPU = True

# Para forçar CPU
USE_GPU = False
```

### Verificar Configuração Atual

```bash
# Opção 1: Verificar arquivo diretamente
cat src/config.py | grep USE_GPU

# Opção 2: Usar script de verificação
python check_gpu.py
```

---

## 🔧 Troubleshooting

### Problema: GPU não está sendo detectada

**Sintomas:**
- `CUDA disponível: False`
- Mensagem "GPU não disponível, usando CPU"

**Soluções:**

1. **Verificar instalação do PyTorch com CUDA:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Verificar drivers NVIDIA:**
   ```bash
   nvidia-smi
   ```

3. **Reinstalar PyTorch com CUDA:**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verificar versão de CUDA:**
   - Sistema deve ter CUDA instalado
   - PyTorch deve ser compatível com versão CUDA do sistema

### Problema: USE_GPU=True mas usando CPU

**Causas possíveis:**
- PyTorch instalado sem suporte CUDA
- GPU não compatível
- Drivers NVIDIA não instalados

**Solução:** O código automaticamente usa CPU quando GPU não está disponível, mesmo com `USE_GPU=True`.

### Problema: Memória GPU insuficiente

**Sintomas:**
- Erro "out of memory"
- Treinamento muito lento

**Soluções:**

1. **Reduzir batch size** em `src/config.py`:
   ```python
   BATCH_SIZE = 16  # Reduzir de 32 para 16
   ```

2. **Ativar lazy loading** (já ativado por padrão):
   ```python
   USE_LAZY_LOADING = True
   ```

3. **Reduzir cache de imagens:**
   ```python
   IMAGE_CACHE_SIZE = 50  # Reduzir de 100 para 50
   ```

4. **Limpar memória GPU periodicamente:**
   - Já está implementado automaticamente
   - Limpa a cada 50 batches

---

## 📊 Informações da GPU Detectada

**GPU Atual:**
- **Nome**: NVIDIA GeForce RTX 3050
- **Memória Total**: 8.00 GB
- **Capacidade CUDA**: (8, 6)
- **Versão CUDA**: 11.8
- **Versão cuDNN**: 90100

**Status de Uso:**
- ✅ GPU está disponível e configurada
- ✅ PyTorch detecta a GPU corretamente
- ✅ Configuração `USE_GPU=True` está correta
- ✅ Testes práticos confirmam uso da GPU

---

## ✅ Verificações Implementadas

O código agora verifica automaticamente:

1. ✅ **Na inicialização do pipeline** - Mostra dispositivo configurado
2. ✅ **No primeiro batch do treinamento** - Confirma localização dos dados
3. ✅ **Durante Random Search** - Verifica dispositivo em cada iteração
4. ✅ **No script de verificação** - Teste completo de GPU/CPU

---

## 🎯 Próximos Passos

1. **Execute o script de verificação:**
   ```bash
   python check_gpu.py
   ```

2. **Execute o pipeline de Deep Learning** para ver as verificações em tempo real:
   ```bash
   python main.py
   # Escolha opção 2 (Pipeline Deep Learning)
   ```

3. **Monitore uso de memória GPU** durante o treinamento:
   - O código mostra automaticamente
   - Ou use `nvidia-smi` em outro terminal

---

## 📝 Notas Importantes

- ⚠️ **Pipeline Clássico (SVM/RF)** sempre usa CPU (não usa GPU)
- ✅ **Pipeline Deep Learning** usa GPU se disponível
- ✅ O código **automaticamente** detecta e usa GPU quando disponível
- ✅ Se GPU não estiver disponível, **automaticamente** usa CPU
- ✅ Verificações são feitas **automaticamente** durante o treinamento

---

**Última verificação**: Script `check_gpu.py` executado com sucesso  
**Status**: ✅ GPU configurada e pronta para uso
