# Solução: Estouro de Memória no ResNet50

**Data**: 11 de janeiro de 2026  
**Problema**: ResNet50 está causando erro de "Out of Memory" (OOM) na GPU

---

## 🔴 Problema Identificado

O ResNet50 está causando estouro de memória na GPU durante o treinamento, mesmo com as configurações otimizadas.

**GPU disponível**: NVIDIA GeForce RTX 3050 (8GB)  
**Modelo**: ResNet50 (25M+ parâmetros)

---

## ✅ Solução Aplicada

### Redução de Batch Sizes

**Configuração anterior**:
```python
RESNET50_BATCH_SIZES = [8, 16, 32]
RESNET50_DEFAULT_BATCH_SIZE = 16
```

**Nova configuração (em `src/config.py`)**:
```python
RESNET50_BATCH_SIZES = [4, 8]  # Reduzido para evitar OOM
RESNET50_DEFAULT_BATCH_SIZE = 8  # Reduzido de 16 para 8
```

### Mudanças Específicas

1. **Batch sizes no Random Search**: Reduzido de `[8, 16, 32]` para `[4, 8]`
   - Remove batch size 32 (muito grande para GPU de 8GB)
   - Remove batch size 16 (pode causar OOM dependendo do dataset)
   - Mantém apenas batch sizes menores (4 e 8)

2. **Batch size padrão**: Reduzido de `16` para `8`
   - Usado quando não há Random Search
   - Mais seguro para evitar OOM

---

## 📊 Impacto Esperado

### Memória GPU

Com batch size reduzido:
- **Batch size 4**: ~2-3 GB de memória GPU
- **Batch size 8**: ~4-5 GB de memória GPU
- **Batch size 16**: ~6-8 GB de memória GPU (pode causar OOM)
- **Batch size 32**: ~10-12 GB de memória GPU (causa OOM em GPU de 8GB)

### Tempo de Treinamento

- ⚠️ **Tempo pode aumentar** com batch sizes menores
- ✅ Mas será **mais estável** e não causará OOM
- Trade-off: Estabilidade vs. Velocidade

### Performance do Modelo

- Batch sizes menores podem ter **ligeira diferença** na performance
- Mas geralmente não é significativa
- Mais importante: **Treinar sem OOM** do que ter batch size grande

---

## 🔧 Configurações Adicionais Já Ativas

O código já possui outras otimizações de memória:

1. **Limpeza de memória entre iterações**: `RESNET50_CLEAR_MEMORY_BETWEEN_ITERATIONS = True`
2. **Épocas reduzidas no Random Search**: `RESNET50_SEARCH_EPOCHS = 10`
3. **Tratamento de OOM**: O código tenta reduzir batch size automaticamente se OOM ocorrer

---

## 📝 Recomendações Adicionais

Se ainda houver problemas de memória, considere:

### Opção 1: Reduzir ainda mais o batch size
```python
RESNET50_BATCH_SIZES = [2, 4]  # Ainda menor
RESNET50_DEFAULT_BATCH_SIZE = 4
```

### Opção 2: Reduzir tamanho da imagem
```python
IMG_SIZE = (128, 128)  # Ao invés de (224, 224)
```
⚠️ **Nota**: Isso pode afetar a performance do modelo

### Opção 3: Reduzir número de épocas no Random Search
```python
RESNET50_SEARCH_EPOCHS = 5  # Reduzido de 10
```

### Opção 4: Usar Gradient Accumulation (não implementado)
- Treinar com batch size pequeno mas acumular gradientes
- Requer modificação no código de treinamento

---

## ✅ Próximos Passos

1. ✅ **Configuração aplicada**: Batch sizes reduzidos
2. ⏳ **Testar**: Executar treinamento do ResNet50 novamente
3. 📊 **Monitorar**: Verificar uso de memória GPU durante treinamento
4. 🔄 **Ajustar se necessário**: Se ainda houver OOM, reduzir mais

---

## 💡 Notas Técnicas

### Por que ResNet50 usa tanta memória?

1. **Modelo grande**: 25+ milhões de parâmetros
2. **Ativações**: Armazenadas durante forward/backward pass
3. **Gradientes**: Armazenados para cada parâmetro
4. **Batch size**: Multiplica memória necessária

### Fórmula aproximada de memória:

```
Memória ≈ (Modelo + Ativações + Gradientes) × Batch Size
```

Para ResNet50:
- Modelo: ~100 MB
- Por imagem (224×224): ~50-100 MB
- Batch size 8: ~400-800 MB por batch
- Total: ~1-2 GB por batch (mas pode picoar mais alto)

---

**Arquivo criado em**: 2026-01-11  
**Status**: Correção aplicada
