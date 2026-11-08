# Parâmetros de treinamento HRSID

Este documento explica cada parâmetro de `notebooks/hrsid_mask_rcnn_transfer_learning.ipynb`: o que controla, o valor atual e o efeito de mudá-lo. O experimento DeepPCB é independente e continua documentado em [parametros_treinamento.md](parametros_treinamento.md).

Os parâmetros ficam no dicionário `CONFIG`, na seção 2 do notebook. Depois de mudar o `CONFIG`, reexecute a célula dele; `batch_size` e `num_workers` só passam a valer depois de recriar os DataLoaders (seção 8).

> **Regra para comparar estratégias:** `frozen_backbone`, `partial_finetune` e `full_finetune` devem usar os mesmos valores em tudo, exceto `experiment` e `trainable_layers`. Cada rodada salva o `CONFIG` usado em `config.json`.

---

## 1. Reprodutibilidade e dados

### `seed` = `42`

Semente de Python, NumPy e PyTorch (CPU e CUDA). Controla a inicialização das cabeças novas, a ordem de embaralhamento do treino, o espelhamento horizontal, a amostra das checagens de pixel e a escolha das cenas de validação. Na GPU, algumas operações não são determinísticas: rodar com 42, 43 e 44 mostra a variação natural do modelo.

### `dataset_dir` = `''`

Pasta do HRSID já extraído. Aceita qualquer caminho: pasta local, disco externo ou Google Drive montado ou sincronizado. Vazio, vale o padrão do ambiente — `data/HRSID/` na raiz do projeto quando a execução é local, ou `MyDrive/vision-transfer-learning/dataset/HRSID/` no Colab.

A estrutura interna é descoberta na seção 4: a pasta pode conter `HRSID_JPG/annotations/` e `HRSID_JPG/JPEGImages/` ou já ser a pasta das anotações. O notebook não baixa, não descompacta e não copia o dataset; se o caminho não existir, a seção 3 para com uma mensagem explicando o que configurar.

### `results_dir` = `None`

Sobrescreve o destino dos resultados. Com `None`, vale o padrão: `runs/results/hrsid_mask_rcnn_v2/` na raiz do projeto quando a execução é local, ou em `MyDrive/vision-transfer-learning/runs/results/hrsid_mask_rcnn_v2/` no Colab, onde os arquivos sobrevivem ao fim da sessão.

### `val_fraction` = `0.15`

Fração de imagens do treino oficial reservada para validação. A separação leva cenas inteiras: recortes da mesma cena SAR nunca ficam dos dois lados. Como cada cena tem um número diferente de recortes, a fração real fica próxima, não exata.

### `limit_images` = `None`

Limita o número de imagens por split. Serve para ensaios rápidos (`200`, por exemplo) e para verificar o notebook ponta a ponta em minutos. Com `None`, usa o dataset inteiro. Resultados obtidos com limite não são comparáveis com os do dataset completo.

### `sample_images` = `200`

Quantas imagens por split são realmente abertas nas checagens de pixel: formato, canais, dimensão real e decodificação das máscaras. Contagens, ids, nomes, caixas e estrutura das segmentações são conferidos em **todas** as anotações; só a leitura dos arquivos é amostrada, porque abrir 5.604 imagens é lento, principalmente num Drive montado. O relatório informa quantas foram abertas.

---

## 2. Estratégia de transferência

### `experiment` = `'frozen_backbone'`

Escolhe o que recebe gradiente. As três estratégias partem dos mesmos pesos COCO e usam o mesmo detector, o Mask R-CNN ResNet-50 FPN V2.

| Valor | Treinado | Congelado | Parâmetros treináveis |
| --- | --- | --- | --- |
| `'frozen_backbone'` | RPN e cabeças de detecção e de máscara | ResNet-50 e FPN | ~22 M de 68 M |
| `'partial_finetune'` | blocos finais do ResNet-50, FPN, RPN e cabeças | blocos iniciais do ResNet-50 | depende de `trainable_layers` |
| `'full_finetune'` | tudo | nada | ~68 M |

- **frozen_backbone** mede quanto das características do COCO serve para SAR sem nenhum ajuste. É a mais rápida por época e a que menos consome memória.
- **partial_finetune** adapta as camadas de alto nível, que são as mais específicas de domínio, sem mexer nas camadas iniciais de borda e textura.
- **full_finetune** adapta tudo e costuma dar o melhor resultado, com o maior custo e o maior risco de sobreajuste.

Nos três casos, `FastRCNNPredictor` e `MaskRCNNPredictor` são trocados por versões de duas classes: fundo e navio.

### `trainable_layers` = `2`

Só vale em `partial_finetune`. É o `trainable_backbone_layers` oficial do torchvision: quantos blocos do ResNet-50 recebem gradiente, contados do fim para o começo. `1` libera `layer4`; `2` libera `layer3` e `layer4`; `3` acrescenta `layer2`. Valores maiores aproximam o resultado do fine-tuning completo.

O `maskrcnn_resnet50_fpn_v2` usa `BatchNorm2d` comum, e não `FrozenBatchNorm2d`. Por isso o notebook também coloca os blocos congelados em `eval()` durante o treino: sem isso, as médias e variâncias continuariam sendo atualizadas e os pesos "congelados" mudariam de comportamento.

---

## 3. Duração do treinamento

### `epochs` = `15`

Número máximo de épocas. Uma época passa pelas imagens de treino e avalia a validação inteira. O treino pode parar antes por `patience`.

### `patience` = `5`

Parada antecipada. Se a mAP@0.50:0.95 de máscara na validação não melhorar por 5 épocas seguidas, o treino para. Só a melhor época fica em `checkpoints/best.pt`.

---

## 4. Carregamento de dados e desempenho

### `batch_size` = `2`

Imagens por passo de otimização. As imagens do HRSID têm 800×800 px e entram no modelo nessa escala, então o consumo de memória é alto: 2 é o valor seguro para a GPU T4 do Colab com `amp=True`. Se aumentar o batch, considere aumentar `learning_rate` na mesma proporção.

### `num_workers` = `2`

Processos paralelos de leitura. No Colab, 2 costuma ser o melhor equilíbrio. Use `0` no Windows ou quando aparecerem erros de processo filho no Jupyter.

### `amp` = `True`

Precisão mista na GPU: parte das operações roda em 16 bits, com `GradScaler` protegendo os gradientes pequenos. Reduz memória e tempo sem mudar imagens nem arquitetura. Vale só para CUDA; em CPU e MPS fica desligado sozinho. Na avaliação, as predições voltam para 32 bits antes das métricas. Use `False` se aparecer "Perda não finita".

### `horizontal_flip` = `0.5`

Probabilidade de espelhar a imagem na horizontal, junto com caixas e máscaras. É a única aumentação: imagens SAR vistas de cima não têm "lado certo", então o espelhamento é seguro. Use `0.0` para desligar.

### `device` = `'auto'`

Com `auto`, escolhe CUDA, depois MPS (Apple Silicon), depois CPU. Aceita `'cuda'`, `'mps'` ou `'cpu'` para forçar. No MPS, alguma operação do Mask R-CNN pode não ter implementação Metal; nesse caso, exporte `PYTORCH_ENABLE_MPS_FALLBACK=1` antes de abrir o Jupyter ou use `'cpu'`.

---

## 5. Otimizador

O otimizador é o **AdamW**, aplicado apenas aos parâmetros com `requires_grad`. O agendamento é `CosineAnnealingLR`, que leva a taxa de aprendizado a zero ao longo de `epochs`, sem degraus.

### `learning_rate` = `0.0001`

Taxa inicial. 1e-4 é a escala usual do AdamW em fine-tuning de detectores. Com SGD, a mesma taxa seria baixa demais — trocar de otimizador exige recalibrar esse valor.

### `weight_decay` = `0.0005`

Regularização L2. 5e-4 é o valor das receitas clássicas de R-CNN e reduz o sobreajuste, que importa mais em `full_finetune`.

---

## 6. Avaliação

### `confidence_threshold` = `0.5`

Confiança mínima para uma predição entrar em precisão, recall, F1 e nas figuras de predição. **Não afeta a mAP**, que percorre todas as faixas de confiança por conta própria.

### `iou_threshold` = `0.5`

Sobreposição mínima entre predição e alvo para a detecção contar como acerto em precisão, recall e F1. Vale tanto para a IoU de caixa quanto para a IoU de máscara; cada alvo é casado no máximo uma vez, do maior para o menor score.

### `mask_threshold` = `0.5`

Limiar que transforma a máscara contínua prevista em máscara binária. Afeta a mAP de máscara, a IoU de máscara e as figuras. Valores menores engordam as máscaras; maiores as afinam.

O checkpoint guardado é o de maior **mAP@0.50:0.95 de máscara** na validação. As métricas registradas por época são: `box_map_50`, `box_map_50_95`, `mask_map_50`, `mask_map_50_95`, precisão, recall e F1 de caixa e de máscara, com TP, FP e FN.

---

## 7. Controle da execução

### `smoke_test` = `True`

Roda as checagens rápidas da seção 13 antes de treinar: caminhos, dataset, leitura COCO, formas dos tensores, conversão das caixas, DataLoader, criação do modelo, número de classes das cabeças, perda em um batch, saída de inferência, checkpoint salvo e recarregado e isolamento da pasta de resultados. O relatório vai para `reports/smoke_test.csv`. Roda em CPU e leva poucos minutos; desligue em rodadas seguintes no mesmo ambiente.

### `run_training` = `False`

Com `True`, a célula de execução cria uma rodada e treina. Vem desligado para que o notebook possa ser lido e validado sem iniciar um treino longo.

### `evaluate_test` = `False`

Com `True`, avalia um checkpoint no teste oficial e salva `metrics_test.json`. Use **uma única vez**, depois de escolher a estratégia pela validação.

### `checkpoint_path` = `None`

Checkpoint que `evaluate_test` e a seção 18 usam quando não há treino na mesma execução. Caminhos relativos partem da raiz do projeto. Carregue apenas checkpoints de origem confiável: o arquivo é aberto com `weights_only=False`.

---

## 8. Padrões do modelo que o notebook não altera

Vêm do `maskrcnn_resnet50_fpn_v2` do torchvision.

| Parâmetro | Padrão | Efeito |
| --- | --- | --- |
| `min_size` / `max_size` | 800 / 1333 | As imagens do HRSID já têm 800 px e entram sem reescala. |
| `image_mean` / `image_std` | médias do ImageNet | Normalização aplicada dentro do modelo; as imagens entram em `[0, 1]`. |
| Tamanhos das âncoras | 32, 64, 128, 256, 512 px | Um tamanho por nível da FPN. A maioria dos navios cai na menor âncora. |
| `box_score_thresh` | 0,05 | Confiança mínima das predições devolvidas. |
| `box_nms_thresh` | 0,5 | IoU de supressão de caixas duplicadas. |
| `box_detections_per_img` | 100 | Máximo de detecções por imagem. |
| Máscara da cabeça | 28×28 por instância | Redimensionada para a caixa prevista na saída. |

---

## Arquivos gerados por rodada

Cada rodada cria `runs/results/hrsid_mask_rcnn_v2/<experimento>/<data_hora>/`:

| Arquivo | Conteúdo |
| --- | --- |
| `config.json` | o `CONFIG` usado |
| `reports/split.csv` | divisão treino/validação/teste por imagem, com a cena de origem |
| `reports/parameters.csv` | parâmetros treináveis e congelados por parte do modelo |
| `reports/history.csv` | perdas e métricas de validação por época |
| `reports/metrics_validation.json`, `metrics_test.json` | métricas da melhor época e do teste |
| `checkpoints/best.pt` | pesos, otimizador, agendador, escala do AMP e `CONFIG` da melhor época |
| `figures/curve_<métrica>.png`, `training_curves.png` | uma curva por métrica e o painel |
| `predictions/prediction_<imagem>.png` | comparação entre anotação e predição |

Fora das rodadas, a pasta `runs/results/hrsid_mask_rcnn_v2/` guarda o que descreve o dataset: `reports/integrity_*.csv`, `reports/smoke_test.csv`, `reports/split.csv`, `reports/comparison.csv`, as figuras da análise exploratória e `figures/comparison.png`.
