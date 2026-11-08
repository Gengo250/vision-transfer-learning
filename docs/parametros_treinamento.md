# Parâmetros de treinamento deep pcb 

Este documento explica cada parâmetro usado em `notebooks/deep_pcb_transfer_learning.ipynb`: o que ele controla, o valor atual e o efeito de mudá-lo.

Os parâmetros aparecem em dois lugares:

1. **`CONFIG`**: o dicionário da seção 1 do notebook, que você ajusta a cada rodada.
2. **Padrões do modelo**: valores do torchvision que o notebook não altera, mas que também definem o treinamento.

Depois de mudar o `CONFIG`, reexecute a célula dele. `batch_size` e `num_workers` só passam a valer depois que você recriar os DataLoaders (seção 4).

> **Regra para comparar experimentos:** A, B e C devem usar os mesmos valores em tudo, exceto `experiment`. Se outro parâmetro mudar entre eles, a diferença de resultado deixa de medir só o efeito da transferência de aprendizado. Cada rodada salva o `CONFIG` usado em `runs/results/<rodada>/config.json`.

---

## 1. Reprodutibilidade e dados

### `seed` = `42`

Semente dos geradores aleatórios do Python, do NumPy e do PyTorch (CPU e CUDA). Ela controla:

- a inicialização dos pesos novos: a cabeça de classificação nos três experimentos e a rede inteira no experimento A;
- a ordem em que as imagens de treino são embaralhadas a cada época;
- o desempate na escolha dos grupos de validação.

Com a mesma semente, os resultados ficam muito próximos entre rodadas. Na GPU, algumas operações não são determinísticas, então pequenas diferenças ainda podem aparecer. Para medir a variação natural do modelo, rode o mesmo experimento com sementes diferentes (por exemplo 42, 43 e 44) e compare a média.

### `dataset_dir` = `'DeepPCB/PCBData'`

Pasta do conjunto de imagens. O caminho é relativo a `data/`, no projeto local, ou a `MyDrive/vision-transfer-learning/dataset/`, no Colab. A pasta precisa conter `trainval.txt`, `test.txt` e os grupos que esses arquivos citam.

O DeepPCB tem 1.500 imagens em tons de cinza de 640×640 px e 10.013 defeitos anotados em 6 classes: `open`, `short`, `mousebite`, `spur`, `copper` e `pin-hole`. O notebook divide as imagens assim:

| Split | Imagens | Origem |
| --- | --- | --- |
| train | 850 | `trainval.txt`, menos os grupos de validação |
| validation | 150 | grupos inteiros de `trainval.txt`, escolhidos para somar cerca de 15% das imagens |
| test | 500 | `test.txt` oficial |

A validação usa grupos que não aparecem no treino. O teste oficial compartilha grupos com o treino, então o resultado de teste não mede generalização para placas novas.

---

## 2. Estratégia de transferência

### `experiment` = `'C'`

Escolhe qual das três estratégias o estudo vai treinar. Todas usam o mesmo detector: Faster R-CNN ResNet50 FPN V2.

| Valor | Pesos iniciais | O que é treinado | Tempo por época |
| --- | --- | --- | --- |
| `'A'` | aleatórios (sem COCO nem ImageNet) | a rede inteira | maior |
| `'B'` | COCO | só RPN e cabeça de detecção; backbone e FPN ficam congelados | menor |
| `'C'` | COCO | a rede inteira (fine-tuning completo) | maior |

- **A** é a referência sem transferência. Costuma precisar de bem mais épocas para convergir, e com 20 épocas tende a ficar abaixo de B e C.
- **B** mede quanto as características aprendidas no COCO servem para PCBs sem nenhum ajuste. No modo B, o backbone fica em modo de avaliação durante o treino, então as estatísticas de normalização também ficam congeladas.
- **C** adapta todas as camadas ao domínio das placas e costuma dar o melhor resultado.

Nos três casos, a última camada de classificação é trocada por uma nova com 7 saídas: as 6 classes de defeito mais o fundo.

---

## 3. Duração do treinamento

### `epochs` = `20`

Número máximo de épocas. Uma época passa uma vez por todas as 850 imagens de treino e depois avalia as 150 de validação. O treino pode parar antes por `patience`.

### `patience` = `5`

Parada antecipada (*early stopping*). Se o mAP@0.50:0.95 de validação não melhorar por 5 épocas seguidas, o treino para. Só a melhor época é salva, em `checkpoints/best.pt`.

- Valores menores economizam tempo, mas podem parar o treino durante uma estagnação passageira.
- Ela interage com `lr_step_size`: depois que a taxa de aprendizado cai, o mAP costuma voltar a subir. Se `patience` for menor que o intervalo entre as quedas, o treino pode parar antes da queda acontecer.

---

## 4. Carregamento de dados e desempenho

### `batch_size` = `2`

Número de imagens processadas por passo de otimização.

- Na RTX 3050 Laptop (4 GB), batch 2 a 800 px usa cerca de 2,7 GB em FP32 e 1,7 GB com AMP.
- Batch maior não deixa o treino mais rápido nessa GPU, porque ela já fica saturada com 2 imagens. Batch maior também usa mais memória.
- Se mudar o batch, considere ajustar `learning_rate` na mesma proporção (por exemplo, batch 4 com lr 0,002).

### `num_workers` = `0`

Número de processos paralelos que leem e decodificam as imagens. Com `0`, a leitura acontece no processo principal.

Ler uma imagem do DeepPCB leva cerca de 3,5 ms, enquanto um passo de treino leva centenas de milissegundos. Por isso, aumentar esse valor quase não muda o tempo total. Valores acima de 0 também podem dar problema no Windows e em alguns notebooks Jupyter.

### `amp` = `True`

Precisão mista automática (*Automatic Mixed Precision*). Parte dos cálculos da GPU roda em 16 bits em vez de 32. Um `GradScaler` multiplica a perda antes do backward para que gradientes pequenos não virem zero, e o fator dessa escala vai junto no checkpoint.

- **Não muda** as imagens, a resolução, a arquitetura nem os hiperparâmetros.
- Medido nesta máquina: o passo de treino cai de cerca de 725 ms para 364 ms (por volta de 2× mais rápido), e o pico de memória cai de 2,7 GB para 1,7 GB.
- Na avaliação, as predições voltam para 32 bits antes do cálculo das métricas.
- Só vale para CUDA. Em CPU ou MPS (Apple), fica desligado automaticamente.
- Nas primeiras iterações, o scaler pode pular alguns passos enquanto ajusta a escala. Isso é esperado.

Use `False` para treinar em FP32, por exemplo se aparecer o erro "Perda não finita".

---

## 5. Otimizador (SGD)

O otimizador é o SGD com momento. Ele só atualiza os parâmetros que estão sendo treinados: no experimento B, os pesos congelados ficam de fora.

### `learning_rate` = `0.001`

Taxa de aprendizado inicial: o tamanho de cada passo de atualização dos pesos.

- Alta demais: a perda oscila ou diverge e pode virar `NaN` ("Perda não finita").
- Baixa demais: o treino converge devagar, o que pesa principalmente no experimento A.
- 0,001 é um valor conservador para fine-tuning de Faster R-CNN com batch 2. As referências do torchvision usam 0,02 com batch 16, ou seja, cerca de 0,0025 para batch 2.

### `momentum` = `0.9`

Média móvel dos gradientes anteriores. Suaviza as atualizações e acelera o avanço em direções consistentes. 0,9 é o valor padrão em detecção e raramente precisa mudar.

### `weight_decay` = `0.0005`

Regularização L2: puxa os pesos levemente em direção a zero a cada passo, o que reduz o sobreajuste (*overfitting*). 5e-4 é o valor usado nas receitas originais do Faster R-CNN.

---

## 6. Agenda da taxa de aprendizado (StepLR)

A taxa é multiplicada por `lr_gamma` a cada `lr_step_size` épocas. Com os valores atuais:

| Épocas | Taxa de aprendizado |
| --- | --- |
| 1–7 | 0,001 |
| 8–14 | 0,0001 |
| 15–20 | 0,00001 |

### `lr_step_size` = `7`

Intervalo, em épocas, entre as reduções da taxa.

### `lr_gamma` = `0.1`

Fator da redução. `0.1` divide a taxa por 10 a cada etapa.

Taxas menores no fim do treino refinam os pesos e geralmente aumentam o mAP logo depois de cada redução.

---

## 7. Avaliação

### `score_threshold` = `0.5`

Confiança mínima para uma predição contar em precisão, recall, F1 e na matriz de confusão. Predições abaixo desse valor são descartadas nessas métricas.

- Valor maior: menos falsos positivos, mais defeitos perdidos (precisão sobe, recall cai).
- Valor menor: o contrário.
- **Não afeta o mAP.** O mAP percorre todas as faixas de confiança por conta própria.

### `iou_threshold` = `0.5`

Sobreposição mínima (IoU) entre a caixa prevista e a caixa real para a detecção contar como acerto. A classe também precisa estar certa. Vale para precisão, recall, F1 e a matriz de confusão.

O mAP usa os limiares do protocolo COCO: **mAP@0.50**, com IoU 0,5, e **mAP@0.50:0.95**, a média de IoU de 0,50 a 0,95. O benchmark original do DeepPCB usa IoU 0,33, então os números não são diretamente comparáveis com os do artigo.

O melhor checkpoint é escolhido pelo **mAP@0.50:0.95 de validação**.

---

## 8. Controle da execução

### `run_training` = `True`

Com `True`, a última célula cria uma rodada em `runs/results/` e treina. Com `False`, nenhum treino acontece.

### `evaluate_test` = `False`

Com `True`, avalia um checkpoint no conjunto de teste oficial e salva `test_metrics.json` e a matriz de confusão de teste. Use só depois de escolher o experimento pela validação, para que o teste não influencie as decisões.

### `checkpoint_path` = `None`

Caminho do checkpoint que `evaluate_test` vai avaliar quando `run_training` for `False`. Caminhos relativos partem da raiz do projeto. Se o treino rodar na mesma execução, o melhor checkpoint dele é usado automaticamente.

Carregue apenas checkpoints de origem confiável: o arquivo é aberto com `weights_only=False`.

---

## 9. Padrões do modelo que o notebook não altera

Estes valores vêm do `fasterrcnn_resnet50_fpn_v2` do torchvision. Não estão no `CONFIG`, mas influenciam o treino e o tempo.

| Parâmetro | Padrão | Efeito |
| --- | --- | --- |
| `min_size` / `max_size` | 800 / 1333 | As imagens de 640 px são **ampliadas para 800 px** antes de entrar na rede. |
| `image_mean` / `image_std` | médias do ImageNet | Normalização aplicada dentro do modelo. As imagens entram em `[0, 1]`. |
| Tamanhos das âncoras | 32, 64, 128, 256, 512 px | Um tamanho por nível da FPN. |
| `box_score_thresh` | 0,05 | Confiança mínima das predições que o modelo devolve. |
| `box_nms_thresh` | 0,5 | IoU de supressão de caixas duplicadas. |
| `box_detections_per_img` | 100 | Máximo de detecções por imagem. |

### Sobre a resolução (800 px)

A ampliação de 640 para 800 px não acrescenta detalhes à imagem, mas aumenta os defeitos em pixels. Nas anotações do DeepPCB, a mediana dos defeitos tem cerca de 34 px a 640 e passa para cerca de 43 px a 800. A menor âncora tem 32 px, e os pesos COCO foram treinados nessa escala de 800.

Treinar a 640 px (`min_size=640, max_size=640` em `make_model`) seria cerca de 40% mais rápido, mas pode reduzir a precisão nos defeitos menores. Se testar, use a mesma resolução nos três experimentos.

---

## Arquivos gerados por rodada

Cada rodada cria `runs/results/<data_hora>_<experimento>/` com:

| Arquivo | Conteúdo |
| --- | --- |
| `config.json` | o `CONFIG` usado |
| `split.csv` | a divisão treino/validação/teste por imagem |
| `history.csv` | perda e métricas de validação por época |
| `checkpoints/best.pt` | pesos, otimizador, scheduler e scaler da melhor época |
| `figures/` | curvas por métrica, painel e matrizes de confusão |
| `confusion_validation.npy` | matriz de confusão da melhor época |
| `test_metrics.json`, `confusion_test.npy` | resultado no teste, quando avaliado |
