# Transfer Learning em Detecção de Objetos

Projeto final de Visão Computacional. O estudo compara três estratégias de treinamento do mesmo detector e mede o ganho real da transferência de aprendizado:

| Experimento | Estratégia | Pesos pré-treinados |
| --- | --- | --- |
| A | Treino do zero | nenhum |
| B | Backbone congelado, só a cabeça de detecção aprende | sim |
| C | Fine-tuning completo | sim |

Cada rodada registra precisão, recall e F1 em curvas individuais, mAP no protocolo COCO e matriz de confusão.

Um segundo estudo, independente, leva a mesma pergunta para a segmentação de instâncias em imagens de radar: [HRSID com Mask R-CNN V2](#segundo-estudo-hrsid-com-mask-r-cnn-v2).

## Instalação

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/).

```sh
# CPU — qualquer sistema operacional
uv sync --extra cpu
```

```sh
# GPU — detecta o sistema operacional e instala os pacotes certos
uv run python scripts/setup_gpu.py
```

O script identifica o sistema antes de instalar, porque os pacotes de GPU não são os mesmos:

| Sistema | Pacotes | Verificação |
| --- | --- | --- |
| Linux, Windows | PyTorch com CUDA 12.8 (`--extra cu128`) | GPU NVIDIA disponível e operação em CUDA |
| macOS | PyTorch oficial do PyPI (`--extra cpu`) | aceleração Metal (MPS) em Apple Silicon |

macOS não tem GPU NVIDIA, então não recebe CUDA: usa as rodas do PyPI, que trazem o backend Metal. Se a verificação falhar — sem GPU, sem driver ou Mac Intel —, o script informa e você segue com `uv sync --extra cpu`.

Abra o JupyterLab pelo mesmo ambiente, com o extra que você instalou:

```sh
uv run --extra cpu jupyter lab      # ou --extra cu128
```

## Execução

Abra `notebooks/deep_pcb_transfer_learning.ipynb` e execute as células em ordem. O notebook escolhe CUDA, MPS ou CPU conforme a disponibilidade.

O treino vem desativado. Para iniciá-lo, ajuste `CONFIG['experiment']` e `CONFIG['run_training']`; para avaliar um checkpoint já salvo, use `evaluate_test` e `checkpoint_path`. Caminhos relativos de checkpoint partem da raiz do projeto no ambiente ativo.

Cada parâmetro do `CONFIG` e os padrões do modelo que influenciam o treino estão explicados em [docs/parametros_treinamento.md](docs/parametros_treinamento.md).

No Google Colab, abra o mesmo notebook, execute as células em ordem e autorize a montagem do Google Drive. O Colab usa os pacotes já presentes no runtime e instala somente os ausentes, sem sincronizar o ambiente `uv` local.

## Dados da rodada

O conjunto de imagens é um parâmetro em `CONFIG['dataset_dir']`, não o objeto do estudo. A rodada configurada usa o [Dataset](https://drive.google.com/drive/folders/1mdTe8Ez5r7KCi5ryaBf_onUCZJHOlPqH?usp=drive_link) com Faster R-CNN ResNet50 FPN V2.

Disponibilize a pasta já existente, com `trainval.txt`, `test.txt` e os grupos referenciados:

- Local: `data/DeepPCB/PCBData/` na raiz do projeto.
- Colab: `MyDrive/vision-transfer-learning/dataset/DeepPCB/PCBData/` no Google Drive.

O notebook não baixa nem duplica o conjunto de imagens.

## Resultados

Cada rodada cria uma pasta própria em `runs/results/` no projeto local ou em `MyDrive/vision-transfer-learning/runs/results/` no Colab, sem apagar as anteriores:

- `config.json`, `split.csv` e `history.csv` com os parâmetros, a divisão e as métricas por época.
- `figures/curve_precision.png`, `curve_recall.png`, `curve_f1.png`, `curve_map_50.png`, `curve_map_50_95.png` e `curve_train_loss.png` — uma figura por métrica.
- `figures/training_curves.png` com as mesmas curvas reunidas em um painel.
- `figures/confusion_validation.png` e `figures/confusion_test.png`, com as matrizes também em `.npy`.
- `checkpoints/best.pt`, o melhor checkpoint por mAP@0.50:0.95 na validação.

## Limites

O teste oficial compartilha grupos com os dados de desenvolvimento; seus resultados não medem generalização para grupos independentes. Precisão, recall e F1 são micro e dependem dos limiares de confiança e IoU em `CONFIG`. O mAP segue o protocolo COCO, enquanto o benchmark DeepPCB original usa IoU 0,33 — os números não são diretamente comparáveis.

## Segundo estudo: HRSID com Mask R-CNN V2

[notebooks/hrsid_mask_rcnn_transfer_learning.ipynb](notebooks/hrsid_mask_rcnn_transfer_learning.ipynb) é um experimento separado, que não interfere no DeepPCB: outro conjunto de imagens, outro modelo, outra pasta de resultados. A pergunta é quanto um modelo treinado em fotografias naturais aproveita em imagens de radar (SAR) e quanto custa segmentar cada navio, além de enquadrá-lo.

O modelo é o [`maskrcnn_resnet50_fpn_v2`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html) com os pesos `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1`. Só as cabeças `FastRCNNPredictor` e `MaskRCNNPredictor` são trocadas por versões de duas classes, fundo e navio; backbone, FPN, RPN e RoI Align continuam com os pesos do COCO.

### Dados

O [HRSID](https://github.com/chaozhong2010/HRSID) traz imagens SAR de 800×800 px anotadas em formato COCO, com caixa e máscara por instância e divisão oficial de treino e teste. O notebook lê a pasta já extraída — não baixa, não descompacta e não copia imagens — e confere o que encontrou em vez de confiar nos números publicados.

`CONFIG['dataset_dir']` aceita qualquer caminho: pasta local, disco externo ou Google Drive montado ou sincronizado. Vazio, vale o padrão do ambiente:

| Ambiente | Padrão |
| --- | --- |
| Local | `data/HRSID/` na raiz do projeto |
| Colab | `MyDrive/vision-transfer-learning/dataset/HRSID/` |

A estrutura interna é descoberta sozinha: tanto faz se a pasta contém `HRSID_JPG/annotations/` e `HRSID_JPG/JPEGImages/` ou já é a pasta das anotações.

### Execução

No Colab, abra o notebook, execute as células em ordem e autorize a montagem do Drive; o runtime já traz PyTorch e TorchVision, e só o que falta é instalado. Localmente, use o mesmo ambiente do DeepPCB (`uv sync --extra cpu` ou `uv run python scripts/setup_gpu.py`) — não há dependência nova. O dispositivo é escolhido entre CUDA, MPS e CPU.

Antes de treinar, `CONFIG['smoke_test']` roda uma verificação curta de ponta a ponta: caminhos, dataset, leitura COCO, tensores, DataLoader, modelo, perda, inferência e checkpoint.

### Estratégias

`CONFIG['experiment']` escolhe uma por vez, e `run_training` vem desligado:

| `experiment` | Treinado | Congelado |
| --- | --- | --- |
| `frozen_backbone` | RPN e cabeças de detecção e de máscara | ResNet-50 e FPN |
| `partial_finetune` | blocos finais do ResNet-50, FPN, RPN e cabeças | blocos iniciais do ResNet-50 |
| `full_finetune` | tudo | nada |

A validação sai do treino oficial separando **cenas inteiras**: as imagens são recortes de cenas SAR maiores, e recortes vizinhos são quase iguais. O teste oficial fica para a avaliação final — mas ele compartilha cenas com o treino, então mede acerto em recortes novos das mesmas cenas, não em cenas novas; o notebook imprime quantas cenas se repetem. Cada parâmetro está explicado em [docs/parametros_hrsid.md](docs/parametros_hrsid.md).

### Métricas e resultados

Detecção e segmentação são medidas separadamente sobre as mesmas predições: `box_map_50` e `box_map_50_95` para as caixas, `mask_map_50` e `mask_map_50_95` para as máscaras, mais precisão, recall e F1 de cada uma nos limiares do `CONFIG`. A mAP não depende do limiar de confiança; precisão, recall e F1 dependem, e por isso aparecem sempre com os limiares usados. O melhor checkpoint é o de maior mAP@0.50:0.95 de máscara na validação.

Os resultados ficam em `runs/results/hrsid_mask_rcnn_v2/`, separados dos do DeepPCB — no projeto quando a execução é local, no Drive quando é no Colab. `reports/` e `figures/` na raiz descrevem o dataset; `<experimento>/<data_hora>/` guarda cada rodada com configuração, histórico, curvas, checkpoint, métricas e figuras de predição.
