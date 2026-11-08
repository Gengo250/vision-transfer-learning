# Transfer Learning em Detecção de Objetos

Projeto final de Visão Computacional. O estudo compara três estratégias de treinamento do mesmo detector e mede o ganho real da transferência de aprendizado:

| Experimento | Estratégia | Pesos pré-treinados |
| --- | --- | --- |
| A | Treino do zero | nenhum |
| B | Backbone congelado, só a cabeça de detecção aprende | sim |
| C | Fine-tuning completo | sim |

Cada rodada registra precisão, recall e F1 em curvas individuais, mAP no protocolo COCO e matriz de confusão.

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

No Google Colab, abra o mesmo notebook, execute as células em ordem e autorize a montagem do Google Drive. O Colab usa os pacotes já presentes no runtime e instala somente os ausentes, sem sincronizar o ambiente `uv` local.

## Dados da rodada

O conjunto de imagens é um parâmetro em `CONFIG['dataset_dir']`, não o objeto do estudo. A rodada configurada usa o [DeepPCB](https://drive.google.com/drive/folders/1mdTe8Ez5r7KCi5ryaBf_onUCZJHOlPqH?usp=drive_link) com Faster R-CNN ResNet50 FPN V2.

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
