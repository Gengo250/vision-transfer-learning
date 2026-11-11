# Transfer Learning em Visão Computacional

Projeto acadêmico com dois experimentos independentes de detecção e segmentação usando PyTorch e TorchVision:

- **DeepPCB:** detecção de seis tipos de defeitos em placas de circuito com Faster R-CNN ResNet-50 FPN V2. Pergunta: *quanto o pré-treino no COCO ajuda?* Compara treino do zero (A), backbone congelado (B) e fine-tuning completo (C).
- **HRSID:** detecção e segmentação de instâncias de navios em imagens SAR com Mask R-CNN ResNet-50 FPN V2. Pergunta: *até que profundidade a rede pré-treinada precisa ser readaptada a um domínio de radar?* Compara backbone congelado, fine-tuning parcial (fronteira configurável) e fine-tuning completo, todos partindo do COCO.

Os dois estudos se complementam: o DeepPCB isola o valor do pré-treino em si, e o HRSID isola quanto da rede precisa mudar. O desenho de cada um (variáveis, hipóteses, comparações e limites) está na [documentação técnica](docs/README.md).

Os notebooks usam Python, JupyterLab, TorchMetrics, pycocotools, NumPy, pandas e Matplotlib.

## Estrutura

```text
├── README.md
├── docs/                      # índice e documentação técnica
├── notebooks/                 # experimentos DeepPCB e HRSID
├── data/                      # datasets locais, fora do controle de versão
├── runs/results/              # resultados locais das rodadas
├── scripts/setup_gpu.py       # configuração e diagnóstico de GPU local
├── transfer_learning.ipynb    # diagnóstico do ambiente
├── pyproject.toml
└── uv.lock
```

## Ambiente e execução

Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/) e use Python 3.12. Na raiz do projeto:

```sh
uv sync --extra cpu
uv run --extra cpu jupyter lab
```

Para GPU local, use `uv run python scripts/setup_gpu.py` e abra o Jupyter com o extra indicado pelo script. Veja [ambiente e GPU](docs/environment.md) para Windows, Linux, macOS e Colab.

Abra [DeepPCB](notebooks/deep_pcb_transfer_learning.ipynb) ou [HRSID](notebooks/hrsid_mask_rcnn_transfer_learning.ipynb), revise `CONFIG` e execute as células em ordem. **Os dois notebooks estão atualmente configurados com `run_training=True`**: altere para `False` antes de executar se não quiser iniciar treinamento. `evaluate_test` controla a avaliação no teste oficial.

## Dataset + modelos 

https://drive.google.com/drive/folders/1mdTe8Ez5r7KCi5ryaBf_onUCZJHOlPqH?usp=drive_link

No Colab, abra o notebook correspondente e autorize a montagem do Drive. Os notebooks leem os dados existentes e instalam somente dependências ausentes; não executam `uv sync` no runtime.

## Dados e resultados

Localmente, os caminhos padrão são `data/DeepPCB/PCBData/` e `data/HRSID/` ou `data/HRSID_jpg/`. No Colab, ficam em `MyDrive/vision-transfer-learning/dataset/DeepPCB/PCBData/` e `MyDrive/vision-transfer-learning/dataset/HRSID/` ou `HRSID_jpg/`. O HRSID precisa estar **extraído**: o notebook não lê nem extrai `HRSID_jpg.zip`. Ambos permitem configurar o caminho; consulte os detalhes de cada experimento antes de executar.

As rodadas locais são gravadas em `runs/results/`; no Colab, em `MyDrive/vision-transfer-learning/runs/results/`. O HRSID usa a subpasta `hrsid_mask_rcnn_v2/` e permite alterar o destino. Cada rodada guarda configuração, métricas, figuras e checkpoint, conforme a execução.

## Documentação

Acesse o [índice técnico](docs/README.md) para navegar pelo ambiente, pelos dados, modelos, estratégias e protocolos de avaliação de cada experimento.
