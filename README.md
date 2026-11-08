# DeepPCB — Transfer Learning

Projeto de detecção de defeitos em placas usando Faster R-CNN ResNet50 FPN V2. O notebook compara treinamento do zero, backbone congelado e fine-tuning completo.

## Executar

Disponibilize o [dataset DeepPCB](https://drive.google.com/drive/folders/1mdTe8Ez5r7KCi5ryaBf_onUCZJHOlPqH?usp=drive_link) já existente no caminho correspondente ao ambiente, com `trainval.txt`, `test.txt` e os grupos dentro de `PCBData/`:

- Local: `data/DeepPCB/PCBData/` na raiz do projeto.
- Colab: `MyDrive/vision-transfer-learning/dataset/DeepPCB/PCBData/` no Google Drive.

O notebook não baixa nem duplica o dataset.

Com Python 3.12 e [uv](https://docs.astral.sh/uv/):

```sh
uv sync --extra cpu
uv run --extra cpu jupyter lab
```

Abra `notebooks/deep_pcb_transfer_learning.ipynb` e execute em ordem. Para NVIDIA/CUDA 12.8, troque `cpu` por `cu128`. O notebook seleciona CUDA, MPS ou CPU conforme a disponibilidade.

No Google Colab, abra o mesmo notebook e execute as células em ordem. Autorize a montagem do Google Drive quando solicitada. O notebook usa os pacotes já disponíveis no runtime e instala somente os pacotes necessários que estiverem ausentes, sem sincronizar o ambiente local `uv`.

O treino vem desativado. Para iniciá-lo, ajuste `CONFIG['experiment']` e `CONFIG['run_training']`; para testar um checkpoint selecionado, use `evaluate_test` e `checkpoint_path`. Cada rodada cria uma pasta em `runs/results/` no projeto local ou em `MyDrive/vision-transfer-learning/runs/results/` no Colab, com métricas, curvas de precisão, recall e F1, matriz de confusão e checkpoints. Caminhos relativos de checkpoint partem da raiz do projeto no ambiente ativo.

O teste oficial compartilha grupos com os dados de desenvolvimento; seus resultados não medem generalização para grupos independentes. Precisão, recall e F1 dependem dos limiares configurados de confiança e IoU.
