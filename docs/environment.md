# Ambiente e execução

O projeto usa Python `>=3.12,<3.13`, `uv` e os pacotes definidos em [pyproject.toml](../pyproject.toml) e [uv.lock](../uv.lock). PyTorch e TorchVision são instalados por um dos extras mutuamente exclusivos `cpu` ou `cu128`. Outras dependências incluem JupyterLab, ipykernel, NumPy, pandas, Pillow, Matplotlib, scikit-learn, tqdm, TorchMetrics e pycocotools.

## Execução local

Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/) e, na raiz do projeto, sincronize o ambiente e abra o JupyterLab:

```sh
uv sync --extra cpu
uv run --extra cpu jupyter lab
```

Para GPU local, o script a seguir instala e verifica o ambiente adequado:

```sh
uv run python scripts/setup_gpu.py
```

Em Windows e Linux, o script usa o extra `cu128` e testa CUDA com uma GPU NVIDIA e driver compatível. No macOS, usa os pacotes oficiais do PyPI e testa MPS em Apple Silicon. Abra o Jupyter com o extra indicado pelo script, por exemplo `uv run --extra cu128 jupyter lab` no ambiente CUDA. Se a verificação falhar, é possível usar o ambiente CPU. A instalação não garante compatibilidade com toda GPU ou driver.

Os notebooks procuram a raiz por `pyproject.toml` e recomendam o kernel de `.venv`. O notebook [transfer_learning.ipynb](../transfer_learning.ipynb) faz diagnóstico de ambiente e CUDA; ele não é um experimento de treinamento e seu diagnóstico de dispositivo não cobre MPS.

## Google Colab

Abra o notebook do experimento no Colab, selecione um runtime com GPU quando disponível, execute as células em ordem e autorize a montagem do Google Drive. Os notebooks usam os pacotes do runtime e instalam apenas os módulos ausentes; não sincronizam o ambiente local com `uv` no Colab. As versões do runtime podem diferir do `uv.lock`, portanto registre-as ao comparar rodadas.

No Colab, a raiz de dados e resultados é `MyDrive/vision-transfer-learning/`. Os caminhos de dados, inclusive o requisito de extrair o HRSID, estão nos documentos [DeepPCB](parametros_treinamento.md) e [HRSID](parametros_hrsid.md).

## Dispositivo e execução

Os dois notebooks de experimento selecionam CUDA, depois MPS e por fim CPU; o HRSID também aceita `CONFIG['device']` explícito. O AMP é ativado apenas em CUDA. Em MPS, operações não implementadas podem exigir `PYTORCH_ENABLE_MPS_FALLBACK=1` antes de abrir o Jupyter ou execução em CPU.

Revise o `CONFIG` e o kernel antes de executar todas as células: **ambos os experimentos estão com `run_training=True`**. Altere para `False` para evitar treinamento; a análise dos dados e a construção do modelo ainda podem consumir recursos. Pesos pré-treinados podem ser obtidos pelo TorchVision na primeira construção do modelo, mesmo sem treino. Altere `batch_size` ou `num_workers` antes de recriar os DataLoaders; em Windows, `num_workers=0` pode ajudar com problemas de processos no Jupyter.

[Índice](README.md).
