# Vision Transfer Learning

Projeto de visão computacional com PyTorch. O notebook começa com diagnóstico de CPU/GPU e acesso aos dados; os modelos e o treinamento ainda serão desenvolvidos.

## Sem GPU: Google Colab

1. Abra [transfer_learning.ipynb no Colab](https://colab.research.google.com/github/Gengo250/vision-transfer-learning/blob/main/transfer_learning.ipynb), depois que a versão desejada estiver no GitHub. Também é possível enviar o arquivo `.ipynb` local pelo menu **Arquivo → Fazer upload de notebook**.
2. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, escolha **T4 GPU**, se disponível, e conecte.
3. Execute a célula de diagnóstico e confirme `CUDA disponível: True` e `Operação CUDA concluída`. O Colab define a disponibilidade e o tipo da GPU.
4. Execute a célula do Drive, autorize o acesso e ajuste `DATA_DIR` para a pasta do seu dataset. O Drive precisa conter os dados previamente; a célula não baixa nem altera arquivos.
5. Desenvolva e execute as células de treinamento no notebook. Para guardar resultados além da sessão, escolha explicitamente um caminho no Drive, sem sobrescrever arquivos existentes.

O Colab fornece PyTorch e TorchVision no runtime; o notebook não os reinstala. As alterações na cópia do Colab não são sincronizadas automaticamente com este repositório: baixe o notebook atualizado se quiser trazê-las de volta. Também é possível usar a [extensão Google Colab para VS Code](https://github.com/googlecolab/colab-vscode/wiki/User-Guide): abra o notebook, escolha **Select Kernel → Colab → New Colab Server**, selecione T4 GPU e execute as células. O kernel remoto acessa o Drive, não os arquivos locais automaticamente.

## Com GPU NVIDIA: computador local

Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/) e um driver NVIDIA compatível com CUDA 12.8. Na pasta do projeto, execute:

```sh
uv run --no-sync python scripts/setup_gpu.py
```

O mesmo comando funciona no Linux e no Windows. O script instala as dependências na `.venv` com `uv`, seleciona o PyTorch CUDA 12.8 e testa uma operação na GPU. Depois abra `transfer_learning.ipynb` no VS Code, selecione o interpretador Python da `.venv`, coloque seu dataset em `data/` e execute o notebook. Ajuste `DATA_DIR` se estiver em outro local.

Para abrir o JupyterLab pelo terminal:

```sh
uv run --no-sync jupyter lab
```

`data/` e `runs/results/` são ignorados pelo Git. O instalador não modifica datasets, checkpoints nem resultados. Linux e Windows são suportados; a execução CUDA depende de GPU e driver compatíveis.
