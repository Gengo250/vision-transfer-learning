# Vision Transfer Learning

Projeto inicial em Python 3.12, gerenciado com [uv](https://docs.astral.sh/uv/). Compatível com Linux e Windows.

Instale o uv e, na pasta do projeto, escolha **um** perfil:

```sh
uv sync --extra cpu
```

Para GPU NVIDIA com driver compatível com CUDA 12.8:

```sh
uv sync --extra cu128
```

Inicie o JupyterLab usando o mesmo perfil escolhido na instalação:

```sh
uv run --extra cpu jupyter lab
```

Ou, para CUDA:

```sh
uv run --extra cu128 jupyter lab
```

Abra `transfer_learning.ipynb` e execute a célula de diagnóstico. Os diretórios `data/train`, `data/val`, `data/test` e `runs/results` são locais e ignorados pelo Git; crie-os se não existirem após clonar o repositório.
