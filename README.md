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

## Sincronização com GitHub e Colab

A sincronização funciona enquanto o monitor estiver ativo. Ela usa somente a branch `main`, respeita o `.gitignore` e não executa push forçado. Arquivos já rastreados que também passaram a ser ignorados precisam de revisão manual. Não coloque credenciais em arquivos do projeto: o Git não detecta todos os segredos.

### Configuração inicial

1. No GitHub, crie um token fine-grained para este repositório, com permissão **Contents: Read and write**. No Colab, em **Segredos**, salve-o como `GITHUB_TOKEN` e permita que `colab.ipynb` acesse o segredo. O token não entra na URL remota nem é salvo no repositório.
2. Após disponibilizar estes arquivos na `main` do GitHub, abra `colab.ipynb` no Colab, execute a célula inicial e autorize o Google Drive. O clone persistente será criado em `Meu Drive/vision-transfer-learning`. Nas sessões seguintes, abra os notebooks a partir dessa pasta do Drive e execute novamente a célula inicial.
3. No computador local, configure o acesso ao GitHub pelo SSH ou pelo gerenciador de credenciais do Git, além de seu nome e e-mail de commit no Git. Na raiz do projeto, inicie o monitor em um terminal e mantenha-o aberto:

   ```sh
   uv run --no-sync python sync.py
   ```

   Também é possível executar `python sync.py` com Python 3. Interrompa com `Ctrl+C`.

Antes de iniciar o monitor em qualquer ambiente, **salve e feche os editores que possam ter arquivos abertos nessa cópia**; depois da atualização inicial, reabra os arquivos. O Git não consegue identificar alterações ainda não salvas no VS Code ou no navegador. No Colab, executar uma célula não salva automaticamente edições pendentes do notebook: salve o notebook no Drive e confirme que a gravação terminou. O notebook aberto a partir do GitHub pode ser uma cópia diferente da pasta do Drive.

O monitor consulta o GitHub e faz um commit e push quando os arquivos salvos permanecem estáveis por pelo menos um minuto. No Colab, a célula opcional de publicação solicita uma verificação antecipada ao mesmo monitor. Alterações ignoradas, como `data/` e `runs/results/`, não são enviadas. Treinamentos e dados precisam ser armazenados separadamente no Drive quando sua persistência for necessária. A sessão Colab pode ser encerrada sem aviso; depois disso, o monitor para, e alterações ainda não enviadas permanecem somente no Drive até a próxima sessão.

Para proteger arquivos abertos, atualizações remotas são aplicadas apenas ao iniciar o monitor, com a cópia limpa e sem divergência. Se a `main` remota avançar enquanto o monitor estiver ativo, ele para: salve e feche os editores e reinicie-o. Se também houver alterações locais ou commits divergentes, concilie os trabalhos manualmente antes de reiniciar. Falhas de autenticação, conflitos e índice já preparado também interrompem a execução com uma mensagem; o monitor nunca limpa o índice ou apaga alterações.

Uma pasta oculta `.vision-transfer-learning.sync-lock` ao lado do repositório impede monitores concorrentes que compartilhem o mesmo sistema de arquivos. Após interrupção abrupta, o bloqueio pode permanecer: somente depois de verificar que nenhum monitor está ativo, remova essa pasta vazia e reinicie. O Google Drive montado não oferece garantias suficientes de bloqueio distribuído entre máquinas/sessões: use **uma única sessão Colab por pasta**. As cópias local e do Drive podem ter monitores próprios; se ambas produzirem commits concorrentes, o push recusado preserva o commit local para conciliação.

Se o próprio `colab.ipynb` tiver mudado no GitHub, a atualização no Colab também é interrompida: feche-o e atualize a cópia do Drive manualmente antes de reabrir, evitando que o navegador sobrescreva uma versão mais nova.
