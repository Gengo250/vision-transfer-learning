# Diagnóstico e recuperação da cópia do Colab

A pasta do Drive pode estar em um commit anterior que ainda não contém `sync.py`. Abra **um notebook Colab novo e vazio**, feche os notebooks abertos da pasta do projeto e copie as células abaixo para ele. A primeira célula apenas lê o estado existente; não acessa segredos nem envia alterações. Salve os notebooks no Drive antes de começar: o Git não vê alterações ainda não salvas no navegador.

## 1. Diagnóstico

```python
from google.colab import drive
from pathlib import Path
import importlib.util
import json
import os
import subprocess

try:
    drive.mount('/content/drive')
except Exception as exc:
    raise RuntimeError('Não foi possível montar o Drive; nenhuma alteração foi feita.') from exc

repo = Path('/content/drive/MyDrive/vision-transfer-learning')
if not (repo / '.git').exists():
    raise RuntimeError(f'Não há clone Git em {repo}; revise a pasta existente sem apagá-la.')

env = os.environ.copy()
env['GIT_TERMINAL_PROMPT'] = '0'

def ler_git(*args):
    return subprocess.run(['git', *args], cwd=repo, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print('Branch:', ler_git('branch', '--show-current').stdout.strip() or '(não identificada)')
print('Commit local:', ler_git('rev-parse', '--short', 'HEAD').stdout.strip() or '(erro)')
print('Estado local:')
print(ler_git('status', '--short', '--branch').stdout.strip() or '(sem saída)')
print('Conflitos:', ler_git('diff', '--name-only', '--diff-filter=U').stdout.strip() or '(nenhum)')
for nome in ('sync.py', 'tests', 'README.md', 'colab.ipynb', 'transfer_learning.ipynb'):
    print(f'{nome}:', 'presente' if (repo / nome).exists() else 'ausente')

for nome in ('transfer_learning.ipynb', 'colab.ipynb'):
    anterior = ler_git('show', f'HEAD:{nome}')
    if anterior.returncode or not (repo / nome).is_file():
        print(f'{nome}: comparação indisponível')
        continue
    try:
        local = json.loads((repo / nome).read_text(encoding='utf-8'))
        base = json.loads(anterior.stdout)
        lc, bc = local['cells'], base['cells']
        fontes = lambda cells: [(c.get('cell_type'), ''.join(c.get('source', []))
                                  if isinstance(c.get('source'), list) else c.get('source', ''))
                                 for c in cells]
        metadados = lambda nb: (nb.get('metadata'), [c.get('metadata') for c in nb['cells']])
        saidas = lambda cells: [(c.get('execution_count'), c.get('outputs', []))
                                 for c in cells if c.get('cell_type') == 'code']
        print(f'{nome}: células local/HEAD = {len(lc)}/{len(bc)}; '
              f'código ou texto mudou = {fontes(lc) != fontes(bc)}; '
              f'outputs ou contagens mudaram = {saidas(lc) != saidas(bc)}; '
              f'metadados mudaram = {metadados(local) != metadados(base)}')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'{nome}: não foi possível comparar JSON ({type(exc).__name__}); arquivo preservado.')

print('Pacotes disponíveis neste runtime:')
for nome in ('torch', 'torchvision', 'numpy', 'pandas', 'matplotlib', 'sklearn', 'PIL', 'jupyterlab', 'ipykernel'):
    print(f'  {nome}:', 'sim' if importlib.util.find_spec(nome) else 'NÃO')
try:
    import torch
    print('CUDA disponível:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except ImportError:
    print('PyTorch ausente neste runtime.')
```

O resultado compara os notebooks do Drive com o **commit local do próprio Drive**, não com a versão do GitHub. Ele identifica diferenças sem imprimir código nem outputs. Se o código mudou ou houver conflitos, mantenha os arquivos e concilie as versões antes de prosseguir. O Drive pode ter alterações que ainda não foram salvas pelo navegador.

## 2. Atualização segura

Depois de examinar o diagnóstico, com todos os notebooks da pasta salvos e fechados, execute esta célula **no notebook vazio separado**. Ela baixa a `main` pública e aplica somente um avanço rápido que o Git considere seguro. Se houver alterações locais no mesmo arquivo atualizado pelo GitHub, o Git interromperá a operação sem sobrescrevê-las. Ela também protege arquivos ignorados e recusa índices já preparados e operações Git concorrentes.

```python
from pathlib import Path
import os
import subprocess

repo = Path('/content/drive/MyDrive/vision-transfer-learning')
lock = repo.parent / '.vision-transfer-learning.sync-lock'
env = os.environ.copy()
env['GIT_TERMINAL_PROMPT'] = '0'

def git(*args):
    p = subprocess.run(['git', *args], cwd=repo, env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        # Git pode incluir URLs ou detalhes de autenticação em erros; não os imprima.
        raise RuntimeError(f'Git não concluiu {args[0]}; verifique conflitos, conexão e autenticação.')
    return p.stdout.strip()

if not (repo / '.git').exists():
    raise RuntimeError('Clone Git não encontrado; nenhuma pasta foi apagada.')
if git('branch', '--show-current') != 'main':
    raise RuntimeError('A branch ativa deve ser main.')
if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=repo, env=env).returncode:
    raise RuntimeError('Há alterações preparadas no índice; preserve-as e resolva antes de continuar.')
urls = git('remote', 'get-url', '--all', 'origin').splitlines()
if len(urls) != 1 or urls[0] not in (
    'https://github.com/Gengo250/vision-transfer-learning',
    'https://github.com/Gengo250/vision-transfer-learning.git',
):
    raise RuntimeError('Origin não é a URL HTTPS pública esperada; configuração preservada.')
try:
    lock.mkdir()
except FileExistsError as exc:
    raise RuntimeError(f'Outro monitor está ativo ou há bloqueio antigo: {lock}') from exc
try:
    git('-c', 'http.followRedirects=false', 'fetch', '--no-tags', 'origin',
        'refs/heads/main:refs/remotes/origin/main')
    print('Commits locais/remotos:', git('rev-list', '--left-right', '--count', 'HEAD...origin/main'))
    git('merge', '--ff-only', '--no-overwrite-ignore', 'origin/main')
    print('Atualização aplicada. Estado restante:')
    print(git('status', '--short', '--branch'))
finally:
    lock.rmdir()
```

Se a célula falhar por conflito, não limpe outputs nem use `reset`: compartilhe o resultado do diagnóstico e preserve os notebooks. A sessão precisa continuar aberta até a atualização terminar. Após atualizar, abra `colab.ipynb` a partir da pasta do Drive e inicie o monitor. O token `GITHUB_TOKEN` continua necessário para **enviar** alterações.

## 3. Dependências de diagnóstico no Colab

O Colab fornece sua própria interface Jupyter e não necessariamente instala o pacote `jupyterlab`. O notebook de treinamento atual importa esse pacote. Se a célula 1 indicar que falta, execute a célula abaixo para instalá-lo **somente no runtime atual**; a instalação não altera `pyproject.toml` nem `uv.lock`. Após reconectar, pode ser preciso executá-la novamente.

```python
import importlib.util
import subprocess
import sys

if importlib.util.find_spec('jupyterlab') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'jupyterlab'], check=True)
else:
    print('jupyterlab já está disponível.')
```
