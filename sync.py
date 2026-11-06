"""Sincronização conservadora da branch main. Execute com Python 3."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

BRANCH = "main"
INTERVAL = 60
REPO = Path(__file__).resolve().parent
LOCK = REPO.parent / f".{REPO.name}.sync-lock"


class SyncError(Exception):
    pass


@contextmanager
def repository_lock():
    try:
        LOCK.mkdir()
    except FileExistsError as exc:
        raise SyncError(f"Sincronização já ativa ou bloqueio antigo: {LOCK}. "
                        "Se nenhum processo estiver ativo, remova a pasta de bloqueio vazia.") from exc
    try:
        yield
    finally:
        LOCK.rmdir()


@contextmanager
def authentication(token: str | None):
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if not token:
        yield env
        return
    # The token never appears in Git arguments, remote URLs or repository files.
    with tempfile.TemporaryDirectory(prefix="vision-sync-") as directory:
        helper = Path(directory) / "askpass"
        helper.write_text('#!/bin/sh\ncase "$1" in *sername*) echo x-access-token;; *) printf "%s\\n" "$VISION_SYNC_TOKEN";; esac\n', encoding="utf-8")
        helper.chmod(0o700)
        env.update(GIT_ASKPASS=str(helper), GIT_ASKPASS_REQUIRE="force",
                   VISION_SYNC_TOKEN=token, GIT_CONFIG_COUNT="1",
                   GIT_CONFIG_KEY_0="credential.helper", GIT_CONFIG_VALUE_0="")
        yield env


def git(*args: str, env: dict[str, str], check: bool = True, input: bytes | None = None):
    result = subprocess.run(["git", *args], cwd=REPO, env=env, input=input,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        token = env.get("VISION_SYNC_TOKEN")
        if token:
            detail = detail.replace(token, "[TOKEN OCULTO]")
        raise SyncError(f"Git falhou ({args[0]}): {detail or 'verifique autenticação e conexão.'}")
    return result


def output(*args: str, env: dict[str, str]) -> str:
    return git(*args, env=env).stdout.decode("utf-8", "replace").strip()


def check_repo(env: dict[str, str], colab: bool):
    if output("rev-parse", "--show-toplevel", env=env) != str(REPO):
        raise SyncError("Execute o script na raiz do repositório correto.")
    if output("branch", "--show-current", env=env) != BRANCH:
        raise SyncError("A sincronização exige a branch main ativa.")
    if colab:
        allowed = {"https://github.com/Gengo250/vision-transfer-learning.git",
                   "https://github.com/Gengo250/vision-transfer-learning"}
        fetch_urls = output("remote", "get-url", "--all", "origin", env=env).splitlines()
        push_urls = output("remote", "get-url", "--push", "--all", "origin", env=env).splitlines()
        if len(fetch_urls) != 1 or len(push_urls) != 1 or fetch_urls[0] not in allowed or push_urls[0] not in allowed:
            raise SyncError("As URLs de fetch e push de origin devem apontar apenas ao repositório HTTPS esperado; token não utilizado.")
    if git("diff", "--cached", "--quiet", env=env, check=False).returncode:
        raise SyncError("Há alterações já preparadas no índice. Preserve ou finalize-as manualmente antes de sincronizar.")
    if git("diff", "--name-only", "--diff-filter=U", env=env).stdout:
        raise SyncError("Há conflitos Git pendentes. Resolva-os antes de sincronizar.")
    tracked = git("ls-files", "-m", "-d", "-z", env=env).stdout
    if tracked:
        ignored = git("check-ignore", "--no-index", "--stdin", "-z", env=env,
                      check=False, input=tracked)
        if ignored.returncode not in (0, 1):
            raise SyncError("Não foi possível verificar arquivos rastreados ignorados.")
        if ignored.stdout:
            raise SyncError("Há arquivos rastreados e ignorados modificados; revise-os antes de sincronizar.")


def status(env: dict[str, str]) -> bytes:
    return git("status", "--porcelain=v1", "-z", "--untracked-files=all", env=env).stdout


def fetch(env: dict[str, str]):
    git("-c", "http.followRedirects=false", "fetch", "--no-tags", "origin",
        f"refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}", env=env)


def relation(env: dict[str, str]) -> tuple[int, int]:
    value = output("rev-list", "--left-right", "--count", f"HEAD...refs/remotes/origin/{BRANCH}", env=env)
    left, right = value.split()
    return int(left), int(right)


def initialize(env: dict[str, str], colab: bool):
    check_repo(env, colab)
    fetch(env)
    ahead, behind = relation(env)
    if behind:
        if ahead:
            raise SyncError("Há commits locais e remotos divergentes. Resolva a divergência manualmente.")
        if colab and git("diff", "--name-only", "-z", "HEAD", f"refs/remotes/origin/{BRANCH}",
                         "--", "colab.ipynb", env=env).stdout:
            raise SyncError("O colab.ipynb remoto mudou e pode estar aberto no navegador. "
                            "Feche o notebook e atualize a cópia do Drive manualmente antes de continuar.")
        try:
            git("merge", "--ff-only", "--no-overwrite-ignore", f"refs/remotes/origin/{BRANCH}", env=env)
        except SyncError as error:
            raise SyncError(f"Atualização não aplicada; confira alterações locais e possíveis conflitos. {error}") from error
        print("Atualizações do GitHub recebidas.", flush=True)
    else:
        print("Repositório conferido; nenhuma atualização remota pendente.", flush=True)


def notebook_without_volatile_outputs(data: bytes):
    """Compare notebooks without changing or discarding the saved original."""
    notebook = copy.deepcopy(json.loads(data))
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("notebook inválido")
    for cell in notebook["cells"]:
        if not isinstance(cell, dict):
            raise ValueError("célula inválida")
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
        if cell.get("cell_type") == "code":
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
        if isinstance(cell.get("metadata"), dict):
            cell["metadata"].pop("execution", None)
    return notebook


def output_only_notebooks(env: dict[str, str]) -> list[str]:
    paths = git("diff", "--name-only", "--diff-filter=M", "-z", "HEAD", "--", env=env).stdout.split(b"\0")
    excluded = []
    for raw in paths:
        if not raw or not raw.endswith(b".ipynb"):
            continue
        path = os.fsdecode(raw)
        try:
            old = git("show", f"HEAD:{path}", env=env).stdout
            current = (REPO / path).read_bytes()
            if notebook_without_volatile_outputs(old) == notebook_without_volatile_outputs(current):
                excluded.append(path)
        except (OSError, ValueError, TypeError, SyncError):
            # Invalid or unparseable notebooks are never silently excluded.
            continue
    return excluded


def sync_once(env: dict[str, str], colab: bool):
    check_repo(env, colab)
    fetch(env)
    ahead, behind = relation(env)
    if behind:
        raise SyncError("A main remota avançou durante o trabalho. Salve e feche os editores; "
                        "reinicie a sincronização para atualizar. Se houver alterações locais, resolva a divergência manualmente.")
    if status(env):
        excluded = output_only_notebooks(env)
        git("add", "--all", "--", ".", *(f":(exclude,literal){path}" for path in excluded), env=env)
        if excluded:
            print("Outputs e contagens permanecem no notebook local, sem commit automático: "
                  + ", ".join(excluded), flush=True)
        if git("diff", "--cached", "--quiet", env=env, check=False).returncode:
            args = ["commit", "-m", "Sincronização automática"]
            if colab:
                args = ["-c", "user.name=Colab Sync", "-c", "user.email=Gengo250@users.noreply.github.com", *args]
            git(*args, env=env)
            ahead += 1
            print("Alterações salvas em um commit local.", flush=True)
    if ahead:
        # Inspect every outgoing commit, including files added and later removed.
        commits = output("rev-list", f"refs/remotes/origin/{BRANCH}..HEAD", env=env).splitlines()
        changed = b"".join(git("diff-tree", "--root", "--no-commit-id", "--name-only",
                               "--no-renames", "-r", "-m", "-z", commit, env=env).stdout
                           for commit in commits)
        if changed:
            ignored = git("check-ignore", "--no-index", "--stdin", "-z", env=env,
                          check=False, input=changed)
            if ignored.returncode not in (0, 1):
                raise SyncError("Não foi possível verificar os arquivos dos commits locais.")
            if ignored.stdout:
                raise SyncError("Commits locais incluem arquivos atualmente ignorados; revise-os antes do push.")
        git("-c", "http.followRedirects=false", "push", "origin", f"HEAD:refs/heads/{BRANCH}", env=env)
        print("Alterações enviadas ao GitHub.", flush=True)


def watch(token: str | None = None, colab: bool = False, interval: int = INTERVAL, ready=None, wake=None):
    with repository_lock(), authentication(token) as env:
        initialize(env, colab)
        print(f"Monitor ativo: verifica alterações estáveis a cada {interval} segundos. "
              "Salve os arquivos antes de sincronizar.", flush=True)
        if ready is not None:
            ready.set()
        while True:
            if wake is None:
                time.sleep(interval)
            else:
                wake.wait(interval)
                wake.clear()
            current = status(env)
            # Status alone does not detect edits to an already modified file. Use Git's
            # content hashes below to require the exact snapshot to remain unchanged.
            snapshot = fingerprint(env)
            time.sleep(interval)
            if current != status(env) or snapshot != fingerprint(env):
                print("Arquivos ainda mudando; aguardando estabilidade.", flush=True)
                continue
            sync_once(env, colab)


def fingerprint(env: dict[str, str]) -> bytes:
    # Include content of all tracked/untracked non-ignored files without modifying the index.
    import hashlib
    digest = hashlib.sha256()
    paths = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", env=env).stdout.split(b"\0")
    for raw in sorted(set(paths)):
        if not raw:
            continue
        path = REPO / os.fsdecode(raw)
        digest.update(raw + b"\0")
        try:
            if path.is_symlink():
                digest.update(os.fsencode(os.readlink(path)))
            elif path.is_file():
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        except FileNotFoundError:
            digest.update(b"<removido>")
    return digest.digest()


def main():
    parser = argparse.ArgumentParser(description="Monitora e sincroniza a branch main.")
    parser.add_argument("--colab", action="store_true", help="Usa o segredo do Colab recebido no ambiente")
    args = parser.parse_args()
    token = os.environ.pop("VISION_COLAB_TOKEN", None) if args.colab else None
    if args.colab and not token:
        raise SyncError("Segredo GITHUB_TOKEN ausente ou sem acesso ao notebook no Colab.")
    watch(token, args.colab)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor encerrado.")
    except (SyncError, OSError) as error:
        print(f"Sincronização interrompida: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
