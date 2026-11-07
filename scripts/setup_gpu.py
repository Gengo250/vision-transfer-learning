"""Instala o ambiente local CUDA e verifica o funcionamento da GPU NVIDIA."""

from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        print("Instale o uv primeiro: https://docs.astral.sh/uv/getting-started/installation/", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent.parent
    print("Instalando dependências do projeto com PyTorch CUDA 12.8...", flush=True)
    try:
        subprocess.run([uv, "sync", "--locked", "--extra", "cu128"], cwd=root, check=True)
        subprocess.run(
            [uv, "run", "--no-sync", "python", "-c", "\n".join([
                "import torch",
                "print('PyTorch:', torch.__version__)",
                "print('CUDA do PyTorch:', torch.version.cuda)",
                "if not torch.cuda.is_available():",
                "    raise SystemExit('GPU CUDA indisponível. Verifique a GPU NVIDIA e o driver.')",
                "print('GPU:', torch.cuda.get_device_name(0))",
                "x = torch.ones((2, 2), device='cuda')",
                "print('Operação CUDA:', x @ x)",
                "torch.cuda.synchronize()",
            ])],
            cwd=root,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print("Não foi possível concluir a instalação ou a verificação da GPU.", file=sys.stderr)
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
