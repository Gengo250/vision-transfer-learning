"""Instala as bibliotecas de GPU conforme o sistema operacional e valida a aceleração."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

CUDA_CHECK = "\n".join([
    "import torch",
    "print('PyTorch:', torch.__version__)",
    "print('CUDA do PyTorch:', torch.version.cuda)",
    "if not torch.cuda.is_available():",
    "    raise SystemExit('GPU CUDA indisponível. Verifique a GPU NVIDIA e o driver.')",
    "print('GPU:', torch.cuda.get_device_name(0))",
    "x = torch.ones((2, 2), device='cuda')",
    "print('Operação na GPU:', (x @ x).tolist())",
    "torch.cuda.synchronize()",
])

MPS_CHECK = "\n".join([
    "import torch",
    "print('PyTorch:', torch.__version__)",
    "if not torch.backends.mps.is_available():",
    "    raise SystemExit('Metal (MPS) indisponível. Requer macOS 12.3+ com Apple Silicon.')",
    "print('GPU: Apple Metal (MPS)')",
    "x = torch.ones((2, 2), device='mps')",
    "print('Operação na GPU:', (x @ x).tolist())",
    "torch.mps.synchronize()",
])

# O macOS não tem GPU NVIDIA: em vez de CUDA, usa as rodas oficiais do PyPI,
# que trazem a aceleração Metal (MPS) para chips Apple Silicon.
TARGETS = {
    "Linux": ("cu128", CUDA_CHECK, "PyTorch com CUDA 12.8 para GPU NVIDIA"),
    "Windows": ("cu128", CUDA_CHECK, "PyTorch com CUDA 12.8 para GPU NVIDIA"),
    "Darwin": ("cpu", MPS_CHECK, "PyTorch do PyPI com aceleração Metal (MPS)"),
}


def main() -> int:
    system = platform.system()
    target = TARGETS.get(system)
    if target is None:
        print(f"Sistema não suportado: {system}. Use 'uv sync --extra cpu'.", file=sys.stderr)
        return 1
    extra, check, description = target

    uv = shutil.which("uv")
    if uv is None:
        print("Instale o uv primeiro: https://docs.astral.sh/uv/getting-started/installation/", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent.parent
    print(f"{system} {platform.machine()}: instalando {description}...", flush=True)
    try:
        subprocess.run([uv, "sync", "--locked", "--extra", extra], cwd=root, check=True)
        subprocess.run([uv, "run", "--no-sync", "python", "-c", check], cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        print("Não foi possível concluir a instalação ou a verificação da GPU.", file=sys.stderr)
        print("Para seguir sem GPU: uv sync --extra cpu", file=sys.stderr)
        return exc.returncode or 1

    print(f"Ambiente pronto. Abra o notebook com: uv run --extra {extra} jupyter lab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
