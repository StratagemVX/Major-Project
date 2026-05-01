# PRISM-A Predictive Risk Modeling Framework

## Setup (Windows)

1) Install Python 3.12.
2) Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

3) Activate the venv:

```powershell
.venv\Scripts\Activate.ps1
```

4) Install CUDA-enabled PyTorch (cu121):

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Run

```powershell
python main.py
```

## Notes

- `main.py` calls `torch.cuda.get_device_name()`. This requires an NVIDIA GPU with drivers installed.
- If you do not have an NVIDIA GPU/driver, the program will error. Update the code to handle CPU-only execution if needed.
