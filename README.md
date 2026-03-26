# Setup

```bash
# python, python3, py, py3
$ python3 -m venv .venv
$ .venv\Scripts\activate
$ python3 -m pip install -r requirements.txt

# Install missing only
# (If specific torch version (or other lib) is already installed)
$ pip install -r requirements.txt --ignore-installed --no-warn-conflicts
```

## CUDA shenanigans
```bash
# Adjust url according to version
$ %pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# If CPU version has been installed previously, run with --upgrade --force-reinstall
$ %pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --upgrade --force-reinstall
```