# DevContainer Setup Guide

This directory contains a **lightweight CPU-only** DevContainer configuration that works with both **VS Code** and **IntelliJ IDEA**.

## Features

- **Python 3.11** with CPU-only PyTorch
- **Lightweight base image** (~500 MB vs 5-8 GB for CUDA)
- **PyTorch** CPU version for development and small-scale experiments
- **PyTorch Geometric** for graph neural networks
- **Jupyter** notebooks
- **Data science libraries**: pandas, numpy, matplotlib, seaborn, plotly
- **Transformers** library for LLM work

## Prerequisites

### Common Requirements
- Docker Desktop (with WSL2 backend on Windows)

### For VS Code
- [Visual Studio Code](https://code.visualstudio.com/)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### For IntelliJ IDEA
- [IntelliJ IDEA 2022.3+](https://www.jetbrains.com/idea/) (Professional or Ultimate)
- [Dev Containers plugin](https://plugins.jetbrains.com/plugin/21962-dev-containers) (built-in since 2022.3)

## Setup Instructions

### Using VS Code

1. **Open the project in VS Code**
   ```bash
   code .
   ```

2. **Reopen in Container**
   - Press `F1` or `Ctrl+Shift+P` (Windows/Linux) / `Cmd+Shift+P` (Mac)
   - Type "Dev Containers: Reopen in Container"
   - Select the command and wait for the container to build

3. **First-time setup**
   - The container will automatically run `pip install -r .devcontainer/requirements-cpu.txt`
   - This takes 5-10 minutes for PyTorch and dependencies

4. **Verify installation**
   - Open a terminal in VS Code
   - Run: `python -c "import torch; print(f'PyTorch version: {torch.__version__}')"`

### Using IntelliJ IDEA

1. **Open the project in IntelliJ IDEA**
   - File → Open → Select the project directory

2. **Create DevContainer configuration**
   - Go to Settings/Preferences → Build, Execution, Deployment → Docker
   - Ensure Docker is configured and connected
   - Close settings

3. **Connect to DevContainer**
   - Right-click on `.devcontainer/devcontainer.json`
   - Select "Create Dev Container and Mount Sources"
   - Wait for the container to build

4. **Configure Python interpreter**
   - Go to File → Project Structure → Project
   - Click on "Add SDK" → "On Docker"
   - Select the running container
   - Choose `/usr/local/bin/python` as the interpreter

5. **Verify installation**
   - Open the Python Console
   - Run: `import torch; print(f'PyTorch version: {torch.__version__}')`

## Port Forwarding

The following ports are automatically forwarded:
- **8888**: Jupyter Lab/Notebook
- **6006**: TensorBoard

### Starting Jupyter
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Access at: `http://localhost:8888`

## Performance Note

This is a **CPU-only** configuration optimized for:
- Development and code testing
- Small-scale experiments
- Learning and prototyping
- Data exploration and visualization

For production training or large-scale experiments, consider using a GPU-enabled environment.

## Troubleshooting

### Container Build Fails
- Check Docker Desktop is running
- Ensure sufficient disk space (5GB+ recommended)
- Try rebuilding: VS Code → "Dev Containers: Rebuild Container"

### IntelliJ Can't Connect
- Ensure Docker plugin is enabled
- Check Docker is running: `docker ps`
- Try manually building: `docker build -t city-of-agents-devcontainer .devcontainer`

### Slow Package Installation
- PyTorch CPU version is much faster than CUDA (~200MB vs 2GB+)
- First build may take 5-10 minutes
- Subsequent builds use Docker cache and are faster

## Customization

### Adding Python Packages
1. Add to `.devcontainer/requirements-cpu.txt`
2. Rebuild container or run: `pip install <package>`

### Need GPU Support?
If you need GPU acceleration later:
1. Edit `.devcontainer/Dockerfile` - change to `FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`
2. Edit `.devcontainer/devcontainer.json` - add `"runArgs": ["--gpus=all"]`
3. Change `postCreateCommand` to use `requirements.txt` instead of `requirements-cpu.txt`

### Modifying Container
1. Edit `.devcontainer/Dockerfile`
2. Rebuild container:
   - VS Code: `F1` → "Dev Containers: Rebuild Container"
   - IntelliJ: Right-click `.devcontainer/devcontainer.json` → "Rebuild Dev Container"

### VS Code Extensions
Edit `.devcontainer/devcontainer.json` under `customizations.vscode.extensions`

## Notes

- **Lightweight**: Container uses minimal Python 3.11 slim image (~500MB)
- **CPU-only**: PyTorch runs on CPU for development and small experiments
- **Workspace**: Project is mounted at `/workspace` in the container
- **Python version**: Python 3.11 for best compatibility with modern packages
- **Fast startup**: Much faster than GPU-enabled containers

