"""
Setup script for City of Agents package.
For modern Python projects, pyproject.toml is preferred, but setup.py
provides backward compatibility and Colab compatibility.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="city-of-agents",
    version="0.1.0",
    author="City of Agents Research Team",
    description="Code Analysis and Graph Generation Framework for Software Metrics Visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohameddhameem/City-of-Agents",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "plotly>=5.0",
        "datasets",
        "matplotlib",
        "seaborn",
        "networkx",
        "ogb>=1.3.6",
        "transformers==4.47.0",
        "torch>=2.0.0",
        "torch-geometric",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.0",
            "flake8>=6.0",
        ],
        "notebooks": [
            "ipykernel",
            "jupyter",
            "jupyterlab",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
    ],
    keywords="code-analysis graph-neural-networks software-metrics visualization ast pytorch-geometric",
)
