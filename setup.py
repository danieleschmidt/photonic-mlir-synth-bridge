"""
Setup script for Photonic MLIR.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Version
__version__ = "0.1.0"

setup(
    name="photonic-mlir",
    version=__version__,
    author="Photonic MLIR Team",
    author_email="team@photonic-mlir.org",
    description="MLIR Dialect and HLS Generator for Silicon Photonic AI Accelerators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/photonic-mlir-synth-bridge",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.21.0",
        "pybind11>=2.10.0",
        "psutil>=5.9.0",
        "python-json-logger>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=2.5.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "benchmark": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "photonic-compile=photonic_mlir.cli:compile_command",
            "photonic-simulate=photonic_mlir.cli:simulate_command",
            "photonic-benchmark=photonic_mlir.cli:benchmark_command",
        ],
    },
    include_package_data=True,
    package_data={
        "photonic_mlir": [
            "pdks/*.json",
            "templates/*.mlir",
            "examples/*.py",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/danieleschmidt/photonic-mlir-synth-bridge/issues",
        "Source": "https://github.com/danieleschmidt/photonic-mlir-synth-bridge",
        "Documentation": "https://photonic-mlir.readthedocs.io/",
    },
)