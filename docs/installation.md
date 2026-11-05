# Installation

### Installation instructions for users

Install `l2co` from PyPI into an environment with Python 3.12 or newer:

```bash
pip install l2co
```

### Installation instructions for developers

Clone the repository and install it in editable mode, with the development dependencies (also needed for building the mkdocs documentation):

```bash
git clone https://github.com/bessagroup/l2co.git
cd l2co
pip install -e '.[dev,docs,tests]'
```

With `uv`, a plain `uv sync` expects checkouts of `f3dasm`, `l2co-tasks`, `l2co-optimizers` and `bessaplots` next to this repository (see `[tool.uv.sources]` in `pyproject.toml`). Run `uv sync --no-sources` to take them from PyPI instead.

> For detailed instructions on how to develop on this package, please refer to the Contributing Guide on the [GitHub repository](https://github.com/bessagroup/l2co).

After installation, start by looking at the [Getting Started Guide](usage.ipynb).