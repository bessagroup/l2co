# Installation

The package is currently not available on PyPI. To install the package, follow the steps below:

### Installation instructions for users


1. Create a new virtual environment (optional) with Python 3.10+


2. Clone the repository:

```bash
git clone https://github.com/bessagroup/L2CO.git
```

3. Navigate to the repository and install the package using `pip`:

```bash
pip install -e .
```

### Installation instructions for developers

To install the package for development (or for building the sphinx documentation), install the development dependencies after normal installation:

```bash
pip install -e '.[dev,docs,tests,app]'
```

> For detailed instructions on how to develop on this package, please refer to the Contributing Guide on the [GitHub repository](https://github.com/bessagroup/l2co).

After installation, start by looking at the [Getting Started Guide](usage.md).