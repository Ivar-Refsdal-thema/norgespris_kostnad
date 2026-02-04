# norgespris-surveillance


## To run this dashboard you need an .env file with an entsoe api key
ENTSOE_KEY=


## 🧑‍💻 For Developers: Environment & Dependency Management

This project uses **[`uv`](https://github.com/astral-sh/uv)** for Python environments and dependency management.

### 📌 Requirements for using `uv`

To work on the codebase as a developer, you need:

| Requirement | Recommended |
|-------------|-------------|
| Python | 3.12+ |
| Git | Latest |
| uv | Latest |

#### 👉 Install `uv`
```bash
pip install uv
```

Alternatively, download from the official repository.

### 📦 Setup (first time)

Clone the repository and install all dependencies:
```bash
uv sync
```
This command will:
* create or update a .venv virtual environment
* install all dependencies from pyproject.toml
* respect pinned versions from the lock file

### To run the dashboard 
```bash
uv run  streamlit run home.py
```