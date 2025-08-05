# Graph-Isomorphism


## Setup

To setup the project, you will need [uv](https://github.com/astral-sh/uv). You can download it via `pip` or `pipx`. After instalation you can run:

```
uv sync
source .venv/bin/activate
python main.py
```

### Precommit 

To enable automatic pre-commit hooks on every commit, run:
```
pre-commit install
```

To manually run all configured hooks on (without making a commit), use:
```
pre-commit run --all-files
```

