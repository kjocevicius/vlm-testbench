# AGENTS.md

## Notebook workflow

All notebook-style work **must** use Python scripts in **percent format** (`# %%` cell markers), paired with `.ipynb` via [Jupytext](https://jupytext.readthedocs.io/).

- Prefer editing and running the `.py` percent-format script (VS Code / Cursor support cell execution natively).
- Use the `.ipynb` notebook only when you need rendered outputs or rich display.
- Jupytext keeps both files in sync — edits to either propagate on save/sync.
- Configuration lives in `notebooks/jupytext.toml`.

### Percent format reference

```python
# %% [markdown]
# # Title
# Some markdown description.

# %%
import pandas as pd

df = pd.read_csv("data.csv")

# %%
df.head()
```
