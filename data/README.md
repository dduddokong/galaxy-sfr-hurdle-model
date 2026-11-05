# Data

The competition data is **not included** in this repository (it is not redistributed here).

| | |
|---|---|
| **Dataset** | *Galaxy features with position and velocity for star formation rate prediction* (Fukushima & Kim, 2026) |
| **Source** | SNU IDIS DataHub, dataset 42, v1.0.0: <https://datahub.snu.ac.kr/dataset/42> |
| **Licence** | CC BY 4.0 |
| **Size** | 208,437 rows × 15 columns |

## Expected location

Download the CSV and save it as:

```text
data/raw/galaxy_sfr.csv
```

All scripts in `src/` read this path relative to the repo root (e.g. `src/rs_00_common.py`).
Everything under `data/` except this README is excluded by `.gitignore`.
