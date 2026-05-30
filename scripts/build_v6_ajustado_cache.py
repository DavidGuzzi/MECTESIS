"""
Construye notebooks/results/covariate_v6_ajustado_vertexai/ fusionando:
  - todos los exp_*.csv del v6_vertexai original *excepto* los del bloque C-D
  - los exp_C-D_*.csv del v6_garch_vertexai filtrando la fila de SARIMAX(1,0,0)

El notebook experimentos_covariables_v6_cloud_ajustado.ipynb usa esa carpeta como
caché de `run_exp_cov`: al ejecutar "Run All" cada celda hace cache-hit y no re-simula.

Idempotente: si el destino ya existe, los archivos se sobreescriben.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_BASE = ROOT / "notebooks" / "results" / "covariate_v6_vertexai"
SRC_GARCH = ROOT / "notebooks" / "results" / "covariate_v6_garch_vertexai"
DST = ROOT / "notebooks" / "results" / "covariate_v6_ajustado_vertexai"

SARIMAX_DROPPED = "SARIMAX(1, 0, 0) con X"


def is_cd(path: Path) -> bool:
    return path.name.startswith("exp_C-D_")


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_filtered = 0

    base_csvs = sorted(SRC_BASE.glob("exp_*.csv"))
    if not base_csvs:
        print(f"[ERROR] no se encontraron CSVs en {SRC_BASE}", file=sys.stderr)
        return 1

    for src in base_csvs:
        if is_cd(src):
            continue
        shutil.copy2(src, DST / src.name)
        n_copied += 1

    garch_csvs = sorted(SRC_GARCH.glob("exp_C-D_*.csv"))
    if not garch_csvs:
        print(f"[ERROR] no se encontraron C-D CSVs en {SRC_GARCH}", file=sys.stderr)
        return 1

    for src in garch_csvs:
        df = pd.read_csv(src)
        before = len(df)
        df = df[df["model"] != SARIMAX_DROPPED].reset_index(drop=True)
        after = len(df)
        if before == after:
            print(
                f"[WARN] {src.name}: no se encontraron filas {SARIMAX_DROPPED!r} "
                f"para filtrar (¿corrida con benchmark distinto?).",
                file=sys.stderr,
            )
        df.to_csv(DST / src.name, index=False)
        n_filtered += 1

    total = len(list(DST.glob("exp_*.csv")))
    print(f"Copiados sin tocar (no-C-D):     {n_copied}")
    print(f"Filtrados (C-D, sin SARIMAX):    {n_filtered}")
    print(f"Total CSVs en {DST.name}: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
