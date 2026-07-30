"""Loader for NGFS short-term datasets.

The Excel files use an IAMC-style layout:
    Model, Scenario, Region, Variable, Unit, <year columns>

This module standardizes them to:
    Scenario, Region, Variable, Secteur, Unit, <year columns>

By default it keeps production-related rows only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NGFS_DIR = REPO_ROOT / "Data" / "NGFS_ST"
PROCESSED_DIR = REPO_ROOT / "Data" / "processed"

ID_COLUMNS = ["Model", "Scenario", "Region", "Variable", "Unit"]


def _discover_sources() -> dict[str, Path]:
    return {path.stem: path for path in sorted(NGFS_DIR.glob("*.xlsx"))}


def _normalize_time_label(value: object) -> str:
    text = str(value)

    if isinstance(value, (int, float)) and not pd.isna(value):
        if float(value).is_integer():
            return str(int(value))

    if text.isdigit():
        return text

    ts = pd.to_datetime(value, errors="coerce")
    if not pd.isna(ts):
        if ts.month in (1, 4, 7, 10) and ts.day == 1:
            return f"{ts.year}Q{((ts.month - 1) // 3) + 1}"
        return ts.strftime("%Y-%m-%d")

    return text


def _split_variable(series: pd.Series) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=["Variable", "Secteur"], index=series.index)

    parts = series.astype(str).str.split("|", n=1, expand=True)
    if 0 not in parts.columns:
        parts = parts.rename(columns={parts.columns[0]: 0})
    return pd.DataFrame(
        {
            "Variable": parts[0],
            "Secteur": parts[1].where(parts[1].notna() & (parts[1] != ""), pd.NA),
        },
        index=series.index,
    )


def _load_raw(source: str) -> pd.DataFrame:
    sources = _discover_sources()
    if source not in sources:
        raise ValueError(f"Unknown source {source!r}. Available: {list(sources)}")

    xlsx_path = sources[source]
    df = pd.read_excel(xlsx_path, sheet_name=0)
    df.columns = [str(col) if isinstance(col, int) else col for col in df.columns]

    missing = [col for col in ID_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {xlsx_path.name}: {missing}")

    year_cols = [col for col in df.columns if col not in ID_COLUMNS]
    renamed_years = {col: _normalize_time_label(col) for col in year_cols}
    df = df[ID_COLUMNS + year_cols].rename(columns=renamed_years)
    return df


def load_st(
    sources: list[str] | str | None = None,
    variable: str | None = "Production",
) -> dict[str, pd.DataFrame]:
    """Load NGFS short-term dataframes by source.

    Parameters
    ----------
    sources:
        Source name, list of source names, or None for all files.
    variable:
        Root variable to keep. Use None to keep all variables.
    """
    available = _discover_sources()
    if sources is None:
        selected = list(available)
    elif isinstance(sources, str):
        selected = [sources]
    else:
        selected = list(sources)

    out: dict[str, pd.DataFrame] = {}
    for source in selected:
        raw = _load_raw(source).copy()
        year_cols = [col for col in raw.columns if col not in ID_COLUMNS]
        df = raw

        if variable is not None:
            raw_variable = df["Variable"].astype(str)
            mask = raw_variable.eq(variable) | raw_variable.str.startswith(f"{variable}|")
            df = df.loc[mask].copy()

        if df.empty:
            result = pd.DataFrame(columns=["Scenario", "Region", "Variable", "Secteur", "Unit", *year_cols])
        else:
            split = _split_variable(df["Variable"])
            result = pd.concat([df[["Scenario", "Region", "Unit"]], split, df[year_cols]], axis=1)
            result = result[["Scenario", "Region", "Variable", "Secteur", "Unit", *year_cols]]

        out[source] = result.reset_index(drop=True)

    return out


def export_production_excel(
    output_path: Path | str | None = None,
    sources: list[str] | str | None = None,
) -> Path:
    """Export the loaded production tables to one Excel workbook."""
    if output_path is None:
        output_path = PROCESSED_DIR / "production.xlsx"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_st(sources=sources, variable="Production")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for source, frame in data.items():
            sheet_name = source[:31]
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

        combined_frames = [
            frame.assign(Source=source)
            for source, frame in data.items()
            if not frame.empty
        ]
        if combined_frames:
            combined = pd.concat(combined_frames, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=["Scenario", "Region", "Variable", "Secteur", "Unit", "Source"])
        combined.to_excel(writer, sheet_name="combined"[:31], index=False)

    return output_path


def _compute_yoy_growth(frame: pd.DataFrame) -> pd.DataFrame:
    year_cols = [col for col in frame.columns if str(col).isdigit()]
    if not year_cols:
        return frame.copy()

    growth = frame[ ["Scenario", "Region", "Variable", "Secteur", "Unit"] ].copy()
    growth["Variable"] = "Production YoY Growth"

    for idx, year in enumerate(year_cols):
        if idx == 0:
            growth[year] = pd.NA
            continue

        previous_year = year_cols[idx - 1]
        current = pd.to_numeric(frame[year], errors="coerce")
        previous = pd.to_numeric(frame[previous_year], errors="coerce")
        valid = previous.notna() & current.notna()

        values = pd.Series(pd.NA, index=frame.index, dtype="Float64")
        nonzero = valid & (previous != 0)
        values.loc[nonzero] = ((current.loc[nonzero] - previous.loc[nonzero]) / previous.loc[nonzero]) * 100

        zero_base = valid & (previous == 0) & (current == 0)
        values.loc[zero_base] = 0.0
        growth[year] = values

    return growth[["Scenario", "Region", "Variable", "Secteur", "Unit", *year_cols]]


def export_production_growth_excel(
    output_path: Path | str | None = None,
    sources: list[str] | str | None = None,
) -> Path:
    """Export the production YoY growth tables to one Excel workbook."""
    if output_path is None:
        output_path = PROCESSED_DIR / "production_growth.xlsx"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_st(sources=sources, variable="Production")
    growth_data = {source: _compute_yoy_growth(frame) for source, frame in data.items()}

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for source, frame in growth_data.items():
            sheet_name = source[:31]
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

        combined_frames = [
            frame.assign(Source=source)
            for source, frame in growth_data.items()
            if not frame.empty
        ]
        if combined_frames:
            combined = pd.concat(combined_frames, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=["Scenario", "Region", "Variable", "Secteur", "Unit", "Source"])
        combined.to_excel(writer, sheet_name="combined"[:31], index=False)

    return output_path


if __name__ == "__main__":
    production_output = export_production_excel()
    growth_output = export_production_growth_excel()
    print(f"Saved: {production_output}")
    print(f"Saved: {growth_output}")