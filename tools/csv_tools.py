# csv_tools.py
import pandas as pd
from langchain.tools import tool
from pathlib import Path

@tool
def load_waste_management_data() -> str:
    """
    Load waste management data from all CSV files under project_root/data and
    return a brief summary. Uses recursive search (rglob) and tags rows with
    their source filename.
    """
    project_root = Path(__file__).resolve().parents[2]
    folder = project_root / "GAD2"

    if not folder.exists():
        return f"Data folder not found: {folder}"

    # recursive search for .csv files
    csv_files = sorted(folder.rglob("*.csv"))
    if not csv_files:
        return f"No CSV files found in: {folder}"

    dfs = []
    filenames = []
    errors = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["_source_file"] = f.name
            dfs.append(df)
            filenames.append(f.name)
        except Exception as e:
            errors.append(f"{f.name}: {e}")

    if not dfs:
        return "No valid CSV files could be read. Errors: " + "; ".join(errors)

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    summary_lines = [
        f"Loaded {len(filenames)} CSV file(s) from: {folder}",
        f"Files: {', '.join(filenames)}",
        f"Total rows: {len(combined)}",
        f"Columns: {', '.join(map(str, combined.columns))}",
    ]
    if errors:
        summary_lines.append("\nSome files failed to load:")
        summary_lines.extend(errors)

    summary = "\n".join(summary_lines)
    sample = "\n\nSample (first 5 rows):\n" + combined.head(5).to_string(index=False)

    return summary + sample