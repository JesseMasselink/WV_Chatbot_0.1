# rag.py
"""
RAG (Retrieval Augmented Generation) PREPROCESSING

This file is responsible for:
1) Loading many CSV files from a folder
2) Cleaning and normalizing ID columns so merges work better
3) Merging key tables into one container dataset
4) Creating readable text "summaries" of each container
5) Storing those summaries in a vector store (Chroma) for semantic search

The vector store is then used by retrieve_context_tool (RAG tool).
"""

import pandas as pd
from pathlib import Path
from langchain.tools import tool
from typing import Tuple, List, Dict
from langchain_chroma import Chroma
from langchain_core.documents import Document
import globals_space

def load_csvs_from_folder(folder: Path) -> Dict[str, pd.DataFrame]:
    """
    Read all CSVs under folder into a dict keyed by file stem (lowercase).
    Skips empty DataFrames and adds '_source_file' column to each.
    """
    globals_space.logger.info("load_csvs_from_folder function called.\n")

    folder = Path(folder)   
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    dataframes = {}
    for file in sorted(folder.rglob("*.csv")):
        try:
            df = pd.read_csv(file)
            # Check if DataFrame is empty
            if df.empty or df.shape[1] == 0:
                globals_space.logger.info(f"Warning: {file.name} is empty and will be skipped.")
                continue
            df["_source_file"] = file.name
            dataframes[file.stem.lower()] = df
        except Exception as e:
            globals_space.logger.warning(f"Warning reading {file.name}: {e}")

    return dataframes


def clean_id_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Clean an ID column to reduce merge issues:
    - Convert to string
    - Strip whitespace
    - Replace missing values with empty strings
    """
    globals_space.logger.info(f"clean_id_column function called for column: {column_name}\n")

    if df is None or df.empty:
        return df
    if column_name not in df.columns:
        return df
        
    df[column_name] = (
        df[column_name]
        .astype(str)             # Convert everything to strings
        .str.strip()             # Remove leading/trailing whitespace
        .replace({"nan": ""})    # Convert "nan" strings to empty
        .fillna("")              # Replace NaN with empty string
    )
    return df


def normalize_id_series(s: pd.Series) -> pd.Series:
    """
    Normalize ID values so that:
    - "123.0" becomes "123"
    - empty/None becomes <NA>
    - strings remain strings

    This helps when IDs are stored inconsistently across CSV files.
    """
    globals_space.logger.info("normalize_id_series function called.\n")

    if s is None or s.empty:
        return s
        
    ser = s.copy()
    # Replace common empty-like tokens with NA
    ser = ser.replace({"": pd.NA, "None": pd.NA})

    # Convert to pandas StringDtype so we can use str.* safely and preserve NA
    ser = ser.astype("string").str.strip()

    def _clean_val(v):
        # v is a python scalar or <NA>
        if pd.isna(v):
            return pd.NA
        # Try numeric conversion first to canonicalize floats like '123.0'
        try:
            # Note: v is a string here (StringDtype), so convert to float
            num = float(v)
            # If integer-valued, render without decimal
            if num.is_integer():
                return str(int(num))
            # Otherwise, normalize float representation (strip trailing zeros)
            s_num = repr(num)
            # Remove trailing zeros and optional trailing dot
            if "." in s_num:
                s_num = s_num.rstrip('0').rstrip('.')
            return s_num
        except Exception:
            # Not numeric — return stripped string
            return str(v)

    cleaned = ser.apply(_clean_val)
    # Ensure dtype is pandas StringDtype and normalize any 'nan' literal to pd.NA
    cleaned = cleaned.replace({"nan": pd.NA}).astype("string")
    return cleaned


def clean_dataframes_ids(dataframes: Dict[str, pd.DataFrame], id_columns_config: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Apply ID cleaning and normalization to specified columns in a dict of DataFrames.
    """
    globals_space.logger.info("clean_dataframes_ids function called.\n")

    for df_name, columns in id_columns_config.items():
        if df_name not in dataframes:
            globals_space.logger.warning(f"Warning: {df_name} not found in dataframes dict. Skipping.")
            continue
        
        df = dataframes[df_name]
        for col in columns:
            if col not in df.columns:
                globals_space.logger.warning(f"Warning: {col} not found in {df_name}. Skipping.")
                continue
            
            # First clean, then normalize
            df = clean_id_column(df, col)
            df[col] = normalize_id_series(df[col]).astype(str).fillna("")
        
        dataframes[df_name] = df

    return dataframes


def load_all_csvs(folder: Path) -> pd.DataFrame:
    """
    Load all CSVs, clean ID columns, and merge container-related tables.
    
    Create one Dataframe that contains:
    - container ID + name
    - location info (street/city)
    - waste fraction
    - container model details
    - filling level + last emptied date (if present)

    This merged data is later converted to summary text chunks.
    """
    globals_space.logger.info("load_all_csvs function called.\n")

    # Load raw CSV files from the data folder
    dataframes = load_csvs_from_folder(folder)
    
    if not dataframes:
        raise RuntimeError("No CSVs loaded from folder")

    # Which columns should be cleaned for each dataset
    id_columns_to_clean = {
        "wastecontainer_export": ["DesignatedLocationId", "WasteContainerModelId", "FractionId"],
        "wastecontainerisle_export": ["Id"],
        "wastecontainermodel_export": ["Id"],
        "designatedlocation_export": ["Id", "DesignatedLocationTypeId"],
        "designatedlocationtype_export": ["Id"],
        "device_export": ["Id", "DeviceModelId", "WasteContainerId"],
        "devicemodel_export": ["Id"],
        "fraction_export": ["Id"],
        "ordercontainers_export": ["Id", "OrderId", "WasteContainerId"],
        "orders_export": ["Id"],
    }

    # Apply cleaning to all specified columns
    dataframes = clean_dataframes_ids(dataframes, id_columns_to_clean)

    # Ensure required DataFrames exist (or create empty ones)
    def safe_get(key: str) -> pd.DataFrame:
        return dataframes.get(key, pd.DataFrame())

    wastecontainer_export = safe_get("wastecontainer_export")
    wastecontainerisle_export = safe_get("wastecontainerisle_export")
    fraction_export = safe_get("fraction_export")
    wastecontainermodel_export = safe_get("wastecontainermodel_export")

    if wastecontainer_export.empty:
        raise RuntimeError("wastecontainer_export is required but not found or empty")

    #  Merge multiple tables into one wide table (left join keeps all containers)
    container_merged = (
        wastecontainer_export
        .merge(
            wastecontainerisle_export,
            left_on="DesignatedLocationId",
            right_on="Id",
            how="left",
            suffixes=("", "_isle"),
        )
        .merge(
            fraction_export,
            left_on="FractionId",
            right_on="Id",
            how="left",
            suffixes=("", "_fraction"),
        )
        .merge(
            wastecontainermodel_export,
            left_on="WasteContainerModelId",
            right_on="Id",
            how="left",
            suffixes=("", "_model"),
        )
    )

    return container_merged


def build_summary_chunks(df: pd.DataFrame) -> List[Dict]:
    """
    Produce a list of chunks from merged container dataframe.
    """
    globals_space.logger.info("Build_summary_chunks function called.\n")

    def format_date(date_str):
        """Format date string or return 'unknown date' if invalid"""
        if pd.isna(date_str):
            return "unknown date"
        try:
            return pd.to_datetime(date_str).strftime('%b %d')
        except:
            return "unknown date"

    def format_filling(level):
        """Format filling level or return 'unknown' if invalid"""
        if pd.isna(level):
            return "unknown"
        try:
            return f"{int(float(level))}%"
        except:
            return "unknown"

    def to_summary(row):
        # Extract values with safe fallbacks so missing data does not crash the pipeline
        container_id = row.get('Number', 'Unknown ID')
        display_name = f" ({row['DisplayName']})" if pd.notna(row.get('DisplayName')) else ""
        
        # Address
        street = row.get('Address_StreetAddress', 'unknown location')
        if pd.isna(street) or street == 'nan':
            street = 'unknown location'
        
        city = row.get('Address_AddressLocality', 'unknown city')
        if pd.isna(city) or city == 'nan':
            city = 'unknown city'
        
        # Waste type
        fraction_name = row.get('Name_fraction', 'unknown type')
        if pd.isna(fraction_name) or fraction_name == 'nan':
            fraction_name = 'unknown type'
        
        # Model info
        model_name = row.get('Name_model', 'unknown model')
        if pd.isna(model_name) or model_name == 'nan':
            model_name = 'unknown model'
        
        # Capacity
        capacity_val = row.get('Capacity')
        if capacity_val is None or pd.isna(capacity_val):
            capacity = "unknown capacity"
        else:
            try:
                capacity_num = float(capacity_val)
                capacity_str = f"{capacity_num:.1f}".rstrip('0').rstrip('.')
                capacity = f"{capacity_str} m3"
            except:
                capacity = "unknown capacity"
        
        # Filling level
        filling = format_filling(row.get('FillingLevel'))
        
        # Last emptying date
        last_empty = format_date(row.get('DateLastEmptying'))
        
        # Status
        status = "under maintenance" if row.get('IsUnderMaintenance') == True else "operational"
        
        # This is the text summary for one container that will be stored in the vector store
        return (
            f"Container {container_id}{display_name} is located at "
            f"{street}, {city}. "
            f"It collects {fraction_name} waste, uses model {model_name} with {capacity}. "
            f"Current fill level is {filling}. Last emptied on {last_empty}. "
            f"It is {status}."
        )

    summary_chunks = []
    for _, row in df.iterrows():
        content = to_summary(row)
        metadata = {
            "container_id": str(row.get('Number', '')),
            "location_street": str(row.get('Address_StreetAddress', '')),
            "location_city": str(row.get('Address_AddressLocality', '')),
            "waste_type": str(row.get('Name_fraction', '')),
            "model": str(row.get('Name_model', '')),
            "source": "container-details"
        }
        summary_chunks.append({"content": content, "metadata": metadata})

    if not summary_chunks:
        globals_space.logger.info("No summary chunks were created.\n")
    else:
        globals_space.logger.info(f"Created {len(summary_chunks)} summary chunks.")    

    return summary_chunks


def build_chroma_vector_store(summary_chunks: List[Dict], embedding_model, persist_path: str = "./chroma_location_embeddings", collection_name: str = "location_summaries") -> Chroma:
    """
    Create a Chroma vector store and add texts + metadata.
    - "texts" are the container summaries
    - "metadatas" are extra fields we store for filtering/debugging
    """
    globals_space.logger.info("build_vector_store function called.\n")

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_path
    )

    if not summary_chunks:
        globals_space.logger.warning("No chunks to add to vector store.")
        return vector_store

    # Use add_texts to avoid Document API mismatches
    texts = [c["content"] for c in summary_chunks]
    metadatas = [c["metadata"] for c in summary_chunks]
    vector_store.add_texts(texts, metadatas)
    
    return vector_store


def get_retriever(vector_store: Chroma, k: int):
    """
    Get a retriever from a Chroma vector store.
    - k = how many top relevant chunks are retrieved per query
    """
    globals_space.logger.info("get_retriever function called.\n")

    return vector_store.as_retriever(search_kwargs={"k": k})
