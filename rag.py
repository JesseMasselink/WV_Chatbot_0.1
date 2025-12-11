import pandas as pd
from pathlib import Path
from langchain.tools import tool
from typing import Tuple, List, Dict
from langchain_chroma import Chroma
from langchain_core.documents import Document

def load_csvs_from_folder(folder: Path) -> Dict[str, pd.DataFrame]:
    """
    Read all CSVs under folder into a dict keyed by file stem (lowercase).
    Skips empty DataFrames and adds '_source_file' column to each.
    
    Args:
        folder: Path to folder containing CSV files
        
    Returns:
        Dict mapping {stem_name: DataFrame}
    """
    folder = Path(folder)   
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    dataframes = {}
    for file in sorted(folder.rglob("*.csv")):
        try:
            df = pd.read_csv(file)
            # Check if DataFrame is empty
            if df.empty or df.shape[1] == 0:
                print(f"Warning: {file.name} is empty and will be skipped.")
                continue
            df["_source_file"] = file.name
            dataframes[file.stem.lower()] = df
        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    return dataframes


def clean_id_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Clean an ID column by standardizing data types and handling nulls consistently.
    
    Args:
        df: DataFrame containing the column to clean
        column_name: Name of the column to clean
        
    Returns:
        DataFrame with cleaned column
    """
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
    Normalize an ID-looking Series to a consistent string form while preserving missingness.

    Rules:
    - Trim whitespace
    - Treat empty strings / 'None' as missing (pd.NA)
    - If value looks numeric and is integer-valued (e.g. 123.0), convert to '123'
    - Preserve non-numeric strings as-is (after strip)
    - Return pandas StringDtype to keep <NA> semantics
    
    Args:
        s: pandas Series (typically of ID values)
        
    Returns:
        Normalized Series with StringDtype
    """
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
    
    Args:
        dataframes: Dict of DataFrames keyed by name
        id_columns_config: Dict mapping {df_name: [list of column names to clean]}
        
    Returns:
        Dict of cleaned DataFrames (modifies in place)
    """
    for df_name, columns in id_columns_config.items():
        if df_name not in dataframes:
            print(f"Warning: {df_name} not found in dataframes dict. Skipping.")
            continue
        
        df = dataframes[df_name]
        for col in columns:
            if col not in df.columns:
                print(f"Warning: {col} not found in {df_name}. Skipping.")
                continue
            
            # First clean, then normalize
            df = clean_id_column(df, col)
            df[col] = normalize_id_series(df[col]).astype(str).fillna("")
        
        dataframes[df_name] = df

    return dataframes


def load_all_csvs(folder: Path) -> pd.DataFrame:
    """
    Load all CSVs, clean ID columns, and merge container-related tables.
    
    Args:
        folder: Path to folder containing CSV files
        
    Returns:
        Merged container DataFrame with isle, fraction, and model details
    """
    # Step 1: Load raw CSVs
    dataframes = load_csvs_from_folder(folder)
    
    if not dataframes:
        raise RuntimeError("No CSVs loaded from folder")

    # Step 2: Define which columns to clean in which DataFrames
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

    # Step 3: Apply cleaning to all specified columns
    dataframes = clean_dataframes_ids(dataframes, id_columns_to_clean)

    # Step 4: Ensure required DataFrames exist (or create empty ones)
    def safe_get(key: str) -> pd.DataFrame:
        return dataframes.get(key, pd.DataFrame())

    wastecontainer_export = safe_get("wastecontainer_export")
    wastecontainerisle_export = safe_get("wastecontainerisle_export")
    fraction_export = safe_get("fraction_export")
    wastecontainermodel_export = safe_get("wastecontainermodel_export")

    if wastecontainer_export.empty:
        raise RuntimeError("wastecontainer_export is required but not found or empty")

    # Step 5: Perform merges (left joins to keep all containers)
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
    Produce a list of {'content', 'metadata'} chunks from merged container dataframe.
    """

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
        """Generate a natural language summary of a container's details"""
        # Container core info
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
        capacity_val = row.get('Statistics_CapacityM3') or row.get('Capacity')
        if pd.notna(capacity_val):
            try:
                capacity = f"{float(capacity_val):.1f}m³".rstrip('0').rstrip('.')
            except:
                capacity = "unknown capacity"
        else:
            capacity = "unknown capacity"
        
        # Filling level
        filling = format_filling(row.get('FillingLevel'))
        
        # Last emptying date
        last_empty = format_date(row.get('DateLastEmptying'))
        
        # Status
        status = "under maintenance" if row.get('IsUnderMaintenance') == True else "operational"
        
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
        print("No summary chunks were created.")
    else:
        print(f"Created {len(summary_chunks)} summary chunks.")    

    return summary_chunks


def build_vector_store(summary_chunks: List[Dict], embedding_model, persist_path: str = "./chroma_location_embeddings", collection_name: str = "location_summaries") -> Chroma:
    """
    Create a Chroma vector store and add texts+metadatas.
    
    Args:
        summary_chunks: List of {'content', 'metadata'} dicts
        embedding_model: Embedding function (e.g. OllamaEmbeddings)
        persist_path: Directory to persist the vector store
        collection_name: Name of the Chroma collection
        
    Returns:
        Chroma vector store object
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_path
    )

    if not summary_chunks:
        print("No chunks to add to vector store.")
        return vector_store

    # Use add_texts to avoid Document API mismatches
    texts = [c["content"] for c in summary_chunks]
    metadatas = [c["metadata"] for c in summary_chunks]
    vector_store.add_texts(texts, metadatas)
    
    return vector_store


def get_retriever(vector_store: Chroma, k: int):
    """
    Get a retriever from a Chroma vector store.
    
    Args:
        vector_store: Chroma vector store object
        k: Number of top results to retrieve
        
    Returns:
        Retriever object
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
