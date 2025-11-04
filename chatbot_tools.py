from langchain_core.tools import tool
import pandas as pd

@tool("search_database")
def search_database(file_path: str) -> str:
    """Searches the database CSV file for records matching the query.
    
    Args:
        Query: Search terms to look for in the database.
        Returns: Relevant records from the database as a string.
    """
    

    # Load the CSV file
    df = pd.read_csv(file_path)

    # For demonstration, return the first 5 rows as a string
    return df.head().to_string()

def create_csv_agent(
    llm: BaseLanguageModel,
    path: Union[str, IOBase, List[Union[str, IOBase]]],
    pandas_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Create csv agent by loading to a dataframe and using pandas agent."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas package not found, please install with `pip install pandas`"
        )

    _kwargs = pandas_kwargs or {}
    if isinstance(path, (str, IOBase)):
        df = pd.read_csv(path, **_kwargs)
    elif isinstance(path, list):
        df = []
        for item in path:
            if not isinstance(item, (str, IOBase)):
                raise ValueError(f"Expected str or file-like object, got {type(path)}")
            df.append(pd.read_csv(item, **_kwargs))
    else:
        raise ValueError(f"Expected str, list, or file-like object, got {type(path)}")
    return create_pandas_dataframe_agent(llm, df, **kwargs)