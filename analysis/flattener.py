import pandas as pd
import ast
from tqdm import tqdm

def safe_parse(x):
    """
    Convert a stringified JSON to a Python object if it starts with '{' or '['.
    Otherwise, return the original value.
    """
    if isinstance(x, str) and (x.startswith("{") or x.startswith("[")):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x

def flatten_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Flattens a single column in a DataFrame if it contains nested lists or dictionaries.
    It converts any stringified JSON into a Python object first.
    """
    if col_name not in df.columns:
        return df

    # Parse stringified JSON for this column
    df[col_name] = df[col_name].apply(safe_parse)

    # CASE 1: The column contains lists
    if df[col_name].apply(lambda x: isinstance(x, list)).any():
        # Explode lists into multiple rows.
        df_expanded = df.explode(col_name).reset_index(drop=True)
        # Add an index for items in the list.
        df_expanded["item_index"] = df_expanded.groupby(df_expanded.index).cumcount() + 1

        # Normalize dictionaries inside these lists.
        series_nonnull = df_expanded[col_name].dropna()
        if series_nonnull.apply(lambda x: isinstance(x, dict)).any():
            normalized = pd.json_normalize(series_nonnull)
            normalized.columns = [f"{col_name}.{subcol}" for subcol in normalized.columns]
            df_expanded = pd.concat([df_expanded.drop(columns=[col_name]), normalized], axis=1)
        else:
            # If the exploded list is not a dict, just rename the column.
            df_expanded.rename(columns={col_name: f"{col_name}.value"}, inplace=True)

        return df_expanded

    # CASE 2: The column contains dictionaries
    if df[col_name].apply(lambda x: isinstance(x, dict)).any():
        normalized = pd.json_normalize(df[col_name].dropna())
        normalized.columns = [f"{col_name}.{subcol}" for subcol in normalized.columns]
        df = pd.concat([df.drop(columns=[col_name]), normalized], axis=1)

    return df

def fully_flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recursively flattens all columns in a DataFrame that contain nested data (lists or dictionaries),
    including those represented as strings.
    Uses tqdm to display progress.
    """
    # First, apply safe_parse to every cell in the DataFrame.
    df = df.applymap(safe_parse)
    
    while True:
        # Identify columns that still contain nested structures.
        nested_cols = [
            col for col in df.columns
            if df[col].dropna().apply(lambda x: isinstance(x, (list, dict))).any()
        ]
        if not nested_cols:
            break  # Stop when no nested columns remain.
        
        # Process each nested column with a progress bar.
        for col in tqdm(nested_cols, desc="Flattening columns"):
            df = flatten_column(df, col)
    return df
def load_data(file_path:str = "data/raw/sample.csv") -> pd.DataFrame:
    """Loads data from a csv file and return a pandas Dataframe.
    """
    df = pd.read_csv(file_path, low_memory=False)
    return df
if __name__ == "__main__":
    df = load_data()
    flattened = fully_flatten_dataframe(df)
    flattened.to_csv("data/processed/sample.csv", index=False)