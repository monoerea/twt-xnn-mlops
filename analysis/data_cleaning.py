import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path:str = "data/raw/status_25k.csv") -> pd.DataFrame:
    """Loads data from a csv file and return a pandas Dataframe.
    """
    df = pd.read_csv(file_path, low_memory=False)
    return df

def data_inspection(df:pd.DataFrame) -> None:
    """Prints the first 5 rows of the dataframe, the shape of the dataframe and the columns of the dataframe.
    """
    print("First 5 rows of the dataframe:")
    print(df.head())
    print(f"Shape of the dataframe: {df.shape}")
    print(f"Columns of the dataframe: {df.columns}")
    print(f"Data types of the columns: {df.dtypes}")
    print(f"Missing values in the dataframe: {df.isnull().sum()}")
    print(f"Data types of the columns: {df.dtypes}")
    print(f"Number of unique values in each column: {df.nunique()}")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    print(f"Number of missing values in each column: {df.isnull().sum()}")
    print(df.describe())
    print(df.info())

def plot_null(df: pd.DataFrame) -> None:
    """Plots the null values in the dataframe.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()
    print(f"Missing values in the dataframe: {df.isnull().sum()}")

import pandas as pd
import ast

def flatten_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Flattens a column with nested lists or dictionaries and numbers list items if needed."""
    if col_name not in df.columns:
        return df  # Skip if column doesn't exist

    # Convert string JSON to Python objects
    df[col_name] = df[col_name].dropna().apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # If column contains a list, explode and number items
    if df[col_name].apply(lambda x: isinstance(x, list)).any():
        df_expanded = df.explode(col_name).reset_index(drop=True)
        df_expanded["item_index"] = df_expanded.groupby(df_expanded.index).cumcount() + 1

        # Normalize dictionaries inside lists
        normalized = pd.json_normalize(df_expanded[col_name].dropna())
        normalized.columns = [f"{col_name}.{subcol}" for subcol in normalized.columns]

        return pd.concat([df_expanded.drop(columns=[col_name]), normalized], axis=1)

    # If column contains dictionaries, flatten them
    if df[col_name].apply(lambda x: isinstance(x, dict)).any():
        normalized = pd.json_normalize(df[col_name])
        normalized.columns = [f"{col_name}.{subcol}" for subcol in normalized.columns]
        return pd.concat([df.drop(columns=[col_name]), normalized], axis=1)

    return df  # Return as-is

def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Flattens all JSON-like structures in a DataFrame."""
    nested_cols = [col for col in df.columns if df[col].dropna().apply(lambda x: isinstance(x, (list, dict)) or (isinstance(x, str) and x.startswith("{"))).any()]

    for col in nested_cols:
        df = flatten_column(df, col)

    return df


if __name__ == "__main__":
    df = load_data()
    #data_inspection(df)
    #plot_null(df)
    flattened = flatten_dataframe(df)
    flattened.to_csv("data/processed/flattened.csv", index=False)