import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path:str = "data/raw/bots/status_25k.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False)
    return df

def data_inspection(df:pd.DataFrame) -> None:
    summary = {
        "Shape": df.shape,
        "Columns of the DataFrame": df.columns,
        "Data Types": df.dtypes,
        
        }
    print(f"Shape of the dataframe: {df.shape}")
    print(f"Columns of the dataframe: {df.columns}")
    print(f"Data types of the columns: {df.dtypes}")
    print(f"Missing values in the dataframe: {df.isnull().sum()}")
    print(f"Data types of the columns: {df.dtypes}")
    cols_with_lists = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]
    # Apply nunique only to columns that do not contain lists
    hashable_cols = [col for col in df.columns if col not in cols_with_lists]
    print(f"Number of unique values in each column: {df[hashable_cols].nunique()}")
    print(f"Number of duplicate rows: {df[hashable_cols].duplicated().sum()}")
    print(f"Number of missing values in each column: {df.isnull().mean().sort_values()}")
    print(df.describe())
    print(df.info())

def plot_null(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title('Missing Values Heatmap')
    plt.show()
    print(f"Missing values in the dataframe: {df.isnull().sum()}")

def flatten_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
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

def t_test(df: pd.DataFrame, column_to_test: str)-> int:
    df_test = df.copy()
    df_test[column_to_test]
    df_test.loc[np.random(len(df_test)<0.05, seed = 21): column_to_test] = np.nan
    print(df_test[column_to_test].is_null().sum())
if __name__ == "__main__":
    df = load_data()
    df =df[df.columns.difference(['user','entities'])]
    df = flatten_dataframe(df)
    cols_in_list = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list) and bool(x)).any()]
    print(cols_in_list)
    #df =df[df.columns.difference(cols_in_list)]
    df.to_csv("data/processed/flattened_status.csv", index=False)

    #Make the null values a category of NaN
    data_inspection(df)
    # plot_null(df)
    # col_to_test = df[df.isnull().mean()>.5].columns
    # for i in col_to_test:
    #     t_test(df, i)
    # 