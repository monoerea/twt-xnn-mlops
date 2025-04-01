import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class DataInspector:
    """
    Automates dataframe inspection, generates CSV files for summaries,
    and saves Seaborn plots as images (e.g., missing values heatmap).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
    def basic_info(self):
        """Returns basic dataset information as a dataframe."""
        types = self.df.loc[:, ~self.df.columns.str.contains("id", case=False)].dtypes.value_counts()
        info = {
            "Metric": ["Shape (rows, cols)", "Total Columns", "Missing Values (Total)", "Duplicate Rows", "Shape of dtypes (not id): " + ', '.join(str(dtype) for dtype in types.keys())],
            "Value": [
                self.df.shape,
                len(self.df.columns),
                self.df.isnull().sum().sum(),
                self.df.duplicated().sum(),
                [int(key) for key in types.values]
            ]
        }
        basic_info_df = pd.DataFrame(info)

        # Print basic info to the console
        print("\nBasic Information:")
        print(tabulate(basic_info_df, headers='keys', tablefmt='pretty', showindex=False))
        print(info)

        return basic_info_df

    def unique_values(self):
        """Returns unique values per column (excluding unhashable types like lists)."""
        return self.df.nunique().reset_index().rename(columns={"index": "Column", 0: "Unique Values"})

    def missing_values(self):
        """Returns missing values per column."""
        missing = self.df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        missing["Percentage"] = (missing["Missing Values"] / len(self.df)) * 100
        return missing.sort_values(by="Missing Values", ascending=False)

    def descriptive_statistics(self):
        """Returns descriptive statistics of the dataset."""
        return self.df.describe(include='all').T.reset_index().rename(columns={"index": "Column"})

    def plot_missing_values(self, filename="missing_values_heatmap.png"):
        """Visualizes missing values using a heatmap and saves it to an image file."""
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.df.isnull(), cmap="coolwarm", cbar=False, yticklabels=False)
        plt.title("Missing Values Heatmap")
        plt.savefig(filename, format="png")
        plt.show()
        plt.close()  # Close the plot to avoid memory issues
        print(f"Missing values heatmap saved as {filename}.")

    def plot_distribution(self, filename="numeric_distribution.png"):
        """Plots distribution of numeric columns and saves as an image."""
        numeric_cols = self.df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            print("\n❌ No numeric columns found for distribution plots.")
            return

        self.df[numeric_cols].hist(figsize=(100, 30), bins=301, edgecolor='black')
        plt.suptitle("Distribution of Numeric Columns", fontsize=14)
        plt.savefig(filename, format="png")
        plt.show()
        plt.close()
        print(f"Numeric distributions saved as {filename}.")

    def save_csv_summary(self, filename="data/processed/data_summary.csv"):
        # Prepare the summary table in a single step for efficiency
        summary = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes,
            'Missing Values': self.df.isnull().sum(),
            'Unique Values': self.df.nunique(),
            'Percentage of Missing Values': (self.df.isnull().mean() * 100)
        })
        # Ensure all columns are of type float or int (for clarity)
        summary['Missing Values'] = summary['Missing Values'].astype(int)
        summary['Unique Values'] = summary['Unique Values'].astype(int)
        summary['Percentage of Missing Values'] = summary['Percentage of Missing Values'].round(2)
            # Save to CSV
        summary.to_csv(filename, index=False)
        print(f"CSV summary saved as {filename}.")

    def generate_reports(self, filename):
        """Generates CSV summary and saves images for plots."""
        self.basic_info()
        self.save_csv_summary(filename)
        self.plot_missing_values("missing_values_heatmap.png")
        self.plot_distribution("numeric_distribution.png")
        print("\n✅ All reports generated: CSV summary and plot images saved.")

# Example Usage:
df = pd.read_csv("data/processed/flattened_status.csv", low_memory=False)  # Load your dataframe
inspector = DataInspector(df)
inspector.basic_info()
