import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

class DataInspector:
    """
    Automates dataframe inspection, generates CSV files for summaries,
    and saves Seaborn plots as images (e.g., missing values heatmap).
    """

    def __init__(self, df: pd.DataFrame, thresholds = None, exclude_colums = None):
        self.df = df
        self.summary = None
        self.thresholds = thresholds
        self.exclude_columns = exclude_colums
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
        print(basic_info_df)
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

    def missing_value_analysis(self):
        """Analyzes columns with missing values within the specified threshold range and performs statistical tests."""
        missing_pct = self.calculate_missing_percentage()

        # Filter columns based on missing percentage and other criteria
        missing_cols = self.filter_columns_by_missing_data(missing_pct)

        if not missing_cols:
            print("\n✅ No columns meet the missing value criteria.")
            return

        # Perform t-tests on filtered columns
        t_test_results = self.perform_t_tests(missing_cols)

        # Save the results to CSV
        if t_test_results:
            results_df = pd.DataFrame(t_test_results)
            print(results_df)
            self.save_dataframe_to_csv(results_df)

    def calculate_missing_percentage(self):
        """Calculates the missing percentage for each column."""
        return self.df.isnull().mean()

    def filter_columns_by_missing_data(self, missing_pct):
        """Filters columns that have missing data within the specified thresholds and excludes unwanted columns."""
        min_missing, max_missing = self.thresholds

        return self.df.columns[
            (missing_pct > min_missing) &
            (missing_pct < max_missing) &
            self.df.columns.str.contains(",".join(self.exclude_columns), case=False) &
            (self.df.dtypes != 'object')
        ].tolist()

    def perform_t_tests(self, missing_cols):
        """Performs t-tests on the given columns and returns a list of results."""
        t_test_results = []
        
        for col in missing_cols:
            result = self._perform_t_test(col)
            if result:
                t_test_results.append(result)

        return t_test_results

    def _perform_t_test(self, col: str):
        """Performs a t-test between available and missing-value samples of a column."""
        if col not in self.df.columns:
            print(f"\n❌ Column '{col}' not found in dataset.")
            return None

        df_test = self.df[col].copy()

        # Simulate missing data in 5% of the column
        df_test = self.simulate_missing_data(df_test)

        # Perform the t-test
        original_values = self.df[col].dropna()
        test_values = df_test.dropna()

        return self.calculate_t_statistic_and_p_value(original_values, test_values, col)

    def simulate_missing_data(self, df_test):
        """Simulates missing data by introducing NaNs in 5% of the data."""
        mask = df_test.sample(frac=0.05, random_state=21).index
        df_test.loc[mask] = np.nan
        return df_test

    def calculate_t_statistic_and_p_value(self, original_values, test_values, col):
        """Calculates the t-statistic and p-value for the given data."""
        if len(original_values) > 1 and len(test_values) > 1:
            t_stat, p_value = ttest_ind(original_values, test_values, equal_var=False, nan_policy='omit')
            return {
                "Column": col,
                "t-statistic": t_stat,
                "p-value": p_value
            }
        else:
            print(f"\n⚠️ Not enough data for t-test on '{col}'.")
            return None

    def save_dataframe_to_csv(self, df: pd.DataFrame, filename="analysis/t_test_results.csv"):
        """Saves the DataFrame to a CSV file."""
        df.to_csv(filename, index=False)
        print(f"\n✅ Results saved to {filename}.")
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

    def save_csv_summary(self, filename="analysis/data_summary.csv"):
        """Saves a CSV summary of dtype, missing values, and unique values."""
        self.summary = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes,
            'Missing Values': self.df.isnull().sum(),
            'Unique Values': self.df.nunique(),
            'Percentage of Missing Values': (self.df.isnull().mean() * 100).round(2)
        })

        self.summary.to_csv(filename, index=False)
        print(f"📄 CSV summary saved as {filename}.")

    def generate_reports(self, filename="analysis/data_summary.csv"):
        """Generates CSV summary and saves images for plots."""
        self.basic_info()
        self.save_csv_summary(filename)
        self.plot_missing_values("missing_values_heatmap.png")
        #self.plot_distribution("numeric_distribution.png")
        self.missing_value_analysis()
        print("\n✅ All reports generated: CSV summary and plot images saved.")


if __name__ == "__main__":
    # Example Usage:
    df = pd.read_csv("data/processed/flattened_status.csv", low_memory=False)  # Load your dataframe
    inspector = DataInspector(df, [0.5,.95], "id")
    inspector.generate_reports()