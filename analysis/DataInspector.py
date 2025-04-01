import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from scipy.stats import ttest_ind
from tabulate import tabulate
from typing import Dict, List, Optional, Union, Type

class AnalysisStrategy(ABC):
    """Base interface for all analysis strategies"""
    @abstractmethod
    def execute(self, inspector: 'DataInspector', **kwargs) -> Union[pd.DataFrame, Dict]:
        pass

class VisualizationStrategy(ABC):
    """Base interface for visualization strategies"""
    @abstractmethod
    def execute(self, inspector: 'DataInspector', **kwargs) -> Optional[str]:
        pass

# ====================== CONCRETE STRATEGIES ======================
class BasicInfoStrategy(AnalysisStrategy):
    def execute(self, inspector, **kwargs):
        types = inspector.df.dtypes.value_counts()
        return {
            "Shape": inspector.df.shape,
            "Columns": len(inspector.df.columns),
            "Missing Values": inspector.df.isnull().sum().sum(),
            "Duplicates": inspector.df.duplicated().sum(),
            "Data Types": dict(types).values()
        }

class ThresholdMCARStrategy(AnalysisStrategy):
    def execute(self, inspector, columns=None, pct=0.05, random_state=None, **kwargs):
        results = {}
        target_cols = columns if columns else self._get_auto_columns(inspector)

        for col in target_cols:
            if col not in inspector.df.columns:
                continue

            original = inspector.df[col].copy()
            simulated = self._simulate_missing(original, pct, random_state)
            results[col] = self._create_comparison(original, simulated)
            print(results)
        return results

    def _get_auto_columns(self, inspector):
        missing_pct = inspector.df.isnull().mean()
        return [
            col for col in inspector.df.columns
            if missing_pct[col] > inspector.thresholds[0] and missing_pct[col] < inspector.thresholds[1]
            and not any(excl.lower() in col.lower() for excl in inspector.exclude_columns)
            and pd.api.types.is_numeric_dtype(inspector.df[col])
        ]


    def _simulate_missing(self, series, pct, random_state):
        if random_state:
            np.random.seed(random_state)
        mask = np.random.rand(len(series)) < pct
        return series.mask(mask)

    def _create_comparison(self, original, simulated):
        comparison = pd.concat([
            original.describe().rename('Original'),
            simulated.describe().rename('MCAR')
        ], axis=1)
        comparison['Diff'] = comparison['MCAR'] - comparison['Original']
        comparison['Pct_Change'] = (comparison['Diff'] / comparison['Original']).abs() * 100
        return comparison.round(2)

class TTestStrategy(AnalysisStrategy):
    def execute(self, inspector, target_cols=None, **kwargs):
        target_cols = target_cols or inspector.df.select_dtypes(include='number').columns
        results = []
        df_copy = inspector.df.copy()
        for col in target_cols:
            if col in inspector.df.columns:
                mask = inspector.df[col].sample(frac=0.05, random_state=21).index
                df_copy.loc[mask, col] = np.nan
                group_col = df_copy[col].isna().astype(int)
                result = self._perform_ttest(inspector.df, col, group_col)
                if result:
                    results.append(result)

        return pd.DataFrame(results)

    def _perform_ttest(self, df, target_col, group_col):
        present = df[group_col == 0][target_col]
        missing = df[group_col == 1][target_col]

        if len(present) > 1 and len(missing) > 1:
            t_stat, p_val = ttest_ind(present, missing, equal_var=False)
            return {
                'Target': target_col,
                't-stat': t_stat,
                'p-value': p_val,
                'Present Mean': present.mean(),
                'Missing Mean': missing.mean()
            }
        return None


class OutlierAnalysisStrategy(AnalysisStrategy):
        def execute(self, inspector, columns=None, threshold=3, **kwargs):
            cols = columns or inspector.df.select_dtypes(include='number').columns
            results = {}
            for col in cols:
                if col in inspector.df.columns:
                    z_scores = (inspector.df[col] - inspector.df[col].mean()) / inspector.df[col].std()
                    outliers = abs(z_scores) > threshold
                    results[col] = {
                        'outlier_count': outliers.sum(),
                        'outlier_pct': (outliers.mean() * 100).round(2)
                    }
            return pd.DataFrame(results).T

class MissingValueHeatmapStrategy(VisualizationStrategy):
    def execute(self, inspector, filename="missing_heatmap.png", **kwargs):
        plt.figure(figsize=(12, 6))
        sns.heatmap(inspector.df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return filename

# ====================== CORE INSPECTOR CLASS ======================
class DataInspector:
    """Main inspector class with strategy registry"""
    _default_strategies = {
        'basic_info': BasicInfoStrategy(),
        'mcar': ThresholdMCARStrategy(),
        'ttest': TTestStrategy(),
        'missing_heatmap': MissingValueHeatmapStrategy(),
        'outlier_strategy': OutlierAnalysisStrategy()
        }

    def __init__(self, df: pd.DataFrame, thresholds=(0.05, 0.95), exclude_columns=None):
        self.df = df
        self.thresholds = thresholds
        self.exclude_columns = exclude_columns or []
        self._strategies = self._default_strategies.copy()
    def register_strategy(self, name: str, strategy: Union[AnalysisStrategy, VisualizationStrategy],
                        overwrite=False):
        """Register a new strategy"""
        if name in self._strategies and not overwrite:
            raise ValueError(f"Strategy '{name}' already exists. Set overwrite=True to replace")
        self._strategies[name] = strategy
    def execute(self, strategy_name: str, **kwargs):
        """Execute a registered strategy"""
        strategy = self._strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(self._strategies.keys())}")
        return strategy.execute(self, **kwargs)
    def generate_report(self, output_dir="analysis", strategies=None, **kwargs):
        """Generate a customizable report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        strategies = strategies or [
            ('basic_info', {}),
            ('mcar', {'columns': None, 'pct': 0.05}),
            ('ttest', {'columns': None, 'pct': 0.05}),
            ('missing_heatmap', {'filename': f"{output_dir}/missing_heatmap.png"})
        ]

        results = {}
        for strategy_name, strategy_kwargs in strategies:
            print(f"\n=== {strategy_name.upper()} ===")
            try:
                result = self.execute(strategy_name, **strategy_kwargs)
                results[strategy_name] = result
                summary_path = f"{output_dir}/result/{strategy_name}_result.csv"
                if isinstance(result, dict):
                    result = pd.DataFrame([result])
                    result.to_csv(summary_path)
                    print(df)
                elif isinstance(result, pd.DataFrame):
                    print(result)
                    result.to_csv(summary_path)
                else:
                    print(result)
            except Exception as e:
                print(f"⚠️ Error in {strategy_name}: {str.__name__} - {str(e)}")

        summary_path = f"{output_dir}/data_summary.csv"
        pd.DataFrame({
            'Column': self.df.columns,
            'Missing %': (self.df.isnull().mean() * 100).round(2),
            'Data Type': self.df.dtypes
        }).to_csv(summary_path, index=False)
        print(f"\n✅ Report generated in {output_dir}")
        return results

# ====================== USAGE & EXTENSION EXAMPLES ======================
if __name__ == "__main__":
    # Sample data
    data =pd.read_csv("data/processed/flattened_status.csv", low_memory=False)
    df = pd.DataFrame(data).sample(frac=0.95)  # Introduce missing values

    # Initialize inspector
    inspector = DataInspector(
        df=df,
        thresholds=[0.5, 0.95],
        exclude_columns=['id']
    )

    print("=== DEFAULT REPORT ===")
    output_dir = 'analysis'
    strategies = [
            ('basic_info', {}),
            ('mcar', {'columns': None, 'pct': 0.05}),
            ('ttest', {'target_cols': None, 'group_col': None}),
            ('missing_heatmap', {'filename': f"{output_dir}/missing_heatmap.png"})
        ]
    inspector.generate_report(output_dir=output_dir, strategies=strategies)