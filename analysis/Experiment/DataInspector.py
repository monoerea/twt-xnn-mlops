import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from scipy.stats import ttest_ind, levene, shapiro
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
        results = self.format_mcar_results(results)
        return results

    def _get_auto_columns(self, inspector):
        missing_pct = inspector.df.isnull().mean()
        excl =  [
            col for col in inspector.df.columns
            if missing_pct[col] > inspector.thresholds[0] and missing_pct[col] < inspector.thresholds[1]
            and not any(exclude_str in col.lower() for exclude_str in inspector.exclude_columns)
            and pd.api.types.is_numeric_dtype(inspector.df[col])
        ]
        return excl

    def _simulate_missing(self, series, pct, random_state=None):
        return series.mask(
        np.random.default_rng(random_state).random(len(series)) < pct
        )

    def _create_comparison(self, original, simulated):
        comparison = pd.concat([
            original.describe().rename('Original'),
            simulated.describe().rename('MCAR')
        ], axis=1)
        comparison['Diff'] = comparison['MCAR'] - comparison['Original']
        comparison['Pct_Change'] = (comparison['Diff'] / comparison['Original']).abs() * 100
        return comparison.round(2)
    def format_mcar_results(self, comparison: dict)-> pd.DataFrame:
        combined = pd.concat(comparison.values(), 
                            keys=comparison.keys(),
                            names=['Feature', 'Metric'])
        combined = combined.reset_index(level='Metric')
        combined = combined.reset_index(drop=False)
        return combined

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, levene, shapiro

class TTestStrategy(AnalysisStrategy):
    def execute(self, inspector, target_cols=None, sample_frac=0.0, random_state=21, **kwargs):
        """Execute MCAR test with comprehensive statistics"""
        df = inspector.df
        df_copy = df.copy()

        # Auto-select target columns if not specified
        if target_cols is None:
            numeric_cols = df.select_dtypes(include='number').columns
            missing_pct = df[numeric_cols].isnull().mean()

            target_cols = [
                col for col in numeric_cols
                if (inspector.thresholds[0] < missing_pct[col] < inspector.thresholds[1] and
                    not any(ex in col.lower() for ex in (inspector.exclude_columns or [])))
            ]

        results = []
        for col in target_cols:
            if col not in df.columns:
                continue

            # Create synthetic missing values
            mask = df[col].sample(frac=sample_frac, random_state=random_state).index
            df_copy.loc[mask, col] = np.nan
            missing_mask = df_copy[col].isna()
            
            # Calculate statistics
            result = self._calculate_stats(df, col, missing_mask)
            if result:
                results.append(result)
        
        return pd.DataFrame(results) if results else pd.DataFrame()

    def _calculate_stats(self, df, target_col, missing_mask):
        """Calculate comprehensive statistics with robust error handling"""
        present = df.loc[~missing_mask, target_col].dropna()
        missing = df.loc[missing_mask, target_col].dropna()
        # Base result with guaranteed fields
        result = {
            'Target': target_col,
            'Missing %': missing_mask.mean(),
            'N Present': len(present),
            'N Missing': len(missing)
        }
        # Central tendency
        result.update(self._calculate_central_tendency(present, missing))

        # Dispersion
        result.update(self._calculate_dispersion(present, missing))

        # Normality tests
        result.update(self._calculate_normality_tests(present, missing))

        # Variance and effect size
        result.update(self._calculate_variance_effect(present, missing))

        # T-test
        result.update(self._calculate_ttest(present, missing))

        return result

    def _calculate_central_tendency(self, present, missing):
        """Calculate mean and median metrics"""
        stats = {}
        stats['Present Mean'] = present.mean() if len(present) > 0 else np.nan
        stats['Missing Mean'] = missing.mean() if len(missing) > 0 else np.nan
        stats['Present Median'] = present.median() if len(present) > 0 else np.nan
        stats['Missing Median'] = missing.median() if len(missing) > 0 else np.nan
        return stats

    def _calculate_dispersion(self, present, missing):
        """Calculate standard deviation and IQR"""
        stats = {}
        if len(present) >= 2:
            stats['Present Std'] = present.std()
            stats['Present IQR'] = present.quantile(0.75) - present.quantile(0.25)
        if len(missing) >= 2:
            stats['Missing Std'] = missing.std()
            stats['Missing IQR'] = missing.quantile(0.75) - missing.quantile(0.25)
        return stats

    def _calculate_normality_tests(self, present, missing):
        """Calculate Shapiro-Wilk normality tests"""
        stats = {}
        if 3 <= len(present) <= 5000:
            stats['Shapiro-Wilk (Present)'] = shapiro(present)[1]
        if 3 <= len(missing) <= 5000:
            stats['Shapiro-Wilk (Missing)'] = shapiro(missing)[1]
        return stats

    def _calculate_variance_effect(self, present, missing):
        """Calculate Levene's test and Cohen's d"""
        stats = {}
        if len(present) >= 2 and len(missing) >= 2:
            stats['Levene p-value'] = levene(present, missing)[1]

            # Properly parenthesized Cohen's d calculation
            numerator = (len(present)-1)*present.std()**2 + (len(missing)-1)*missing.std()**2
            denominator = len(present) + len(missing) - 2
            pooled_var = numerator / denominator
            pooled_std = np.sqrt(pooled_var)

            if pooled_std != 0:
                stats["Cohen's d"] = (present.mean() - missing.mean()) / pooled_std
        return stats

    def _calculate_ttest(self, present, missing):
        """Perform Welch's t-test"""
        stats = {}
        if len(present) >= 2 and len(missing) >= 2:
            try:
                t_stat, p_val = ttest_ind(present, missing, equal_var=False)
                stats.update({'t-stat': t_stat, 'p-value': p_val})
            except:
                pass
        return stats


class OutlierAnalysisStrategy(AnalysisStrategy):
        def execute(self, inspector, columns=None, threshold=3, **kwargs):
            if columns is None:
                numeric_cols = df.select_dtypes(include='number').columns
                missing_pct = df[numeric_cols].isnull().mean()
                cols = columns or [
                    col for col in numeric_cols
                    if (inspector.thresholds[0] < missing_pct[col] < inspector.thresholds[1] and
                        not any(ex in col.lower() for ex in (inspector.exclude_columns or [])))
                ]
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
        'outlier': OutlierAnalysisStrategy()
        }

    def __init__(self, df: pd.DataFrame, thresholds=(0.05, 0.95), exclude_columns=None):
        self.df = df
        self.thresholds = thresholds
        self.exclude_columns = exclude_columns or [
            col for col in inspector.df.select_dtypes(include='number')
            if inspector.thresholds[0] < inspector.df[col].isnull().mean() < inspector.thresholds[1]
            and not any(ex in col.lower() for ex in inspector.exclude_columns)
        ]

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
    def generate_report(self, output_dir="analysis", strategies=None):
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
                summary_path = f"{output_dir}/{strategy_name}_result.csv"
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

        data_summary = pd.DataFrame({
            'Column': self.df.columns,
            'Missing %': (self.df.isnull().mean() * 100).round(2),
            'Data Type': self.df.dtypes
        })
        data_summary.to_csv(f"{output_dir}/data_summary.csv", index=False)
        print("\n === Data Summary ==",data_summary)
        print(f"\n✅ Report generated in {output_dir}")
        return results

# ====================== USAGE & EXTENSION EXAMPLES ======================
if __name__ == "__main__":
    # Sample data
    data =pd.read_csv("data/processed/flattened_status.csv", low_memory=False)
    df = pd.DataFrame(data).drop_duplicates()
    value = True
    if value == True:
        to_remove = df.columns[df.isna().mean() > 0.95]
        df = df.drop(columns=to_remove)

    inspector = DataInspector(
        df=df,
        thresholds=[0.5, 0.95],
        exclude_columns=['id']
    )

    print("=== DEFAULT REPORT ===")
    output_dir = 'analysis/result'
    strategies = [
            ('basic_info', {}),
            ('mcar', {'columns': None, 'pct': 0.05}),
            ('ttest', {'target_cols': None, 'group_col': None, }),
            ('outlier',{'threshold':5}),
            ('missing_heatmap', {'filename': f"{output_dir}/missing_heatmap.png"})
        ]
    inspector.generate_report(output_dir=output_dir, strategies=strategies)