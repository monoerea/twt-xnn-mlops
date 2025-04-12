from concurrent.futures import ThreadPoolExecutor
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Dict, Union, List, Optional
from Pipeline.base import Analysis

import threading
import matplotlib
matplotlib.use('Agg')  # Set before importing pyplot
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
class DataProfiler(Analysis):
    """Basic data profiling and analysis"""
    def __init__(self, name = None):
        super().__init__(name or self.__class__.__name__)

    def transform(self, data: pd.DataFrame, config: Any = None) -> pd.DataFrame:
        config = config or {}
        output_dir = config.get('output_dir')

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        stats = self._basic_stats(data)
        strategy = config.get('strategy','analyze_correlations')
        self._save_stats_report(stats, output_dir)

        if strategy == 'analyze_correlations':
            corr = self._analyze_correlations(data)
            self._save_correlation_report(corr, output_dir)

        elif strategy == 'analyze_distributions':
            profiler = DataDistribution(name='DataDistribution')
            profiler.fit_transform(data, config=config)
            # self._analyze_distributions(data, output_dir)

        elif strategy == 'group_by':
            group = GroupAnalysis()
            group.fit_transform(data, config=config)
        elif strategy  == 'one_sample':
            one_sample = TTestAnalysis()
            result = one_sample.transform(data, config)
            result.to_csv(f"{output_dir}/one_sample.csv")
        else:
            print(f'The{strategy} is not registered.')
        return data

    def _basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics"""
        return {
            'shape': df.shape,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict(),
            'categorical_summary': {
                col: {
                    'unique_values': df[col].nunique(),
                    'top_values': df[col].value_counts().head(3).to_dict()
                }
                for col in df.select_dtypes(exclude='number').columns
            }
        }

    def _save_stats_report(self, stats: Dict[str, Any], output_dir: str):
        """Save basic statistics to CSV files"""

        pd.DataFrame([stats['shape']], columns=['rows', 'columns']).to_csv(
            os.path.join(output_dir, 'data_shape.csv'), index=False
        )

        pd.DataFrame.from_dict(stats['dtypes'], orient='index', columns=['count']).to_csv(
            os.path.join(output_dir, 'data_types.csv')
        )

        pd.DataFrame.from_dict(stats['missing_values'], orient='index', columns=['missing_count']).to_csv(
            os.path.join(output_dir, 'missing_values.csv')
        )

        pd.DataFrame(stats['numeric_summary']).to_csv(
            os.path.join(output_dir, 'numeric_summary.csv')
        )

        cat_summary = []
        for col, values in stats['categorical_summary'].items():
            cat_summary.append({
                'column': col,
                'unique_values': values['unique_values'],
                'top_value': next(iter(values['top_values'])),
                'top_count': next(iter(values['top_values'].values()))
            })
        pd.DataFrame(cat_summary).to_csv(
            os.path.join(output_dir, 'categorical_summary.csv'), index=False
        )

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns"""
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) < 2:
            return {'error': 'Not enough numeric columns for correlation analysis'}

        corr_matrix = df[numeric_cols].corr()
        significant = []

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.5:  # Threshold for significant correlation
                    significant.append({
                        'variable_1': col1,
                        'variable_2': col2,
                        'correlation': corr
                    })

        return {
            'correlation_matrix': corr_matrix,
            'significant_correlations': significant
        }

    def _save_correlation_report(self, corr_data: Dict[str, Any], output_dir: str):
        """Save correlation analysis to CSV files"""

        corr_data['correlation_matrix'].to_csv(os.path.join(output_dir,'correlation_matrix.csv'))

        if corr_data['significant_correlations']:
            pd.DataFrame(corr_data['significant_correlations']).to_csv(
                os.path.join(output_dir, 'significant_correlations.csv'), index=False
            )

    def _analyze_distributions(self, df: pd.DataFrame, output_dir: str):
        """Generate distribution plots for numeric columns"""
        numeric_cols = df.select_dtypes(include='number').columns
        n = len(df)
        width = min(20, 20 / max(8, n // 200))
        height = 4
        for col in numeric_cols:
            plot_path = os.path.join(output_dir, f'distribution_{col}.png')

            plt.figure(figsize=(width, height))
            plt.subplot(1, 2, 1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')

            plt.subplot(1, 2, 2)
            sns.boxplot(df[col].dropna())
            plt.title(f'Boxplot of {col}')

            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        return df
class CorrelationHeatmap(Analysis):
    """
    A class to create correlation heatmaps with various methods and column selection options.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing the data
    cmap : str, optional
        Colormap for the heatmap (default: 'coolwarm')
    figsize : tuple, optional
        Figure size (default: (10, 8))
    annot : bool, optional
        Whether to annotate the heatmap cells (default: True)
    fmt : str, optional
        String formatting for annotations (default: '.2f')
    title : str, optional
        Title for the plot (default: 'Correlation Heatmap')
    """

    def __init__(self, name = None, cmap: str = 'coolwarm', figsize: tuple = (10, 8), annot: bool = True, fmt: str = '.2f', title: str = 'Correlation Heatmap'):
        super().__init__(name or self.__class__.__name__)
        self.data = None
        self.cmap = cmap
        self.figsize = figsize
        self.annot = annot
        self.fmt = fmt
        self.title = title
    def transform(self, data:pd.DataFrame, config: dict):
        self.data = data
        strategy = config.get('strategy','pearson')
        path = config.get('output_dir','Pipeline/analysis/report')
        cols = config.get('cols',data.columns.to_list())
        strategy_method = getattr(self, strategy)
        result = strategy_method(cols = cols)
        result.show()
        result.savefig(fname=f'{path}{strategy}.png', transparent='True')
        return data
    def _select_columns(self, cols: Union[str, List[str], slice, None] = None) -> pd.DataFrame:
        if cols is None:
            return self.data
        elif isinstance(cols, str):
            return self.data[[cols]]
        elif isinstance(cols, list):
            return self.data[cols]
        elif isinstance(cols, slice):
            start = cols.start if cols.start else self.data.columns[0]
            stop = cols.stop if cols.stop else self.data.columns[-1]
            all_cols = self.data.columns
            start_idx = list(all_cols).index(start)
            stop_idx = list(all_cols).index(stop) + 1
            return self.data.iloc[:, start_idx:stop_idx]
        else:
            raise ValueError("cols must be None, str, list of str, or slice")

    def pearson(self, cols: Union[str, List[str], slice, None] = None, **kwargs) -> plt.Figure:
        selected_data = self._select_columns(cols)
        corr = selected_data.corr(method='pearson')
        return self._plot_heatmap(corr, **kwargs)

    def spearman(self, cols: Union[str, List[str], slice, None] = None, **kwargs) -> plt.Figure:
        selected_data = self._select_columns(cols)
        corr = selected_data.corr(method='spearman')
        return self._plot_heatmap(corr, **kwargs)

    def kendall(self, cols: Union[str, List[str], slice, None] = None, **kwargs) -> plt.Figure:
        selected_data = self._select_columns(cols)
        corr = selected_data.corr(method='kendall')
        return self._plot_heatmap(corr, **kwargs)

    def _plot_heatmap(self, corr_matrix: pd.DataFrame, **kwargs) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            corr_matrix,
            cmap=self.cmap,
            annot=self.annot,
            fmt=self.fmt,
            ax=ax,
            **kwargs
        )
        ax.set_title(self.title)
        plt.tight_layout()
        return fig

    def plot_custom(self, method: str = 'pearson', cols: Union[str, List[str], slice, None] = None,
                   **kwargs) -> plt.Figure:
        selected_data = self._select_columns(cols)
        corr = selected_data.corr(method=method)
        return self._plot_heatmap(corr, **kwargs)



class GroupAnalysis(Analysis):
    def __init__(self, name=None):
        super().__init__(name or self.__class__.__name__)
        self.config = {}
        self._plot_lock = threading.Lock()  # For thread-safe plotting

    def _sanitize_name(self, name: str) -> str:
        """Create filesystem-safe names"""
        return re.sub(r'[<>:"/\\|?*]', '_', str(name))

    def _process_single_group(self, df: pd.DataFrame, feat: str, col: str, output_dir: str) -> None:
        """Process one feature-column combination"""
        try:
            grouped = df.groupby(feat)[col].mean().sort_values(ascending=True)

            with self._plot_lock:
                fig, ax = plt.subplots(figsize=(10, 6))
                try:
                    grouped.plot(kind='bar', ax=ax)
                    ax.set_title(f'{col} grouped by {feat}')
                    fig.tight_layout()
                    safe_feat = self._sanitize_name(feat)
                    safe_col = self._sanitize_name(col)
                    os.makedirs(os.path.join(output_dir, safe_feat), exist_ok=True)
                    fig.savefig(os.path.join(output_dir, safe_feat, f'{safe_feat}_by_{safe_col}.png'))
                    return grouped
                finally:
                    plt.close(fig)
        except Exception as e:
            print(f"Skipping {feat}/{col}: {str(e)}")

    def transform(self, df: pd.DataFrame, config: Any) -> pd.DataFrame:
        """Main analysis entry point"""
        self.set_config(config=config)
        output_dir = self.config.get('output_dir', 'analysis/report/group_by')
        features = self.config.get('features', df.columns.tolist())
        from tqdm import tqdm

        try:
            plt.ioff
            with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
                for feat in tqdm(features, desc="Processing groups:"):
                    futures = [
                        executor.submit(
                            self._process_single_group,
                            df, feat, col,
                            os.path.join(output_dir, 'group_by')
                        )
                        for col in df.columns
                        if col != feat
                    ]
                    # Wait for completion (optional)
                    results = pd.DataFrame([f.result() for f in futures])
                    results.to_csv(f"{output_dir}/group_analysis.csv")
            return df
        finally:
            plt.close('all')  # Cleanup all figures
class TTestAnalysis(Analysis):
    def transform(self, data: pd.DataFrame, config: Any = None):
        self.config = config
        strategy = self.config.get('strategy', 'one_sample')
        print(strategy)
        try:
            if strategy == 'one_sample':
                return self.one_sample(data=data)
        except Exception as e:
            self.logger.error(f"Exception at {e}")
    def one_sample(self, data: pd.DataFrame):
        from scipy.stats import ttest_1samp
        stats = []
        features = self.config.get('features', data.columns)
        for feat in features:
            t_stat, p_value = ttest_1samp(data[feat], data[feat].mean())
            stats.append({'feature': feat, 'p_value': p_value, 't_stat': t_stat})
        stats = pd.DataFrame(stats)
        print(stats)
        return stats
class DataDistribution(Analysis):
    def __init__(self, name):
        super().__init__(name or self.__class__.__name__)

    def plot_distributions(self, data, col):
        """Plot histogram and boxplot for a numeric column"""
        fig, axs = plt.subplots(1, 2, figsize=(max(6, min(20, 20 / max(8, len(data) // 200))), 5))
        sns.histplot(data[col].dropna(), kde=True, ax=axs[0]).set_title(f'Distribution of {col}')
        sns.boxplot(x=data[col].dropna(), ax=axs[1]).set_title(f'Boxplot of {col}')
        return fig

    def plot_grouped_box(self, data, num_col, cat_col):
        """Plot a boxplot grouped by a categorical column using adaptive category limit."""
        n_unique = data[cat_col].nunique()
        max_categories = int(np.sqrt(n_unique)) if n_unique > 0 else 0

        if max_categories < 2:
            self.logger.info(f"Skipping {cat_col} (too few unique categories).")
            return None

        top_cats = data[cat_col].value_counts().nlargest(max_categories).index
        filtered = data[data[cat_col].isin(top_cats)]

        order = filtered.groupby(cat_col)[num_col].mean().sort_values().index
        fig, ax = plt.subplots(figsize=(max(6, min(20, 20 / max(8, len(filtered) // 200))), 10))
        sns.boxplot(x=cat_col, y=num_col, data=filtered, ax=ax, order=order)
        ax.set_title(f'{num_col} by {cat_col}')
        ax.tick_params(axis='x', rotation=45)
        return fig
    def plot_categorical_frequency(self, data, cat_col, logger=None, figsize=(10, 6)):
        n_unique = data[cat_col].nunique()
        max_categories = int(np.sqrt(n_unique)) if n_unique > 0 else 0

        if max_categories < 2:
            msg = f"Skipping {cat_col} (too few unique categories: {n_unique})"
            if logger:
                logger.info(msg)
            else:
                print(msg)
            return None

        top_cats = data[cat_col].value_counts().nlargest(max_categories).index
        filtered = data[data[cat_col].isin(top_cats)]

        plt.figure(figsize=figsize)
        fig, ax = plt.subplots(figsize=figsize)
        sns.countplot(x=cat_col, data=filtered, order=top_cats, ax=ax)

        ax.set_title(f"Top {max_categories} Categories in '{cat_col}' (of {n_unique} total)")
        ax.tick_params(axis='x', rotation=45)

        total = len(filtered)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                    f'{height/total:.1%}',
                    ha="center")
        return fig

    def save_plot(self, fig, name, folder):
        pathfile = os.path.join(self.config.get('output_dir', 'analysis/week_3'), folder)
        """Save figure to output directory"""
        os.makedirs(pathfile, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(pathfile, f"{name}.png"))
        plt.close(fig)

    def transform(self, data: pd.DataFrame, config: Any = None) -> pd.DataFrame:
        """Generate all distribution plots"""
        self.config = config or {}
        num_cols = data.select_dtypes(include='number').columns
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        self.logger.info(f"Categorical columns: {cat_cols}")
        for col in num_cols:
            self.save_plot(self.plot_distributions(data, col), f'distribution_{col}', 'dist')
            for cat in cat_cols:
                if freq_fig := self.plot_categorical_frequency(data, cat):
                    self.save_plot(freq_fig, f'frequency_of_{cat}', folder='frequency')
                if fig := self.plot_grouped_box(data, col, cat):
                    self.save_plot(fig, f'{col}_by_{cat}', folder='grouped_box')
        return data