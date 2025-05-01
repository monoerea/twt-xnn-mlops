from concurrent.futures import ThreadPoolExecutor
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Dict, Union, List, Optional

from tqdm import tqdm
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
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

class DataDistribution(Analysis):
    def __init__(self, name: str = None):
        super().__init__(name or self.__class__.__name__)
        self._current_batch = 0
        self._plot_cache = []
        self._configure_plot_style()

    def _configure_plot_style(self):
        """Set consistent style for all plots"""
        plt.rcParams.update({
            'figure.dpi': 120,
            'savefig.dpi': 300,
            'figure.autolayout': True
        })

    def plot_distributions(self, data: pd.DataFrame, col: str) -> Optional[plt.Figure]:
        """Plot histogram and boxplot with adaptive sizing"""
        try:
            fig, axs = plt.subplots(1, 2, figsize=(self._get_fig_size(len(data)), 5))
            sns.histplot(data[col].dropna(), kde=True, ax=axs[0])
            axs[0].set_title(f'Distribution of {col}')

            sns.boxplot(x=data[col].dropna(), ax=axs[1])
            axs[1].set_title(f'Boxplot of {col}')

            return fig
        except Exception as e:
            self.logger.error(f"Failed plotting {col}: {str(e)}")
            return None

    def plot_grouped_box(self, data: pd.DataFrame, num_col: str, cat_col: str) -> Optional[plt.Figure]:
        """Batch-friendly grouped boxplot with smart category limiting"""
        try:
            n_unique = data[cat_col].nunique()
            if n_unique < 2:
                return None

            max_cats = min(15, int(np.sqrt(n_unique))) if n_unique > 15 else n_unique
            top_cats = data[cat_col].value_counts().nlargest(max_cats).index
            filtered = data[data[cat_col].isin(top_cats)]
            fig, ax = plt.subplots(figsize=(self._get_fig_size(len(filtered)), 8))
            order = filtered.groupby(cat_col)[num_col].median().sort_values().index
            sns.boxplot(x=cat_col, y=num_col, data=filtered,
                       ax=ax, order=order, 
                       flierprops={'marker': 'x', 'markersize': 3})
            ax.set_title(f'{num_col[:15]} by {cat_col[:15]}')
            ax.tick_params(axis='x', rotation=45)
            return fig
        except Exception as e:
            self.logger.error(f"Failed grouped boxplot {num_col}x{cat_col}: {str(e)}")
            return None

    def plot_categorical_frequency(self, data: pd.DataFrame, cat_col: str) -> Optional[plt.Figure]:
        """Optimized categorical frequency plot"""
        try:
            n_unique = data[cat_col].nunique()
            if n_unique < 2:
                return None

            max_cats = min(20, int(np.sqrt(n_unique)))
            top_cats = data[cat_col].value_counts().nlargest(max_cats).index
            filtered = data[data[cat_col].isin(top_cats)]

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(x=cat_col, data=filtered, order=top_cats, ax=ax)
            
            ax.set_title(f"Top {max_cats} of {n_unique} {cat_col[:20]} Categories")
            ax.tick_params(axis='x', rotation=45)
            
            # Add percentage annotations only for small categories
            if max_cats <= 10:
                total = len(filtered)
                for p in ax.patches:
                    ax.annotate(f'{p.get_height()/total:.1%}', 
                              (p.get_x() + p.get_width()/2., p.get_height()),
                              ha='center', va='center', xytext=(0, 5),
                              textcoords='offset points')
            return fig
        except Exception as e:
            self.logger.error(f"Failed frequency plot {cat_col}: {str(e)}")
            return None

    def _get_fig_size(self, sample_size: int) -> int:
        """Dynamic figure sizing based on data volume"""
        return max(6, min(20, 20 / (8 + sample_size // 1000)))

    def _process_batch(self, data: pd.DataFrame, num_cols: List[str], cat_cols: List[str], 
                     batch_num: int, batch_size: int, main_pbar: tqdm = None) -> None:
        """Process a single batch of columns with progress tracking"""
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        current_num_cols = num_cols[start_idx:end_idx]

        for num_col in current_num_cols:
            # Distribution plots
            if fig := self.plot_distributions(data, num_col):
                self._plot_cache.append(('dist', f'distribution_{num_col}', fig))
                if main_pbar:
                    main_pbar.update(1)
            
            # Categorical plots
            for cat_col in cat_cols:
                if freq_fig := self.plot_categorical_frequency(data, cat_col):
                    self._plot_cache.append(('frequency', f'frequency_{cat_col}', freq_fig))
                if box_fig := self.plot_grouped_box(data, num_col, cat_col):
                    self._plot_cache.append(('grouped_box', f'{num_col}_by_{cat_col}', box_fig))
                
                if main_pbar:
                    main_pbar.update(1)
                    main_pbar.set_postfix({'Columns': f"{num_col[:15]}... x {cat_col[:15]}..."})

    def _save_all_plots(self):
        """Save all cached plots in batch"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for folder, name, fig in self._plot_cache:
                futures.append(
                    executor.submit(
                        self.save_plot, 
                        fig, 
                        name, 
                        folder
                    )
                )
            for future in tqdm(futures, desc="Saving plots"):
                future.result()
        self._plot_cache = []

    def _get_optimal_sample(self, data: pd.DataFrame, id_col: str = None, 
                      sample_size: int = 1000) -> pd.DataFrame:
        """
        Get optimal sample with:
        1. Most complete rows (least nulls)
        2. Diverse IDs (if id_col provided)
        3. Representative distribution
        """
        if len(data) == 0:
            return pd.DataFrame()

        # Ensure sample_size is within valid range
        sample_size = min(sample_size, len(data))
        if sample_size <= 0:
            return pd.DataFrame()

        # Score rows by completeness
        completeness = data.notna().mean(axis=1)
        
        try:
            if id_col and id_col in data.columns:
                # Get top complete rows per unique ID
                unique_ids = data[id_col].nunique()
                sample_per_id = max(1, sample_size // unique_ids)
                
                sample = (data
                        .assign(_completeness=completeness)
                        .sort_values('_completeness', ascending=False)
                        .groupby(id_col, group_keys=False)
                        .apply(lambda x: x.head(sample_per_id))
                        .nlargest(sample_size, '_completeness')
                        .drop(columns='_completeness'))
            else:
                # Safely get most complete rows
                valid_sample_size = min(sample_size, len(data))
                sample_indices = completeness.nlargest(valid_sample_size).index
                sample = data.loc[sample_indices]
            
            # Ensure we maintain original distributions
            num_cols = data.select_dtypes(include='number').columns
            if len(num_cols) > 0 and len(sample) > 0:
                # Stratify by numerical columns' quartiles
                strat_col = data[num_cols].mean(axis=1).rank(pct=True)
                n_groups = min(5, len(sample))
                sample = (data
                        .loc[sample.index]
                        .groupby(pd.qcut(strat_col[sample.index], n_groups))
                        .apply(lambda x: x.sample(min(len(x), max(1, sample_size//n_groups))))
                        .reset_index(drop=True))
            
            return sample.sample(min(len(sample), sample_size)) if len(sample) > 0 else pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Error in sampling: {str(e)}")
            # Fallback to random sample if optimal sampling fails
            return data.sample(min(len(data), sample_size))

    def transform(self, data: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        """Batch-processed visualization pipeline with smart sampling"""
        self._configure_plot_style()
        self.config = config or {}
        
        # Get sampling parameters
        sample_size = self.config.get('sample_size', min(2000, len(data)))
        id_col = self.config.get('id_column')
        batch_size = self.config.get('batch_size', 5)
        
        # Create optimal sample
        sample = self._get_optimal_sample(data, id_col, sample_size)
        self.logger.info(f"Analyzing sample of {len(sample)}/{len(data)} rows")
        
        # Get column lists from sample
        num_cols = sample.select_dtypes(include='number').columns.tolist()
        cat_cols = sample.select_dtypes(include=['object', 'category']).columns.tolist()

        total_batches = (len(num_cols) + batch_size - 1) // batch_size
        total_operations = len(num_cols) * (1 + len(cat_cols)) + len(cat_cols)
        
        with tqdm(total=total_operations, desc="Overall Progress") as main_pbar:
            for batch_num in range(total_batches):
                self._process_batch(sample, num_cols, cat_cols, batch_num, batch_size, main_pbar)
                self._save_all_plots()
                
        return data

    def save_plot(self, fig: plt.Figure, name: str, folder: str) -> None:
        """Thread-safe plot saving"""
        try:
            path = os.path.join(self.config.get('output_dir', 'analysis'), folder)
            os.makedirs(path, exist_ok=True)
            fig.savefig(
                os.path.join(path, f"{name}.png"),
                bbox_inches='tight',
                dpi=300
            )
            plt.close(fig)
        except Exception as e:
            self.logger.error(f"Failed saving {name}: {str(e)}")