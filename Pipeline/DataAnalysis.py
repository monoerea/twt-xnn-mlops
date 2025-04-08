import os
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, Union, List, Optional
from Pipeline.base import Analysis
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
        self._save_stats_report(stats, output_dir)

        if config.get('strategy','analyze_correlations'):
            corr = self._analyze_correlations(data)
            self._save_correlation_report(corr, output_dir)

        if config.get('strategy','analyze_distributions'):
            self._analyze_distributions(data, output_dir)

        if config.get('strategy','group_by'):
            self.group_by(data, output_dir)

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

        for col in numeric_cols:
            plot_path = os.path.join(output_dir, f'distribution_{col}.png')

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')

            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col].dropna())
            plt.title(f'Boxplot of {col}')

            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
    def group_by(self, df: pd.DataFrame, output_dir: str):
        """Generate bar plots showing grouped aggregations of features.
        
        Args:
            df: Input DataFrame
            output_dir: Directory to save output plots
        """
        import re
        
        features = self.config.get('features', df.columns.tolist())
        
        # Create output directory structure if it doesn't exist
        os.makedirs(os.path.join(output_dir, 'group_by'), exist_ok=True)
        
        for feat in features:
            # Sanitize feature name for directory
            safe_feat = re.sub(r'[<>:"/\\|?*]', '_', str(feat))
            
            # Create feature subdirectory
            feat_dir = os.path.join(output_dir, 'group_by', safe_feat)
            os.makedirs(feat_dir, exist_ok=True)
            
            for col in df.columns:
                if col == feat:  # Skip grouping by same column
                    continue
                    
                try:
                    # Sanitize column name for filename
                    safe_col = re.sub(r'[<>:"/\\|?*]', '_', str(col))
                    
                    # Group and aggregate data
                    grouped = df.groupby(feat)[col].mean()  # Using mean as default aggregation
                    
                    # Create and save plot
                    plt.figure(figsize=(10, 6))
                    grouped.plot(kind='bar')
                    plt.title(f'{col} grouped by {feat}')
                    plt.ylabel(col)
                    plt.tight_layout()
                    
                    filename = f'{safe_feat}_by_{safe_col}.png'
                    plot_path = os.path.join(feat_dir, filename)
                    plt.savefig(plot_path)
                    plt.close()
                    
                except Exception as e:
                    print(f"Error processing {feat} by {col}: {str(e)}")
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

class DataTrends(Analysis):
    """Class that outputs a line chart for the given feature.

    Args:
        Analysis (PipelineStep)
    """
    def __init__(self, name = None):
        super().__init__(name or self.__class__.__name__)
    def transform(self, data, config = None):
        feature = config.get('feature', None)
        if feature not in data.columns:
            return data
        
        return data
if __name__ == '__main__':
    df = pd.read_csv('Pipeline/analysis/cleaned_data.csv')
    heatmap = CorrelationHeatmap(df, title='Feature Correlations')

    # Different ways to select columns and methods:
    # 1. All columns with Pearson correlation
    fig1 = heatmap.pearson()

    patterns = ['created', 'status', 'user']
    cols = [col for col in df.columns if any(pattern in col for pattern in patterns)]

    fig2 = heatmap.spearman(cols=cols)

    # # 3. Column range with Kendall correlation
    # fig3 = heatmap.kendall(cols=slice(200, -1))  # From column B to E

    # # 4. Custom method (same as pearson in this case)
    # fig4 = heatmap.plot_custom(method='pearson', cols=slice(1, 5))

    # # 5. Single column (will show correlation with itself = 1)
    # fig5 = heatmap.pearson(cols=1)

    # # Show one of the figures
    plt.show()