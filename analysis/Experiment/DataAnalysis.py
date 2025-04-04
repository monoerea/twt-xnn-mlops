import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Optional

class CorrelationHeatmap:
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
    
    def __init__(self, data: pd.DataFrame, cmap: str = 'coolwarm', figsize: tuple = (10, 8),
                 annot: bool = True, fmt: str = '.2f', title: str = 'Correlation Heatmap'):
        self.data = data
        self.cmap = cmap
        self.figsize = figsize
        self.annot = annot
        self.fmt = fmt
        self.title = title
        
    def _select_columns(self, cols: Union[str, List[str], slice, None] = None) -> pd.DataFrame:
        """
        Internal method to select columns based on input specification.
        
        Parameters:
        -----------
        cols : str, list of str, slice, or None
            Column selection specification:
            - None: all columns
            - str: single column name
            - list: list of column names
            - slice: column range (e.g., 'col1':'col5')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with selected columns
        """
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
        """
        Create heatmap using Pearson correlation.
        
        Parameters:
        -----------
        cols : str, list of str, slice, or None
            Column selection (default: None - all columns)
        **kwargs
            Additional keyword arguments passed to seaborn.heatmap
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        selected_data = self._select_columns(cols)
        corr = selected_data.corr(method='pearson')
        return self._plot_heatmap(corr, **kwargs)
    
    def spearman(self, cols: Union[str, List[str], slice, None] = None, **kwargs) -> plt.Figure:
        """
        Create heatmap using Spearman correlation.
        
        Parameters:
        -----------
        cols : str, list of str, slice, or None
            Column selection (default: None - all columns)
        **kwargs
            Additional keyword arguments passed to seaborn.heatmap
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        selected_data = self._select_columns(cols)
        corr = selected_data.corr(method='spearman')
        return self._plot_heatmap(corr, **kwargs)
    
    def kendall(self, cols: Union[str, List[str], slice, None] = None, **kwargs) -> plt.Figure:
        """
        Create heatmap using Kendall's tau correlation.
        
        Parameters:
        -----------
        cols : str, list of str, slice, or None
            Column selection (default: None - all columns)
        **kwargs
            Additional keyword arguments passed to seaborn.heatmap
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        selected_data = self._select_columns(cols)
        corr = selected_data.corr(method='kendall')
        return self._plot_heatmap(corr, **kwargs)
    
    def _plot_heatmap(self, corr_matrix: pd.DataFrame, **kwargs) -> plt.Figure:
        """
        Internal method to plot the heatmap.
        
        Parameters:
        -----------
        corr_matrix : pd.DataFrame
            Correlation matrix to plot
        **kwargs
            Additional keyword arguments passed to seaborn.heatmap
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
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
    
    def plot_custom(self, method: str = 'pearson', 
                   cols: Union[str, List[str], slice, None] = None,
                   **kwargs) -> plt.Figure:
        """
        Create heatmap with custom correlation method.
        
        Parameters:
        -----------
        method : str
            Correlation method ('pearson', 'spearman', 'kendall', or callable)
        cols : str, list of str, slice, or None
            Column selection (default: None - all columns)
        **kwargs
            Additional keyword arguments passed to seaborn.heatmap
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        selected_data = self._select_columns(cols)
        corr = selected_data.corr(method=method)
        return self._plot_heatmap(corr, **kwargs)

if __name__ == '__main__':
    df = pd.read_csv('data/processed/cleaned_data.csv')
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