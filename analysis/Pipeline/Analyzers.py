import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any
from Pipeline import PipelineStep

class DataProfiler(PipelineStep):
    """Basic data profiling and analysis"""
    
    def process(self, data: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        config = config or {}
        output_dir = config.get('output_dir', 'reports')
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate basic statistics
        stats = self._basic_stats(data)
        self._save_stats_report(stats, output_dir)
        
        # Generate correlation analysis
        if config.get('analyze_correlations', True):
            corr = self._analyze_correlations(data)
            self._save_correlation_report(corr, output_dir)
        
        # Generate distribution plots
        if config.get('analyze_distributions', True):
            self._analyze_distributions(data, output_dir)
        
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
        # Save shape info
        pd.DataFrame([stats['shape']], columns=['rows', 'columns']).to_csv(
            os.path.join(output_dir, 'data_shape.csv'), index=False
        )
        
        # Save dtypes info
        pd.DataFrame.from_dict(stats['dtypes'], orient='index', columns=['count']).to_csv(
            os.path.join(output_dir, 'data_types.csv')
        )
        
        # Save missing values
        pd.DataFrame.from_dict(stats['missing_values'], orient='index', columns=['missing_count']).to_csv(
            os.path.join(output_dir, 'missing_values.csv')
        )
        
        # Save numeric summary
        pd.DataFrame(stats['numeric_summary']).to_csv(
            os.path.join(output_dir, 'numeric_summary.csv')
        )
        
        # Save categorical summary
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
        print('\n IN DATA PROFILER',df.select_dtypes(include="number"))
        df.to_csv('test.csv')
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
        # Save full correlation matrix
        print(corr_data)
        corr_data['correlation_matrix'].to_csv(os.path.join(output_dir,'correlation_matrix.csv'))
        
        # Save significant correlations
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