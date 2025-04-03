import os
import json
from typing import Dict, List, Any, Union, Optional, Callable, Tuple, Type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, levene, shapiro

from Pipeline import Step, Strategy, strategy_registry


class AnalyzerStep(Step):
    """Base class for all analyzer steps"""
    
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Analyze data and return modified dataframe"""
        # Perform analysis but return original data by default
        analysis = self.analyze(data, **kwargs)
        self._store_analysis(analysis)
        return data
    
    def analyze(self, df: pd.DataFrame, **kwargs) -> Any:
        """Analyze the dataframe - override in subclasses"""
        return None
    
    def _store_analysis(self, analysis):
        """Store analysis results"""
        self.analysis_results = analysis


class DataProfiler(AnalyzerStep):
    """Step for comprehensive data profiling"""
    
    def __init__(
        self,
        strategies: List[Union[str, Strategy]] = None,
        output_dir: str = None,
        name: str = "DataProfiler",
        **config
    ):
        """
        Args:
            strategies: List of strategy names or instances for analysis
            output_dir: Directory to save analysis results
            name: Name for this analyzer
            config: Additional configuration
        """
        super().__init__(name=name, **config)
        self.strategies = strategies or []
        self.output_dir = output_dir
        self._resolve_strategies()
        self.results = {}
        
    def _resolve_strategies(self):
        """Resolve strategy strings to instances"""
        resolved = []
        for strategy in self.strategies:
            if isinstance(strategy, str):
                resolved.append(strategy_registry.get(strategy))
            else:
                resolved.append(strategy)
        self.strategies = resolved
        
    def add_strategy(self, strategy: Union[str, Strategy]) -> 'DataProfiler':
        """Add an analysis strategy"""
        if isinstance(strategy, str):
            strategy = strategy_registry.get(strategy)
        self.strategies.append(strategy)
        return self
            
    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run all analysis strategies"""
        if not self.strategies:
            self.logger.info("No analysis strategies configured, running basic profile")
            return self._basic_profile(df)
            
        results = {}
        
        for strategy in self.strategies:
            try:
                strategy_name = strategy.name
                
                # Get strategy-specific config from kwargs
                strategy_kwargs = kwargs.get(strategy_name, {})
                
                self.logger.info(f"Applying {strategy_name} analysis strategy")
                result = strategy.execute(df, **strategy_kwargs)
                results[strategy_name] = result
                
                # Save results if output directory specified
                if self.output_dir:
                    self._save_result(strategy_name, result)
                
            except Exception as e:
                self.logger.error(f"Error applying {strategy.name}: {str(e)}")
                results[strategy_name] = {"error": str(e)}
                
        self.results = results
        return results
        
    def _basic_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic profile of the dataframe"""
        profile = {
            "shape": df.shape,
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing_values": {
                "total": df.isnull().sum().sum(),
                "by_column": df.isnull().sum().to_dict()
            },
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric columns summary
        for col in df.select_dtypes(include='number').columns:
            profile["numeric_summary"][col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "25%": df[col].quantile(0.25),
                "50%": df[col].median(),
                "75%": df[col].quantile(0.75),
                "max": df[col].max(),
                "missing": df[col].isnull().sum()
            }
            
        # Categorical columns summary
        for col in df.select_dtypes(exclude='number').columns:
            profile["categorical_summary"][col] = {
                "unique_values": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict(),
                "missing": df[col].isnull().sum()
            }
            
        return profile
        
    def _save_result(self, strategy_name: str, result: Any) -> None:
        """Save analysis result to file"""
        if not self.output_dir:
            return
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        if isinstance(result, pd.DataFrame):
            result.to_csv(f"{self.output_dir}/{strategy_name}.csv", index=True)
        elif isinstance(result, dict):
            with open(f"{self.output_dir}/{strategy_name}.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)
        elif isinstance(result, str) and result.endswith(('.png', '.jpg')):
            # Result is already a saved file path
            pass
        else:
            # Try to save as string representation
            with open(f"{self.output_dir}/{strategy_name}.txt", 'w') as f:
                f.write(str(result))


class DynamicAnalyzer(AnalyzerStep):
    """Generic analyzer with dynamically configurable operations"""
    
    def __init__(
        self, 
        operations: List[Dict] = None,
        output_dir: str = None,
        name: str = "DynamicAnalyzer", 
        **config
    ):
        """
        Args:
            operations: List of operation configs, each with 'type' and params
            output_dir: Directory to save analysis results
            name: Name for this analyzer
            config: Additional configuration
        """
        super().__init__(name=name, **config)
        self.operations = operations or []
        self.output_dir = output_dir
        self.results = {}
        
    def add_operation(self, op_type: str, **params) -> 'DynamicAnalyzer':
        """Add an operation to the analyzer"""
        self.operations.append({
            'type': op_type,
            **params
        })
        return self
        
    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Apply all operations to analyze the dataframe"""
        results = {}
        
        for i, operation in enumerate(self.operations):
            op_type = operation['type']
            
            # Extract parameters for this operation
            params = {k: v for k, v in operation.items() if k != 'type'}
            
            # Override params with those from kwargs if provided
            if op_type in kwargs:
                params.update(kwargs[op_type])
                
            self.logger.info(f"Applying analysis operation {i+1}: {op_type}")
            
            try:
                # Handle different operation types
                if op_type == 'correlation':
                    result = self._analyze_correlation(df, **params)
                elif op_type == 'distribution':
                    result = self._analyze_distribution(df, **params)
                elif op_type == 'missing_pattern':
                    result = self._analyze_missing_pattern(df, **params)
                elif op_type == 'outliers':
                    result = self._analyze_outliers(df, **params)
                elif op_type == 'feature_importance':
                    result = self._analyze_feature_importance(df, **params)
                elif op_type == 'custom':
                    # Custom operation using provided function
                    func = params.pop('func', None)
                    if func and callable(func):
                        result = func(df, **params)
                    else:
                        self.logger.warning(f"Invalid custom function for operation {i+1}")
                        continue
                else:
                    self.logger.warning(f"Unknown operation type: {op_type}")
                    continue
                
                # Store operation result
                results[op_type] = result
                
                # Save results if output directory specified
                if self.output_dir and result is not None:
                    self._save_result(op_type, result)
            
            except Exception as e:
                self.logger.error(f"Error in operation {i+1} ({op_type}): {str(e)}")
                results[op_type] = {"error": str(e)}
                
        self.results = results
        return results
        
    def _analyze_correlation(self, df: pd.DataFrame, columns=None, method='pearson', 
                             min_corr=0.5, plot=False, **kwargs) -> Dict[str, Any]:
        """Analyze correlations between columns"""
        # Get numeric columns
        if columns is None:
            columns = df.select_dtypes(include='number').columns.tolist()
            
        if len(columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}
            
        # Calculate correlation matrix
        corr_matrix = df[columns].corr(method=method)
        
        # Extract significant correlations
        significant = {}
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) >= min_corr:
                    significant[f"{col1}_{col2}"] = corr_value
                    
        # Create plot if requested
        plot_path = None
        if plot and self.output_dir:
            plot_path = os.path.join(self.output_dir, f"correlation_heatmap.png")
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm', fmt=".2f")
            plt.title(f"Correlation Matrix ({method})")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
        return {
            "correlation_matrix": corr_matrix,
            "significant_correlations": significant,
            "method": method,
            "plot_path": plot_path
        }
        
    def _analyze_distribution(self, df: pd.DataFrame, columns=None, bins=20, 
                             plot=False, max_plots=10, **kwargs) -> Dict[str, Any]:
        """Analyze distributions of columns"""
        result = {}
        
        # Get columns to analyze
        if columns is None:
            columns = df.select_dtypes(include='number').columns.tolist()
            if len(columns) > max_plots:
                self.logger.info(f"Limiting analysis to {max_plots} out of {len(columns)} columns")
                columns = columns[:max_plots]
                
        # Analyze each column
        for col in columns:
            if col not in df.columns:
                continue
                
            # Calculate distribution statistics
            stats = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "skew": df[col].skew(),
                "kurtosis": df[col].kurtosis(),
                "min": df[col].min(),
                "max": df[col].max(),
                "q1": df[col].quantile(0.25),
                "q3": df[col].quantile(0.75)
            }
            
            # Normality test
            if len(df[col].dropna()) >= 3:
                stats["shapiro_test"] = {
                    "statistic": shapiro(df[col].dropna())[0],
                    "p_value": shapiro(df[col].dropna())[1]
                }
                
            # Create plot if requested
            plot_path = None
            if plot and self.output_dir:
                plot_path = os.path.join(self.output_dir, f"distribution_{col}.png")
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                sns.histplot(df[col].dropna(), kde=True, bins=bins)
                plt.axvline(stats["mean"], color='red', linestyle='--', label='Mean')
                plt.axvline(stats["median"], color='green', linestyle='-.', label='Median')
                plt.title(f"Distribution of {col}")
                plt.legend()
                
                plt.subplot(1, 2, 2)
                sns.boxplot(x=df[col].dropna())
                plt.title(f"Boxplot of {col}")
                
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
            stats["plot_path"] = plot_path
            result[col] = stats
            
        return result
        
    def _analyze_missing_pattern(self, df: pd.DataFrame, columns=None, 
                               plot=False, **kwargs) -> Dict[str, Any]:
        """Analyze missing value patterns"""
        # Get columns to analyze
        if columns is None:
            columns = df.columns.tolist()
            
        # Create missing mask
        missing_mask = df[columns].isnull()
        
        # Calculate missing statistics
        missing_counts = missing_mask.sum()
        missing_percent = (missing_counts / len(df) * 100).round(2)
        
        # Find cooccurrence of missing values
        cooccurrence = {}
        if len(columns) > 1:
            for i, col1 in enumerate(columns):
                for col2 in columns[i+1:]:
                    both_missing = ((missing_mask[col1]) & (missing_mask[col2])).sum()
                    if both_missing > 0:
                        cooccurrence[f"{col1}_{col2}"] = {
                            "count": int(both_missing),
                            "percent": round(both_missing / len(df) * 100, 2)
                        }
        
        # Create plot if requested
        plot_path = None
        if plot and self.output_dir:
            plot_path = os.path.join(self.output_dir, f"missing_heatmap.png")
            plt.figure(figsize=(12, 8))
            sns.heatmap(missing_mask, cbar=False, cmap='viridis')
            plt.title("Missing Value Patterns")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
        return {
            "missing_counts": missing_counts.to_dict(),
            "missing_percent": missing_percent.to_dict(),
            "cooccurrence": cooccurrence,
            "plot_path": plot_path
        }
        
    def _analyze_outliers(self, df: pd.DataFrame, columns=None, method='zscore',
                        threshold=3, plot=False, **kwargs) -> Dict[str, Any]:
        """Analyze outliers in the data"""
        result = {}
        
        # Get columns to analyze
        if columns is None:
            columns = df.select_dtypes(include='number').columns.tolist()
            
        # Analyze each column
        for col in columns:
            if col not in df.columns:
                continue
                
            outliers = {}
            if method == 'zscore':
                # Z-score method
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:  # Avoid division by zero
                    z_scores = (df[col] - mean) / std
                    outlier_mask = abs(z_scores) > threshold
                    outliers = {
                        "count": int(outlier_mask.sum()),
                        "percent": round(outlier_mask.mean() * 100, 2),
                        "indices": outlier_mask[outlier_mask].index.tolist()
                    }
            elif method == 'iqr':
                # IQR method
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers = {
                    "count": int(outlier_mask.sum()),
                    "percent": round(outlier_mask.mean() * 100, 2),
                    "indices": outlier_mask[outlier_mask].index.tolist(),
                    "bounds": {"lower": lower_bound, "upper": upper_bound}
                }
                
            # Create plot if requested
            plot_path = None
            if plot and self.output_dir and outliers.get("count", 0) > 0:
                plot_path = os.path.join(self.output_dir, f"outliers_{col}.png")
                plt.figure(figsize=(10, 6))
                
                plt.scatter(
                    range(len(df)), 
                    df[col], 
                    c=['red' if x else 'blue' for x in outlier_mask],
                    alpha=0.5
                )
                plt.title(f"Outlier Detection for {col} ({outliers['count']} outliers)")
                plt.ylabel(col)
                plt.xlabel("Index")
                
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
            outliers["plot_path"] = plot_path
            result[col] = outliers
            
        return result
        
    def _analyze_feature_importance(self, df: pd.DataFrame, target=None, features=None,
                                   method='random_forest', **kwargs) -> Dict[str, Any]:
        """Analyze feature importance using ML models"""
        if target is None or target not in df.columns:
            return {"error": "Target column must be specified and exist in the dataframe"}
            
        # Get features
        if features is None:
            features = [col for col in df.select_dtypes(include='number').columns if col != target]
            
        if not features:
            return {"error": "No numeric feature columns available"}
            
        try:
            # Only import ML libraries if needed
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            # Prepare data
            X = df[features].fillna(df[features].median())
            y = df[target].fillna(df[target].median() if df[target].dtype.kind in 'fc' else df[target].mode()[0])
            
            # Determine task type (regression vs classification)
            is_regression = df[target].dtype.kind in 'fc'
            
            # Create model based on method
            if method == 'random_forest':
                if is_regression:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                return {"error": f"Unsupported method: {method}"}
                
            # Fit model and get feature importance
            model.fit(X, y)
            importance = model.feature_importances_
            
            # Create result
            feature_importance = dict(sorted(
                zip(features, importance),
                key=lambda x: x[1],
                reverse=True
            ))
            
            # Create plot if output directory specified
            plot_path = None
            if self.output_dir:
                plot_path = os.path.join(self.output_dir, f"feature_importance.png")
                plt.figure(figsize=(10, 8))
                
                # Plot feature importance
                importance_df = pd.DataFrame(list(feature_importance.items()), 
                                            columns=['Feature', 'Importance'])
                importance_df = importance_df.sort_values('Importance')
                
                plt.barh(importance_df['Feature'], importance_df['Importance'])
                plt.title(f"Feature Importance for {target}")
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
            return {
                "feature_importance": feature_importance,
                "method": method,
                "plot_path": plot_path
            }
            
        except ImportError:
            return {"error": "scikit-learn not installed"}
        except Exception as e:
            return {"error": str(e)}
    
    def _save_result(self, operation_name: str, result: Any) -> None:
        """Save analysis result to file"""
        if not self.output_dir:
            return
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        if isinstance(result, pd.DataFrame):
            result.to_csv(f"{self.output_dir}/{operation_name}.csv", index=True)
        elif isinstance(result, dict):
            # Filter out non-serializable objects
            serializable_result = {}
            for k, v in result.items():
                if k == 'correlation_matrix' and isinstance(v, pd.DataFrame):
                    # Convert DataFrame to dict for JSON serialization
                    serializable_result[k] = v.to_dict()
                elif isinstance(v, (str, int, float, list, dict, bool)) or v is None:
                    serializable_result[k] = v
                else:
                    serializable_result[k] = str(v)
                    
            with open(f"{self.output_dir}/{operation_name}.json", 'w') as f:
                json.dump(serializable_result, f, indent=2, default=str)


# Register analysis strategies for the profiler
@strategy_registry.register
class CorrelationAnalysisStrategy(Strategy):
    """Strategy for correlation analysis"""
    
    def execute(self, data: pd.DataFrame, columns=None, method='pearson', 
                min_corr=0.5, plot=False, **kwargs) -> Dict[str, Any]:
        """Analyze correlations between columns"""
        # Get numeric columns
        if columns is None:
            columns = data.select_dtypes(include='number').columns.tolist()
            
        if len(columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}
            
        # Calculate correlation matrix
        corr_matrix = data[columns].corr(method=method)
        
        # Extract significant correlations
        significant = {}
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) >= min_corr:
                    significant[f"{col1}_{col2}"] = corr_value
                    
        # Create plot if requested
        plot_path = None
        if plot:
            plot_path = "correlation_heatmap.png"
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm', fmt=".2f")
            plt.title(f"Correlation Matrix ({method})")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
        return {
            "correlation_matrix": corr_matrix,
            "significant_correlations": significant,
            "method": method,
            "plot_path": plot_path
        }


@strategy_registry.register
class TimeSeriesAnalysisStrategy(Strategy):
    """Strategy for time series analysis"""
    
    def execute(self, data: pd.DataFrame, date_column=None, value_columns=None, 
                freq=None, plot=False, **kwargs) -> Dict[str, Any]:
        """Analyze time series data"""
        # Check if date column exists
        if date_column is None:
            # Try to detect datetime columns
            date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
            if not date_cols:
                return {"error": "No date column specified or detected"}
            date_column = date_cols[0]
            
        # Ensure date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            try:
                data[date_column] = pd.to_datetime(data[date_column])
            except:
                return {"error": f"Could not convert {date_column} to datetime"}
                
        # Get value columns
        if value_columns is None:
            value_columns = data.select_dtypes(include='number').columns.tolist()
            if date_column in value_columns:
                value_columns.remove(date_column)
                
        if not value_columns:
            return {"error": "No numeric columns for time series analysis"}
            
        # Sort by date
        data_sorted = data.sort_values(date_column)
        
        # Create result dictionary
        result = {
            "date_column": date_column,
            "value_columns": value_columns,
            "date_range": {
                "start": data_sorted[date_column].min(),
                "end": data_sorted[date_column].max(),
                "periods": len(data_sorted)
            },
            "metrics": {}
        }
        
        # Calculate time series metrics for each value column
        for col in value_columns:
            # Create time series
            ts = data_sorted.set_index(date_column)[col]
            
            # Resample if frequency provided
            if freq:
                ts = ts.resample(freq).mean()
                
            # Calculate metrics
            metrics = {
                "mean": ts.mean(),
                "std": ts.std(),
                "trend": None,  # Will calculate below
                "seasonality": None,  # Will calculate below
                "stationarity": None  # Will calculate below
            }
            
            # Calculate trend (simple moving average)
            try:
                window = min(30, len(ts) // 3) if len(ts) > 3 else 1
                metrics["trend"] = {
                    "method": f"SMA({window})",
                    "start": ts.rolling(window=window).mean().iloc[window],
                    "end": ts.rolling(window=window).mean().iloc[-1],
                    "change": ts.rolling(window=window).mean().iloc[-1] - 
                             ts.rolling(window=window).mean().iloc[window]
                }
            except:
                metrics["trend"] = {"error": "Could not calculate trend"}
                
            # Calculate seasonality (autocorrelation)
            try:
                from pandas.plotting import autocorrelation_plot
                import io
                from PIL import Image
                
                # Create autocorrelation plot in memory
                if plot:
                    plt.figure()
                    autocorrelation_plot(ts.dropna())
                    plt.title(f"Autocorrelation for {col}")
                    
                    # Save to buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    plt.close()
                    
                    # Save plot
                    plot_path = f"autocorrelation_{col}.png"
                    with open(plot_path, 'wb') as f:
                        f.write(buf.getvalue())
                        
                    metrics["seasonality"] = {"plot_path": plot_path}
            except:
                metrics["seasonality"] = {"error": "Could not calculate seasonality"}
                
            # Test for stationarity
            try:
                from statsmodels.tsa.stattools import adfuller
                
                # Run Augmented Dickey-Fuller test
                result = adfuller(ts.dropna())
                metrics["stationarity"] = {
                    "test": "ADF",
                    "statistic": result[0],
                    "p_value": result[1],
                    "is_stationary": result[1] < 0.05
                }
            except:
                metrics["stationarity"] = {"error": "Could not test stationarity"}
                
            result["metrics"][col] = metrics
            
        # Create plot if requested
        plot_path = None
        if plot:
            plot_path = "time_series.png"
            plt.figure(figsize=(12, 6 * len(value_columns)))
            
            for i, col in enumerate(value_columns):
                plt.subplot(len(value_columns), 1, i+1)
                plt.plot(data_sorted[date_column], data_sorted[col])
                plt.title(f"Time Series: {col}")
                plt.xlabel(date_column)
                plt.ylabel(col)
                
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
        result["plot_path"] = plot_path
        return result


class AnalysisReport:
    """Class for generating comprehensive data analysis reports"""
    
    def __init__(self, output_dir: str = "analysis_report"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate(self, df: pd.DataFrame, title: str = "Data Analysis Report",
                include_sections: List[str] = None) -> str:
        """Generate full analysis report"""
        sections = include_sections or [
            "basic_stats", "missing_values", "distributions", 
            "correlations", "outliers"
        ]
        
        # Create report sections
        report_sections = {}
        
        # Basic statistics
        if "basic_stats" in sections:
            report_sections["basic_stats"] = self._generate_basic_stats(df)
            
        # Missing values
        if "missing_values" in sections:
            report_sections["missing_values"] = self._generate_missing_values(df)
            
        # Distributions
        if "distributions" in sections:
            report_sections["distributions"] = self._generate_distributions(df)
            
        # Correlations
        if "correlations" in sections:
            report_sections["correlations"] = self._generate_correlations(df)
            
        # Outliers
        if "outliers" in sections:
            report_sections["outliers"] = self._generate_outliers(df)
            
        # Write report to HTML
        html_path = os.path.join(self.output_dir, "report.html")
        self._write_html_report(report_sections, html_path, title)
        
        return html_path
        
    def _generate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic statistics"""
        numeric_summary = df.describe().to_html()
        
        # Get categorical columns stats
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_summary = None
        if len(cat_cols) > 0:
            cat_df = pd.DataFrame({
                'Column': cat_cols,
                'Unique Values': [df[col].nunique() for col in cat_cols],
                'Most Common': [df[col].value_counts().index[0] if len(df[col].dropna()) > 0 else None for col in cat_cols],
                'Most Common Count': [df[col].value_counts().iloc[0] if len(df[col].dropna()) > 0 else 0 for col in cat_cols]
            })
            cat_summary = cat_df.to_html(index=False)
        
        return {
            "shape": df.shape,
            "numeric_summary": numeric_summary,
            "categorical_summary": cat_summary
        }
        
    def _generate_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate missing values analysis"""
        # Calculate missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        # Create missing values table
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df.sort_values('Missing Count', ascending=False)
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        # Create plot
        plot_path = None
        if len(missing_df) > 0:
            plot_path = os.path.join(self.output_dir, "missing_values.png")
            plt.figure(figsize=(10, 6))
            
            plt.barh(missing_df['Column'], missing_df['Missing %'])
            plt.xlabel('Missing %')
            plt.title('Missing Values by Column')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
        return {
            "missing_table": missing_df.to_html(index=False) if len(missing_df) > 0 else None,
            "plot_path": plot_path
        }
        
    def _generate_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate distribution plots"""
        num_cols = df.select_dtypes(include='number').columns
        
        # Limit to 10 columns for visualizations
        if len(num_cols) > 10:
            num_cols = num_cols[:10]
            
        plots = []
        for col in num_cols:
            plot_path = os.path.join(self.output_dir, f"dist_{col}.png")
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col].dropna())
            plt.title(f"Boxplot of {col}")
            
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            plots.append({
                "column": col,
                "plot_path": plot_path
            })
            
        return {
            "distribution_plots": plots
        }
        
    def _generate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation analysis"""
        num_cols = df.select_dtypes(include='number').columns
        
        if len(num_cols) < 2:
            return {"error": "Not enough numeric columns for correlation analysis"}
            
        # Calculate correlation matrix
        corr_matrix = df[num_cols].corr()
        
        # Create heatmap
        plot_path = os.path.join(self.output_dir, "correlation.png")
        plt.figure(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        # Extract significant correlations
        significant = []
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) >= 0.5:  # Threshold for 'significant'
                    significant.append({
                        "variables": f"{col1} vs {col2}",
                        "correlation": round(corr_value, 3)
                    })
                    
        significant_df = pd.DataFrame(significant).sort_values('correlation', ascending=False)
        
        return {
            "plot_path": plot_path,
            "significant_table": significant_df.to_html(index=False) if len(significant_df) > 0 else None
        }
        
    def _generate_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate outliers analysis"""
        num_cols = df.select_dtypes(include='number').columns
        
        outlier_results = []
        for col in num_cols:
            # Calculate z-scores
            z_scores = abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > 3
            
            # Only include columns with outliers
            if outliers.sum() > 0:
                outlier_results.append({
                    "column": col,
                    "count": int(outliers.sum()),
                    "percentage": round(outliers.mean() * 100, 2)
                })
                
                # Create plot
                plot_path = os.path.join(self.output_dir, f"outliers_{col}.png")
                plt.figure(figsize=(10, 6))
                
                plt.scatter(
                    range(len(df)), 
                    df[col], 
                    c=['red' if x else 'blue' for x in outliers],
                    alpha=0.5
                )
                plt.title(f"Outlier Detection for {col} ({outliers.sum()} outliers)")
                plt.ylabel(col)
                plt.xlabel("Index")
                
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
                outlier_results[-1]["plot_path"] = plot_path
                
        outlier_df = pd.DataFrame(outlier_results)
        
        return {
            "outlier_table": outlier_df.to_html(index=False) if len(outlier_df) > 0 else None,
            "outlier_results": outlier_results
        }
        
    def _write_html_report(self, sections: Dict[str, Any], output_path: str, title: str) -> None:
        """Write HTML report with all sections"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="report-date">Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        """
        
        # Basic stats section
        if "basic_stats" in sections:
            html_content += """
            <div class="section">
                <h2>1. Basic Statistics</h2>
            """
            
            basic_stats = sections["basic_stats"]
            html_content += f"<p>Data Shape: {basic_stats['shape'][0]} rows, {basic_stats['shape'][1]} columns</p>"
            
            html_content += "<h3>Numeric Columns</h3>"
            html_content += basic_stats["numeric_summary"]
            
            if basic_stats["categorical_summary"]:
                html_content += "<h3>Categorical Columns</h3>"
                html_content += basic_stats["categorical_summary"]
                
            html_content += "</div>"
            
        # Missing values section
        if "missing_values" in sections:
            html_content += """
            <div class="section">
                <h2>2. Missing Values Analysis</h2>
            """
            
            missing_values = sections["missing_values"]
            if missing_values.get("missing_table"):
                html_content += missing_values["missing_table"]
            else:
                html_content += "<p>No missing values found in the dataset.</p>"
                
            if missing_values.get("plot_path"):
                html_content += f"<img src='{os.path.basename(missing_values['plot_path'])}' alt='Missing Values'>"
                
            html_content += "</div>"
            
        # Distributions section
        if "distributions" in sections:
            html_content += """
            <div class="section">
                <h2>3. Distribution Analysis</h2>
            """
            
            distributions = sections["distributions"]
            for plot in distributions.get("distribution_plots", []):
                html_content += f"<h3>Distribution of {plot['column']}</h3>"
                html_content += f"<img src='{os.path.basename(plot['plot_path'])}' alt='Distribution of {plot['column']}'>"
                
            html_content += "</div>"
            
        # Correlations section
        if "correlations" in sections:
            html_content += """
            <div class="section">
                <h2>4. Correlation Analysis</h2>
            """
            
            correlations = sections["correlations"]
            if correlations.get("error"):
                html_content += f"<p>{correlations['error']}</p>"
            else:
                html_content += f"<img src='{os.path.basename(correlations['plot_path'])}' alt='Correlation Matrix'>"
                
                if correlations.get("significant_table"):
                    html_content += "<h3>Significant Correlations</h3>"
                    html_content += correlations["significant_table"]
                else:
                    html_content += "<p>No significant correlations found.</p>"
                    
            html_content += "</div>"
            
        # Outliers section
        if "outliers" in sections:
            html_content += """
            <div class="section">
                <h2>5. Outlier Analysis</h2>
            """
            
            outliers = sections["outliers"]
            if outliers.get("outlier_table"):
                html_content += outliers["outlier_table"]
                
                for result in outliers.get("outlier_results", []):
                    if "plot_path" in result:
                        html_content += f"<h3>Outliers in {result['column']}</h3>"
                        html_content += f"<img src='{os.path.basename(result['plot_path'])}' alt='Outliers in {result['column']}'>"
            else:
                html_content += "<p>No significant outliers found using the z-score method.</p>"
                
            html_content += "</div>"
            
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)


class MissingValueAnalyzer(AnalyzerStep):
    """Step for analyzing missing values with MCAR tests"""
    
    def __init__(
        self,
        target_cols: List[str] = None,
        threshold: float = 0.05,
        run_littles_test: bool = True,
        output_dir: str = None,
        name: str = "MissingValueAnalyzer",
        **config
    ):
        """
        Args:
            target_cols: Columns to analyze, if None autodetects columns with missing values
            threshold: Significance threshold for tests
            run_littles_test: Whether to run Little's MCAR test if available
            output_dir: Directory to save analysis results
            name: Name for this analyzer
            config: Additional configuration
        """
        super().__init__(name=name, **config)
        self.target_cols = target_cols
        self.threshold = threshold
        self.run_littles_test = run_littles_test
        self.output_dir = output_dir
        
    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze missing value patterns"""
        # Override target columns if provided in kwargs
        target_cols = kwargs.get('target_cols', self.target_cols)
        
        # Auto-detect columns with missing values if not specified
        if target_cols is None:
            missing_counts = df.isnull().sum()
            target_cols = missing_counts[missing_counts > 0].index.tolist()
            
        if not target_cols:
            return {"message": "No columns with missing values detected"}
            
        results = {
            "missing_summary": self._missing_summary(df, target_cols),
            "t_tests": self._run_t_tests(df, target_cols),
            "patterns": self._analyze_patterns(df, target_cols)
        }
        
        # Run Little's MCAR test if requested and available
        if self.run_littles_test:
            littles_test = self._run_littles_test(df, target_cols)
            if littles_test:
                results["littles_test"] = littles_test
                
        # Create visualizations if output directory provided
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            results["visualizations"] = self._create_visualizations(df, target_cols)
            
        return results
        
    def _missing_summary(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Generate missing data summary statistics"""
        missing_counts = df[columns].isnull().sum()
        missing_percent = (missing_counts / len(df) * 100).round(2)
        
        return {
            "counts": missing_counts.to_dict(),
            "percentages": missing_percent.to_dict(),
            "total_cells": len(df) * len(columns),
            "missing_cells": missing_counts.sum(),
            "missing_percent_total": (missing_counts.sum() / (len(df) * len(columns)) * 100).round(2)
        }
        
    def _run_t_tests(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Run t-tests to compare distributions with/without missing values"""
        results = {}
        
        for col in columns:
            # Skip if no missing values
            if df[col].isnull().sum() == 0:
                continue
                
            col_results = {}
            # For each other column, compare groups
            for other_col in df.select_dtypes(include='number').columns:
                if other_col == col or df[other_col].isnull().sum() > 0:
                    continue
                    
                # Create groups
                missing_mask = df[col].isnull()
                group_missing = df.loc[missing_mask, other_col].dropna()
                group_present = df.loc[~missing_mask, other_col].dropna()
                
                # Skip if groups too small
                if len(group_missing) < 2 or len(group_present) < 2:
                    continue
                    
                try:
                    # Run t-test
                    t_stat, p_val = ttest_ind(group_present, group_missing, equal_var=False)
                    
                    # Calculate effect size (Cohen's d)
                    n1, n2 = len(group_present), len(group_missing)
                    s1, s2 = group_present.std(), group_missing.std()
                    
                    # Pooled standard deviation
                    s_pooled = np.sqrt(((n1-1) * s1**2 + (n2-1) * s2**2) / (n1 + n2 - 2))
                    
                    # Cohen's d
                    d = (group_present.mean() - group_missing.mean()) / s_pooled if s_pooled > 0 else 0
                    
                    col_results[other_col] = {
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "significant": p_val < self.threshold,
                        "cohens_d": d,
                        "n_missing": len(group_missing),
                        "n_present": len(group_present)
                    }
                except Exception as e:
                    pass
                    
            if col_results:
                results[col] = col_results
                
        return results
        
    def _analyze_patterns(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Analyze missing value patterns"""
        # Create missing indicator matrix
        missing_matrix = df[columns].isnull()
        
        # Find unique patterns
        pattern_counts = {}
        for _, row in missing_matrix.iterrows():
            pattern = tuple(row)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        # Convert to readable format
        patterns = []
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            patterns.append({
                "pattern": "".join(["1" if x else "0" for x in pattern]),
                "count": count,
                "percentage": round(count / len(df) * 100, 2)
            })
            
        return {
            "unique_patterns": len(patterns),
            "patterns": patterns
        }
        
    def _run_littles_test(self, df: pd.DataFrame, columns: List[str]) -> Optional[Dict[str, Any]]:
        """Run Little's MCAR test if available"""
        try:
            from missingpy import MissingData
            
            # Create subset with only target columns
            subset = df[columns].copy()
            
            # Initialize MissingData
            md = MissingData(subset)
            
            # Perform Little's test
            result = md.test_mcar()
            
            return {
                "statistic": result.statistic,
                "p_value": result.pvalue,
                "dof": result.dof,
                "is_mcar": result.pvalue >= self.threshold
            }
        except ImportError:
            return {"error": "missingpy package not installed"}
        except Exception as e:
            return {"error": str(e)}
            
    def _create_visualizations(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, str]:
        """Create visualizations for missing data"""
        visualizations = {}
        
        # 1. Missing heatmap
        heatmap_path = os.path.join(self.output_dir, "missing_heatmap.png")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[columns].isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Values Heatmap")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()
        visualizations["heatmap"] = heatmap_path
        
        # 2. Missing bar chart
        barchart_path = os.path.join(self.output_dir, "missing_barchart.png")
        plt.figure(figsize=(12, 6))
        missing_pct = (df[columns].isnull().mean() * 100).sort_values(ascending=False)
        missing_pct.plot(kind='bar')
        plt.title("Percentage of Missing Values by Column")
        plt.ylabel("Missing %")
        plt.tight_layout()
        plt.savefig(barchart_path)
        plt.close()
        visualizations["barchart"] = barchart_path
        
        return visualizations


class PredictivePowerAnalyzer(AnalyzerStep):
    """Step for analyzing predictive power of features"""
    
    def __init__(
        self,
        target_col: str = None,
        method: str = "random_forest",
        feature_cols: List[str] = None,
        output_dir: str = None,
        name: str = "PredictivePowerAnalyzer",
        **config
    ):
        """
        Args:
            target_col: Target column for prediction
            method: Method to use ('random_forest', 'mutual_info', 'correlation')
            feature_cols: Feature columns to analyze, if None use all numeric
            output_dir: Directory to save analysis results
            name: Name for this analyzer
            config: Additional configuration
        """
        super().__init__(name=name, **config)
        self.target_col = target_col
        self.method = method
        self.feature_cols = feature_cols
        self.output_dir = output_dir
        
    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze predictive power of features"""
        # Override configuration if provided in kwargs
        target_col = kwargs.get('target_col', self.target_col)
        method = kwargs.get('method', self.method)
        feature_cols = kwargs.get('feature_cols', self.feature_cols)
        
        if target_col is None or target_col not in df.columns:
            return {"error": "Target column must be specified and exist in the dataframe"}
            
        # Identify feature columns if not specified
        if feature_cols is None:
            feature_cols = [col for col in df.select_dtypes(include='number').columns 
                           if col != target_col]
                           
        if not feature_cols:
            return {"error": "No feature columns available for analysis"}
            
        try:
            # Handle different methods
            if method == "random_forest":
                return self._analyze_with_random_forest(df, target_col, feature_cols)
            elif method == "mutual_info":
                return self._analyze_with_mutual_info(df, target_col, feature_cols)
            elif method == "correlation":
                return self._analyze_with_correlation(df, target_col, feature_cols)
            else:
                return {"error": f"Unknown method: {method}"}
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
            
    def _analyze_with_random_forest(self, df: pd.DataFrame, target_col: str, 
                                  feature_cols: List[str]) -> Dict[str, Any]:
        """Analyze predictive power using Random Forest"""
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare data
            X = df[feature_cols].copy()
            
            # Handle missing values in features
            for col in X.columns:
                if X[col].isnull().any():
                    X[col] = X[col].fillna(X[col].median())
                    
            # Handle target variable
            y = df[target_col].copy()
            is_regression = df[target_col].dtype.kind in 'fc'
            
            if not is_regression:
                # Classification task
                le = LabelEncoder()
                y = le.fit_transform(y.fillna('_MISSING_'))
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                # Regression task
                y = y.fillna(y.median())
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
            # Fit model
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Create result
            feature_importance = dict(zip(feature_cols, importance))
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True))
                                          
            # Create visualization if output directory provided
            plot_path = None
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                plot_path = os.path.join(self.output_dir, "feature_importance.png")
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                importance_df = pd.DataFrame({
                    'Feature': list(sorted_importance.keys()),
                    'Importance': list(sorted_importance.values())
                })
                
                sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.title(f"Feature Importance for {target_col}")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
            return {
                "method": f"random_forest_{'regression' if is_regression else 'classification'}",
                "feature_importance": sorted_importance,
                "plot_path": plot_path
            }
        except ImportError:
            return {"error": "scikit-learn package not installed"}
            
    def _analyze_with_mutual_info(self, df: pd.DataFrame, target_col: str, 
                                feature_cols: List[str]) -> Dict[str, Any]:
        """Analyze predictive power using Mutual Information"""
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            # Prepare data
            X = df[feature_cols].copy()
            
            # Handle missing values in features
            for col in X.columns:
                if X[col].isnull().any():
                    X[col] = X[col].fillna(X[col].median())
                    
            # Handle target variable
            y = df[target_col].copy()
            is_regression = df[target_col].dtype.kind in 'fc'
            
            if not is_regression:
                # Classification task
                y = y.fillna('_MISSING_')
                mi_func = mutual_info_classif
            else:
                # Regression task
                y = y.fillna(y.median())
                mi_func = mutual_info_regression
                
            # Calculate mutual information
            mi_scores = mi_func(X, y, random_state=42)
            
            # Create result
            feature_importance = dict(zip(feature_cols, mi_scores))
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True))
                                          
            # Create visualization if output directory provided
            plot_path = None
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                plot_path = os.path.join(self.output_dir, "mutual_info.png")
                
                # Plot mutual information
                plt.figure(figsize=(12, 8))
                importance_df = pd.DataFrame({
                    'Feature': list(sorted_importance.keys()),
                    'Mutual Information': list(sorted_importance.values())
                })
                
                sns.barplot(x='Mutual Information', y='Feature', data=importance_df)
                plt.title(f"Mutual Information with {target_col}")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
            return {
                "method": f"mutual_info_{'regression' if is_regression else 'classification'}",
                "feature_importance": sorted_importance,
                "plot_path": plot_path
            }
        except ImportError:
            return {"error": "scikit-learn package not installed"}
            
    def _analyze_with_correlation(self, df: pd.DataFrame, target_col: str, 
                                feature_cols: List[str]) -> Dict[str, Any]:
        """Analyze predictive power using correlation"""
        # Calculate correlation with target
        correlations = {}
        for col in feature_cols:
            if df[col].dtype.kind in 'fc' and df[target_col].dtype.kind in 'fc':
                correlations[col] = df[[col, target_col]].corr().iloc[0, 1]
            else:
                correlations[col] = 0  # Skip non-numeric correlations
                
        # Sort by absolute correlation value
        sorted_correlations = dict(sorted(correlations.items(), 
                                        key=lambda x: abs(x[1]), 
                                        reverse=True))
                                        
        # Create visualization if output directory provided
        plot_path = None
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            plot_path = os.path.join(self.output_dir, "correlation_with_target.png")
            
            # Plot correlations
            plt.figure(figsize=(12, 8))
            correlation_df = pd.DataFrame({
                'Feature': list(sorted_correlations.keys()),
                'Correlation': list(sorted_correlations.values())
            })
            
            sns.barplot(x='Correlation', y='Feature', data=correlation_df)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f"Correlation with {target_col}")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
        return {
            "method": "correlation",
            "correlations": sorted_correlations,
            "plot_path": plot_path
        }

# Register additional analysis strategies
@strategy_registry.register
class MissingValueStrategy(Strategy):
    """Strategy for comprehensive missing value analysis"""
    
    def execute(self, data: pd.DataFrame, **kwargs):
        analyzer = MissingValueAnalyzer(**kwargs)
        return analyzer.analyze(data)

@strategy_registry.register
class PredictivePowerStrategy(Strategy):
    """Strategy for analyzing feature predictive power"""
    
    def execute(self, data: pd.DataFrame, **kwargs):
        analyzer = PredictivePowerAnalyzer(**kwargs)
        return analyzer.analyze(data)