import re
import ast
import warnings
from typing import Dict, List, Any, Union, Optional, Callable, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from Pipeline import Step, Strategy, strategy_registry


class TransformerStep(Step):
    """Base class for all transformer steps"""
    
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Default implementation that calls transform"""
        return self.transform(data, **kwargs)
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform the dataframe - override in subclasses"""
        return df


class DataCleaner(TransformerStep):
    """General purpose data cleaning step with configurable strategies"""
    
    def __init__(
        self,
        strategies: List[Union[str, Strategy]] = None,
        column_patterns: Dict[str, str] = None,
        name: str = "DataCleaner",
        **config
    ):
        """
        Args:
            strategies: List of strategy names or instances to apply
            column_patterns: Dictionary mapping strategy names to column regex patterns
            name: Name for this transformer
            config: Additional configuration
        """
        super().__init__(name=name, **config)
        self.strategies = strategies or []
        self.column_patterns = column_patterns or {}
        self._resolve_strategies()
        
    def _resolve_strategies(self):
        """Resolve strategy strings to instances"""
        resolved = []
        for strategy in self.strategies:
            if isinstance(strategy, str):
                resolved.append(strategy_registry.get(strategy))
            else:
                resolved.append(strategy)
        self.strategies = resolved
        
    def add_strategy(self, strategy: Union[str, Strategy], column_pattern: str = None) -> 'DataCleaner':
        """Add a cleaning strategy"""
        if isinstance(strategy, str):
            strategy = strategy_registry.get(strategy)
        self.strategies.append(strategy)
        if column_pattern:
            self.column_patterns[strategy.name] = column_pattern
        return self
            
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply all cleaning strategies to the dataframe"""
        result = df.copy()
        
        for strategy in self.strategies:
            try:
                strategy_name = strategy.name
                pattern = self.column_patterns.get(strategy_name)
                
                # Get strategy-specific config from kwargs
                strategy_kwargs = kwargs.get(strategy_name, {})
                
                # Determine target columns based on pattern
                if pattern:
                    columns = [col for col in result.columns if re.search(pattern, col, re.IGNORECASE)]
                    strategy_kwargs['columns'] = columns
                
                self.logger.info(f"Applying {strategy_name} strategy")
                result = strategy.execute(result, **strategy_kwargs)
                
            except Exception as e:
                self.logger.error(f"Error applying {strategy.name}: {str(e)}")
                
        return result


class DynamicTransformer(TransformerStep):
    """Generic transformer with dynamically configurable operations"""
    
    def __init__(
        self, 
        operations: List[Dict] = None,
        name: str = "DynamicTransformer", 
        **config
    ):
        """
        Args:
            operations: List of operation configs, each with 'type', 'columns', and params
            name: Name for this transformer
            config: Additional configuration
        """
        super().__init__(name=name, **config)
        self.operations = operations or []
        
    def add_operation(self, op_type: str, columns=None, **params) -> 'DynamicTransformer':
        """Add an operation to the transformer"""
        self.operations.append({
            'type': op_type,
            'columns': columns,
            **params
        })
        return self
        
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply all operations to the dataframe"""
        result = df.copy()
        
        for i, operation in enumerate(self.operations):
            op_type = operation['type']
            columns = operation.get('columns')
            
            # Get columns based on pattern if it's a string
            if isinstance(columns, str):
                columns = [col for col in result.columns if re.search(columns, col, re.IGNORECASE)]
                
            # Extract parameters for this operation
            params = {k: v for k, v in operation.items() if k not in ['type', 'columns']}
            
            # Override params with those from kwargs if provided
            if op_type in kwargs:
                params.update(kwargs[op_type])
                
            self.logger.info(f"Applying operation {i+1}: {op_type}")
            
            try:
                # Handle different operation types
                if op_type == 'impute':
                    result = self._impute_data(result, columns, **params)
                elif op_type == 'encode':
                    result = self._encode_data(result, columns, **params)
                elif op_type == 'datetime':
                    result = self._process_datetime(result, columns, **params)
                elif op_type == 'custom':
                    # Custom operation using provided function
                    func = params.pop('func', None)
                    if func and callable(func):
                        result = func(result, columns, **params)
                    else:
                        self.logger.warning(f"Invalid custom function for operation {i+1}")
                else:
                    self.logger.warning(f"Unknown operation type: {op_type}")
            
            except Exception as e:
                self.logger.error(f"Error in operation {i+1} ({op_type}): {str(e)}")
                
        return result
        
    def _impute_data(self, df: pd.DataFrame, columns=None, strategy='median', **kwargs) -> pd.DataFrame:
        """Impute missing values in the dataframe"""
        result = df.copy()
        
        # Get columns to impute
        if columns is None:
            columns = result.select_dtypes(include='number').columns.tolist()
        
        # Skip if no columns to impute or no missing values
        if not columns or result[columns].isnull().sum().sum() == 0:
            return result
            
        try:
            # Create and apply imputer
            if strategy == 'iterative':
                estimator = kwargs.get('estimator', BayesianRidge())
                imputer = IterativeImputer(estimator=estimator, **kwargs)
                result[columns] = imputer.fit_transform(result[columns])
            else:
                imputer = SimpleImputer(strategy=strategy, **kwargs)
                result[columns] = imputer.fit_transform(result[columns])
        except Exception as e:
            self.logger.warning(f"Imputation failed: {str(e)}. Using median as fallback.")
            # Fallback to median imputation
            for col in columns:
                if result[col].isnull().sum() > 0:
                    result[col] = result[col].fillna(result[col].median())
                    
        return result
        
    def _encode_data(self, df: pd.DataFrame, columns=None, method='ordinal', **kwargs) -> pd.DataFrame:
        """Encode categorical columns"""
        result = df.copy()
        
        # Get columns to encode
        if columns is None:
            columns = result.select_dtypes(include=['object', 'category']).columns.tolist()
            
        if not columns:
            return result
            
        try:
            # Configure encoder based on method
            if method == 'ordinal':
                encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                    **kwargs
                )
            elif method == 'onehot':
                encoder = OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown encoding method: {method}")
                
            # Apply encoder
            transformer = ColumnTransformer(
                transformers=[(method, encoder, columns)],
                remainder='passthrough',
                verbose_feature_names_out=False
            )
            
            # Transform data
            encoded_data = transformer.fit_transform(result)
            feature_names = transformer.get_feature_names_out()
            
            # Create result dataframe
            result = pd.DataFrame(
                encoded_data,
                columns=feature_names,
                index=result.index
            )
            
        except Exception as e:
            self.logger.error(f"Encoding failed: {str(e)}")
            
        return result
        
    def _process_datetime(self, df: pd.DataFrame, columns=None, extract_components=True, 
                         cyclical=True, reference_date=None, **kwargs) -> pd.DataFrame:
        """Process datetime columns"""
        result = df.copy()
        
        # Identify datetime columns if not specified
        if columns is None:
            datetime_cols = result.select_dtypes(include=['datetime64']).columns.tolist()
            # Try to convert string columns
            for col in result.select_dtypes(include=['object']).columns:
                try:
                    pd.to_datetime(result[col].iloc[0:5], errors='raise')
                    datetime_cols.append(col)
                except:
                    pass
        else:
            datetime_cols = columns
            
        if not datetime_cols:
            return result
            
        # Reference date for time deltas
        ref_date = pd.to_datetime(reference_date) if reference_date else pd.to_datetime('2000-01-01')
        
        # Process each datetime column
        for col in datetime_cols:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(result[col]):
                result[col] = pd.to_datetime(result[col], errors='coerce')
                
            if extract_components:
                # Extract basic components
                result[f'{col}_year'] = result[col].dt.year
                result[f'{col}_month'] = result[col].dt.month
                result[f'{col}_day'] = result[col].dt.day
                result[f'{col}_dayofweek'] = result[col].dt.dayofweek
                result[f'{col}_days_since_ref'] = (result[col] - ref_date).dt.days
                
                # Extract cyclical features if requested
                if cyclical:
                    result[f'{col}_month_sin'] = np.sin(2 * np.pi * result[col].dt.month/12)
                    result[f'{col}_month_cos'] = np.cos(2 * np.pi * result[col].dt.month/12)
                    result[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * result[col].dt.dayofweek/7)
                    result[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * result[col].dt.dayofweek/7)
            
            # Remove original column if specified
            if kwargs.get('remove_original', False):
                result = result.drop(columns=[col])
                
        return result


# Register common cleaning strategies
@strategy_registry.register
class MissingValueStrategy(Strategy):
    """Strategy for handling missing values"""
    
    def execute(self, data: pd.DataFrame, columns=None, threshold=0.995, **kwargs) -> pd.DataFrame:
        """Handle missing values"""
        result = data.copy()
        
        # Get columns to process
        if columns is None:
            columns = result.columns
            
        # Remove columns with too many missing values
        if threshold < 1.0:
            missing_ratio = result[columns].isnull().mean()
            cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
            if cols_to_drop:
                self.logger.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing values")
                result = result.drop(columns=cols_to_drop)
                
        return result


@strategy_registry.register
class OutlierHandlingStrategy(Strategy):
    """Strategy for handling outliers"""
    
    def execute(self, data: pd.DataFrame, columns=None, method='zscore', threshold=3, **kwargs) -> pd.DataFrame:
        """Handle outliers"""
        result = data.copy()
        
        # Get numeric columns if not specified
        if columns is None:
            columns = result.select_dtypes(include='number').columns
            
        if method == 'zscore':
            for col in columns:
                if col in result.columns:
                    # Calculate z-scores
                    mean = result[col].mean()
                    std = result[col].std()
                    if std > 0:  # Avoid division by zero
                        z_scores = (result[col] - mean) / std
                        # Identify outliers
                        outliers = abs(z_scores) > threshold
                        # Handle outliers based on strategy
                        if kwargs.get('action') == 'remove':
                            result = result[~outliers]
                        elif kwargs.get('action') == 'cap':
                            upper_bound = mean + threshold * std
                            lower_bound = mean - threshold * std
                            result.loc[result[col] > upper_bound, col] = upper_bound
                            result.loc[result[col] < lower_bound, col] = lower_bound
                        else:  # Default: replace with NaN
                            result.loc[outliers, col] = np.nan
                            
        elif method == 'iqr':
            for col in columns:
                if col in result.columns:
                    # Calculate IQR
                    q1 = result[col].quantile(0.25)
                    q3 = result[col].quantile(0.75)
                    iqr = q3 - q1
                    # Define bounds
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    # Identify outliers
                    outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
                    # Handle outliers
                    if kwargs.get('action') == 'remove':
                        result = result[~outliers]
                    elif kwargs.get('action') == 'cap':
                        result.loc[result[col] > upper_bound, col] = upper_bound
                        result.loc[result[col] < lower_bound, col] = lower_bound
                    else:  # Default: replace with NaN
                        result.loc[outliers, col] = np.nan
                        
        return result


@strategy_registry.register
class TextCleaningStrategy(Strategy):
    """Strategy for cleaning and processing text data"""
    
    def execute(self, data: pd.DataFrame, columns=None, operations=None, **kwargs) -> pd.DataFrame:
        """Clean text data"""
        result = data.copy()
        
        # Get text columns if not specified
        if columns is None:
            columns = result.select_dtypes(include='object').columns
            
        # Default text cleaning operations
        ops = operations or ['lowercase', 'strip', 'remove_special']
        
        for col in columns:
            if col in result.columns:
                # Apply each text operation
                for op in ops:
                    if op == 'lowercase':
                        result[col] = result[col].str.lower()
                    elif op == 'strip':
                        result[col] = result[col].str.strip()
                    elif op == 'remove_special':
                        result[col] = result[col].str.replace(r'[^\w\s]', '', regex=True)
                        
                # Extract features if requested
                if kwargs.get('extract_length', False):
                    result[f'{col}_length'] = result[col].str.len()
                if kwargs.get('extract_word_count', False):
                    result[f'{col}_word_count'] = result[col].str.split().str.len()
                    
        return result


# Common utility functions
def extract_json_fields(text_series, field_path=None):
    """Extract fields from JSON/dict-like text"""
    results = []
    
    for text in text_series:
        try:
            # Parse text to dict
            if isinstance(text, str):
                data = ast.literal_eval(text)
            else:
                data = text
                
            # Extract specific field if path provided
            if field_path:
                parts = field_path.split('.')
                for part in parts:
                    if isinstance(data, dict) and part in data:
                        data = data[part]
                    elif isinstance(data, list) and part.isdigit() and int(part) < len(data):
                        data = data[int(part)]
                    else:
                        data = None
                        break
            
            results.append(data)
        except:
            results.append(None)
            
    return results