from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from Pipeline.base import Preprocess
from sklearn.utils.validation import check_array, check_is_fitted
from numpy.typing import ArrayLike

class MissingValueRemover(Preprocess):
    """Missing value handler with separate row/column thresholds and batch processing."""

    def __init__(self, name: str = None):
        super().__init__(name or self.__class__.__name__)
        self._null_matrix_cache = None

    def _get_missing_ratios(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Get cached null ratios with shape validation."""
        if self._null_matrix_cache is None or self._null_matrix_cache.shape != data.shape:
            self._null_matrix_cache = data.isnull()
        return self._null_matrix_cache.mean(axis=1), self._null_matrix_cache.mean(axis=0)

    def _get_threshold_violators(self, 
                               null_ratio: pd.Series, 
                               threshold: float,
                               batch_count: int) -> pd.Series:
        """Get top N violators sorted by severity."""
        violators = null_ratio[null_ratio > threshold].sort_values(ascending=False)
        return violators.head(batch_count)

    def _log_violators(self, violators: pd.Series, item_type: str):
        """Log detailed information about violators."""
        if not violators.empty:
            self.logger.info(
                f"Top {len(violators)} {item_type} violators:\n"
                f"Max: {violators.max():.2f}\n"
                f"Min: {violators.min():.2f}\n"
                f"Mean: {violators.mean():.2f}"
            )

    def transform(self, data: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        """
        Transform data with configurable thresholds and batch counts.
        
        Config Options:
        - row_threshold: float (default 0.95)
        - col_threshold: float (default 0.90) 
        - row_batch_count: int (default 10)
        - col_batch_count: int (default 5)
        - max_iterations: int (default 20)
        """
        config = config or {}
        row_thresh = config.get('row_threshold', 0.95)
        col_thresh = config.get('col_threshold', 0.90)
        row_batch = config.get('row_batch_count', 10)
        col_batch = config.get('col_batch_count', 5)
        max_iter = config.get('max_iterations', 20)
        
        df = data.copy()
        self.logger.info(
            f"Initial shape: {df.shape}\n"
            f"Row threshold: >{row_thresh}, batch: {row_batch}\n"
            f"Column threshold: >{col_thresh}, batch: {col_batch}"
        )

        for iteration in range(1, max_iter + 1):
            row_ratios, col_ratios = self._get_missing_ratios(df)
            row_violators = self._get_threshold_violators(row_ratios, row_thresh, row_batch)
            col_violators = self._get_threshold_violators(col_ratios, col_thresh, col_batch)

            self._log_violators(row_violators, "row")
            self._log_violators(col_violators, "column")

            if row_violators.empty and col_violators.empty:
                self.logger.info(f"Clean data achieved at iteration {iteration}")
                break

            # Prioritize rows first
            if not row_violators.empty:
                df = df.drop(index=row_violators.index)
                self.logger.info(f"Iteration {iteration}: Dropped {len(row_violators)} rows")
            elif not col_violators.empty:
                df = df.drop(columns=col_violators.index)
                self.logger.info(f"Iteration {iteration}: Dropped {len(col_violators)} columns")

            self._null_matrix_cache = None  # Invalidate cache

        # Final validation report
        final_rows, final_cols = self._get_missing_ratios(df)
        remaining_row_violations = final_rows[final_rows > row_thresh].count()
        remaining_col_violations = final_cols[final_cols > col_thresh].count()
        
        self.logger.info(
            f"Final shape: {df.shape}\n"
            f"Remaining row violations: {remaining_row_violations}\n"
            f"Remaining column violations: {remaining_col_violations}"
        )
        
        return df

class DataImputer(Preprocess):
    """Handle missing values in the data."""

    def __init__(self, name: str = None):
        super().__init__(name or self.__class__.__name__)

    def transform(self, data: Any, config: Dict = None) -> Any:
        """Process the data to handle missing values."""
        strategy = config.get('strategy', 'median')
        cat_cols = data.select_dtypes(include=['object']).columns
        if strategy == "median":
            self.logger.info("Imputing missing values with median")
            data.fillna(data.median(), inplace=True)
            self.logger.info(f"Shape after imputing missing values: {data.shape}")
        if strategy == "mean":
            self.logger.info("Imputing missing values with mean")
            data.fillna(data.mean(), inplace=True)
            self.logger.info(f"Shape after imputing missing values: {data.shape}")
        if strategy  == "knn":
            from sklearn.impute import KNNImputer
            n_neighbors = config.get('n_neighbors', 5)
            self.logger.info("Imputing missing values with KNN")
            imputer = KNNImputer(n_neighbors=n_neighbors)
            data = imputer.fit_transform(data)
            self.logger.info(f"Shape after imputing missing values: {data.shape}")
        if strategy == "iterative":
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            from sklearn.linear_model import BayesianRidge
            estimator = config.get('imputer', BayesianRidge())
            params = config.get('params', {
                'estimator': estimator,
                'max_iter': 10,
                'tol': 0.001,
                'randoem_state': 21,
                'n_nearest_features': None,
                'skip_complete': False,
                'imputation_order': 'ascending',
                'min_value': None,
                'max_value': None
            })
            imputer = IterativeImputer(**params)
            self.logger.info(f"Imputer params: {params}")
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
            self.logger.info("Imputing missing values with Iterative Imputer")

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        data[cat_cols] = data[cat_cols].astype('object')
        self.logger.info(f"DATA IMPUTER:Data columns {data.select_dtypes(include=['object']).columns} are not numeric")
        return data
class DataScaler(Preprocess):
        def __init__(self, name: str = None):
            super().__init__(name or self.__class__.__name__)
        def transform(self, data: Any, config: Dict = None) -> Any:
            """Process the data to handle normalization of outlier values."""
            strategy = config.get('strategy', 'standard')
            if strategy == "standard":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                data = scaler.fit_transform(data)
                self.logger.info("Standard scaling applied")
            elif strategy == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                data = scaler.fit_transform(data)
                self.logger.info("MinMax scaling applied")
            elif strategy == "robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                data = scaler.fit_transform(data)
                self.logger.info("Robust scaling applied")
            elif strategy == "log":
                base = config.get('base', np.e)
                log_scaler = LogScaler(base=base)
                self.logger.info(f"Log scaling with data{data} and base {base}")
                result = log_scaler.fit_transform(data)
                data = pd.DataFrame(result, columns=data.columns, index=data.index)
                self.logger.info("Log scaling applied")
            self.logger.info(f"DATA IMPUTER:Data columns {data.select_dtypes(include=['object']).columns} are not numeric")
            return data

class LogScaler(BaseEstimator, TransformerMixin):
    """Logarithmic scaling for data."""

    def __init__(self, base: float = np.e):
        self.base = base

    def fit(self, x: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LogScaler':
        x = check_array(x, accept_sparse=False, ensure_2d=False)
        if np.any(x <= -2):
            raise ValueError("Logarithmic scaling requires all values to be positive.")
        self.n_features_in_ = x.shape[1]
        if (hasattr(x, "columns")) and (x.columns is not None):
            self.feature_names_in_ = x.columns
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(self.n_features_in_)]
        return self

    def transform(self, x:  pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=['n_features_in_'])
        cat_features = x.select_dtypes(include=['object','category']).columns
        if not hasattr(self, 'feature_names_in_'):
            raise ValueError("The fit method must be called before transform.")
        x_array = check_array(x[x.columns.difference(cat_features)], accept_sparse=False, ensure_2d=False)
        result = np.log1p(x_array) / np.log(self.base)
        result = pd.DataFrame(result, columns=x.columns, index=x.index)
        result[cat_features] = result[cat_features].astype('object')
        return result
class CategoricalEncoder(Preprocess):
    """Handle categorical data in the data."""

    def __init__(self, name: str = None):
        super().__init__(name or self.__class__.__name__)

    def transform(self, data: Any, config: Dict = None) -> Any:
        """Process the data to handle categorical data."""
        self.config = config or {}
        strategy = config.get('strategy', 'target')
        self.logger.info(f"Encoding categorical variables with strategy: {strategy}")

        if strategy == "target":
            data = TargetEncoder(name='TargetEncoder').fit_transform(data, config)
            self.logger.info("Data in target, with data of", data.head())
        elif strategy == "ordinal":
            self.logger.info(f"Shape before ordinal encoding: {data.shape}")
            from sklearn.preprocessing import OrdinalEncoder
            ordinal_encoder = OrdinalEncoder()
            object_columns = data.select_dtypes(include=['object']).columns
            encoded_colummns = ordinal_encoder.fit_transform(data[object_columns])
            self.logger.info("Encoding categorical variables with ordinal encoding")
            data[object_columns] = pd.DataFrame(encoded_colummns, columns=object_columns, index=data.index)
            self.logger.info(f"Categories encoded: {ordinal_encoder.categories_}")
            self.logger.info(f"Shape after ordinal encoding: {data.shape}")
        elif strategy == "onehot":
            self.logger.info("Encoding categorical variables with one-hot encoding")
            data = pd.get_dummies(data, drop_first=True)
            self.logger.info(f"Shape after one-hot encoding: {data.shape}")
        elif strategy == "label":
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            for col in data.select_dtypes(include=['object']).columns:
                data[col] = label_encoder.fit_transform(data[col])
            self.logger.info("Encoding categorical variables with label encoding")
            self.logger.info(f"Shape after label encoding: {data.shape}")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return data

class TargetEncoder(Preprocess):
    """Handle categorical data in the data."""

    def __init__(self, name: str = None):
        super().__init__(name or self.__class__.__name__)

    def transform(self, data: Any, config: Dict = None) -> pd.DataFrame:
        categorical_data = config['columns'] if config.get('target') is not None else data.select_dtypes(include=['object']).columns

        encoded_columns = {}
        for col in categorical_data:
            encoded_columns[col] = data[col].astype('category').cat.codes
        data.update(pd.DataFrame(encoded_columns, index=data.index))
        return data