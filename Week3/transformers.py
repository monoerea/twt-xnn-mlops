from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from Week3.base import Preprocess
from sklearn.utils.validation import check_array, check_is_fitted
from numpy.typing import ArrayLike

class MissingValueRemover(Preprocess):
    """Handle missing values in the data."""

    def __init__(self, name: str = None):
        super().__init__(name or self.__class__.__name__)
    def _get_missing_ratios_(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Get the ratio of missing values in each row and column."""
        rows_null_ratio = data.isnull().mean(axis=1)
        col_null_ratio = data.isnull().mean(axis=0)
        return rows_null_ratio, col_null_ratio

    def _get_null_candidates_(self, null_ratio: pd.Series, minimum: float = 0.5, maximum: float = 0.95, top_n: Optional[int] = None) -> pd.Series:
        """Get candidates for dropping based on null ratios."""
        candidates = null_ratio[(null_ratio > maximum) & (null_ratio < minimum)]
        self.logger.info(f"Candidates for dropping: {candidates}")
        if candidates.empty:
            return candidates
        if top_n is None:
            top_n = min(len(candidates), int(np.sqrt(len(null_ratio))))
        candidates = candidates.nlargest(top_n)
        self.logger.info(f"Candidates for dropping: {candidates}")
        return candidates

    def _drop_worst_(self, data: pd.DataFrame, col_candidates: pd.Series, row_candidates: pd.Series) -> pd.DataFrame:
        """Drop the worst candidate based on null ratios."""
        if not col_candidates.empty and col_candidates.max() > row_candidates.max() if not row_candidates.empty else True:
            worst_col = col_candidates.idxmax()
            data.drop(columns=worst_col, inplace=True)
            self.logger.info(f"Dropping column: {worst_col}")
        elif not row_candidates.empty:
            worst_col = row_candidates.idxmax()
            self.logger.info(f"Dropping row: {worst_col}")
            data.drop(index=worst_col, inplace=True)
        return data
    def _removed_missing_(self, data: pd.DataFrame, minimum: float = 0.5, maximum: float = 0.95) -> pd.DataFrame:
        """Remove columns with missing values based on a threshold."""
        rows_null_ratio, col_null_ratio = self._get_missing_ratios_(data)

        col_candidates = self._get_null_candidates_(col_null_ratio, minimum, maximum)
        row_candidates = self._get_null_candidates_(rows_null_ratio, minimum, maximum)

        return col_candidates, row_candidates

    def _removed_missing_balanced_(self, data: pd.DataFrame, minimum: float = 0.5, maximum: float = 0.95, to_iterate: bool = False) -> pd.DataFrame:
        """Remove columns with missing values based on a threshold."""
        df = data.copy()
        while True:
            col_candidates, row_candidates = self._removed_missing_(df, minimum, maximum)
            self.logger.info(f"Row null ratios: {row_candidates}")
            self.logger.info(f"Column null ratios: {col_candidates}")
            if col_candidates.empty and row_candidates.empty:
                break
            df = self._drop_worst_(df, col_candidates, row_candidates=row_candidates)
            if to_iterate == False:
                break
        if df.shape[0] == 0:
            raise ValueError("All rows have been dropped. Please check your data.")
        self.logger.info(f"Final shape after removing missing values: {df.shape}")
        return df

    def transform(self, data: Any, config: Dict = None) -> Any:
        """Process the data to handle missing values."""
        strategy = config.get('strategy', 'remove_missing')
        if strategy == "remove_missing":
            minimum, maximum = config.get('threshold', (0.5, 0.95))
            to_iterate = config.get('to_iterate', False)
            self.logger.info(f"Shape before removing missing values: {data.shape}")
            df = self._removed_missing_balanced_(data, minimum, maximum, to_iterate)
            return df
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

class DataImputer(Preprocess):
    """Handle missing values in the data."""

    def __init__(self, name: str = None):
        super().__init__(name or self.__class__.__name__)

    def transform(self, data: Any, config: Dict = None) -> Any:
        """Process the data to handle missing values."""
        strategy = config.get('strategy', 'median')
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
                'n_nearest_features': None,
                'skip_complete': False,
                'imputation_order': 'ascending',
                'min_value': None,
                'max_value': None
            })
            imputer = IterativeImputer(**params)
            self.logger.info(f"Imputer params: {params}")
            data = imputer.fit_transform(data)
            self.logger.info("Imputing missing values with Iterative Imputer")

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
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
                log_scaler = LogScaler()
                data = log_scaler.fit_transform(data)
                self.logger.info("Log scaling applied")
            return data

class LogScaler(BaseEstimator, TransformerMixin):
    """Logarithmic scaling for data."""

    def __init__(self, base: float = np.e):
        self.base = base

    def fit(self, x: ArrayLike, y: Optional[pd.Series] = None) -> 'LogScaler':
        x = check_array(x, accept_sparse=False, ensure_2d=False, dtype=['numeric'])
        if np.any(x <= 0):
            raise ValueError("Logarithmic scaling requires all values to be positive.")
        self.n_features_in_ = x.shape[1]
        if (hasattr(x, "columns")) and (x.columns is not None):
            self.feature_names_in_ = np.array(x.columns)
        return self

    def transform(self, x: ArrayLike) -> pd.DataFrame:
        check_is_fitted(self, self.n_features_in_)
        x = check_array(x, accept_sparse=False, ensure_2d=False, dtype=['numeric'])
        if np.any(x <= 0):
            raise ValueError("Logarithmic scaling requires all values to be positive.")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}")
        result = pd.DataFrame(np.log(x) / np.log(self.base), columns=self.feature_names_in_, index=x.index)
        return result
class CategoricalEncoder(Preprocess):
    """Handle categorical data in the data."""

    def __init__(self, name: str = None):
        super().__init__(name or self.__class__.__name__)

    def transform(self, data: Any, config: Dict = None) -> Any:
        """Process the data to handle categorical data."""
        strategy = config.get('strategy', 'ordinal')
        self.logger.info(f"Encoding categorical variables with strategy: {strategy}")
        if strategy == "ordinal":
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