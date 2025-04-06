from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from Week3.base import Cleaner




class MissingValueHandler(Cleaner):
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

    def fit(self, data: pd.DataFrame, config: Dict = None) -> None:
        """Fit the cleaner to the data."""
        pass
    def transform(self, data: Any, config: Dict = None) -> Any:
        """Process the data to handle missing values."""
        strategy = config.get('strategy', 'removed_missing')
        if strategy == "mean":
            return data.fillna(data.mean())
        elif strategy == "removed_missing":
            minimum, maximum = config.get('threshold', (0.5, 0.95))
            to_iterate = config.get('to_iterate', False)
            df = self._removed_missing_balanced_(data, minimum, maximum, to_iterate)
            return df
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
