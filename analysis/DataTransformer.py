
from abc import ABC, abstractmethod
from typing import Dict, Union
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from abc import ABC, abstractmethod
from typing import Union, Dict, List
import pandas as pd
from sklearn.impute import IterativeImputer, SimpleImputer

from abc import ABC, abstractmethod
from typing import Union, Dict, Optional
import pandas as pd
from sklearn.impute import IterativeImputer, SimpleImputer
from abc import ABC, abstractmethod
from typing import Union, Dict, Optional
import pandas as pd
from sklearn.impute import IterativeImputer, SimpleImputer

class TransformerStrategy(ABC):
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
    
    @abstractmethod
    def transform(self, transformer: Optional['TransformerStrategy'] = None, **kwargs) -> Union[pd.DataFrame, Dict]:
        pass

class DataImputer(TransformerStrategy):
    def __init__(self, df: pd.DataFrame = None, strategy: str = 'iterative'):
        """
        Args:
            df: Input DataFrame
            strategy: 'iterative' (default) or 'simple' imputation
        """
        super().__init__(df)
        self.strategy = strategy
        self._validate_strategy()
        
    def _validate_strategy(self):
        valid_strategies = ['multivariate', 'single']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of {valid_strategies}")

    def _get_imputer(self, **kwargs) -> Union[IterativeImputer, SimpleImputer]:
        """Return the appropriate imputer based on strategy"""
        if self.strategy == 'iterative':
            return IterativeImputer(**kwargs)
        return SimpleImputer(**kwargs)

    def transform(self, df: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """
        Perform missing value imputation
        
        Args:
            df: DataFrame to transform (optional if provided in constructor)
            kwargs: Additional arguments to pass to the imputer
            
        Returns:
            Transformed DataFrame with imputed values
        """
        if df is not None:
            self.df = df.copy()
        
        if self.df is None:
            raise ValueError("No DataFrame provided for imputation")
            
        numeric_cols = self.df.select_dtypes(include='number').columns
        if len(numeric_cols) == 0:
            return self.df  # No numeric columns to impute
            
        imputer = self._get_imputer(**kwargs)
        imputed_data = imputer.fit_transform(self.df[numeric_cols])
        
        # Preserve column names and index
        self.df[numeric_cols] = pd.DataFrame(
            imputed_data, 
            columns=numeric_cols, 
            index=self.df.index
        )
        
        return self.df
class DataTransformer():
    def transform():
        pass
    def generate_report():
        """Generate a customizable report"""
if __name__ == '__main__':
    data =pd.read_csv("data/processed/flattened_status.csv", low_memory=False)
    df = pd.DataFrame(data).drop_duplicates()
    value = True
    if value == True:
        to_remove = df.columns[df.isna().mean() > 0.95]
        df = df.drop(columns=to_remove)
    imputer = DataImputer(df=df, strategy='multivariate')
    df = imputer.transform()
    
    df.to_csv('analysis/result/test_impute.csv')