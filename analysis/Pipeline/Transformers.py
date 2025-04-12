import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OrdinalEncoder
from Pipeline import PipelineStep

class Cleaner(PipelineStep):
    """Basic data cleaning operations"""
    
    def process(self, data: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        config = config or {}
        df = data.copy()
        
        # Handle missing values
        if config.get('handle_missing', True):
            threshold = config.get('missing_threshold', 0.5)
            df = self._handle_missing(df, threshold)
        
        # Handle outliers
        if config.get('handle_outliers', True):
            method = config.get('outlier_method', 'zscore')
            threshold = config.get('outlier_threshold', 3)
            df = self._handle_outliers(df, method, threshold)
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Drop columns with too many missing values"""
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        if cols_to_drop:
            self.logger.info(f"Dropping columns with >{threshold*100}% missing values: {cols_to_drop}")
            return df.drop(columns=cols_to_drop)
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, method: str, threshold: float) -> pd.DataFrame:
        """Handle outliers using specified method"""
        numeric_cols = df.select_dtypes(include='number').columns
        
        for col in numeric_cols:
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    z_scores = (df[col] - mean) / std
                    df.loc[abs(z_scores) > threshold, col] = np.nan
            elif method == 'iqr':
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
        
        return df

class FeatureEngineer(PipelineStep):
    """Basic feature engineering operations"""
    
    def process(self, data: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        config = config or {}
        df = data.copy()
        
        # Handle datetime columns
        if config.get('process_datetime', True):
            date_cols = config.get('datetime_columns', None)
            df = self._process_datetime(df, date_cols)
        
        # Handle categorical columns (object columns are considered categorical)
        if config.get('encode_categorical', False):
            cat_cols = config.get('categorical_columns', None)
            method = config.get('encoding_method', 'numerical')
            df = self._encode_categorical(df, cat_cols, method)
            print(df.isna().mean().dtypes)
        
        
        return df
    
    def _process_datetime(self, df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
        """Convert datetime columns to numeric features with proper timezone handling"""
        # Set reference date with UTC timezone to match Twitter's datetime format
        reference_date = pd.Timestamp('1970-01-01', tz='UTC')
        
        # Identify datetime columns if not specified
        if cols is None:
            cols = [col for col in df.columns 
                if pd.api.types.is_datetime64_any_dtype(df[col]) or 
                    'created_at' in col.lower() or 
                    'date' in col.lower()]
        
        cols_to_drop = []
        
        for col in cols:
            if col in df.columns:
                # Convert to datetime with UTC timezone (matching Twitter's format)
                try:
                    df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
                except TypeError:
                    # Fallback if column is already datetime but not timezone-aware
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize('UTC')
                
                # Only proceed if conversion was successful
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Ensure timezone is consistent
                    if df[col].dt.tz is None:
                        df[col] = df[col].dt.tz_localize('UTC')
                    elif df[col].dt.tz != reference_date.tz:
                        df[col] = df[col].dt.tz_convert('UTC')
                    
                    # Basic datetime features
                    df[f'{col}_year'] = df[col].dt.year.astype('float32')
                    df[f'{col}_quarter'] = df[col].dt.quarter.astype('float32')
                    df[f'{col}_month'] = df[col].dt.month.astype('float32')
                    df[f'{col}_day'] = df[col].dt.day.astype('float32')
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek.astype('float32')
                    df[f'{col}_hour'] = df[col].dt.hour.astype('float32')
                    df[f'{col}_minute'] = df[col].dt.minute.astype('float32')
                    
                    # Derived features
                    df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype('float32')
                    
                    # Timezone-aware date difference calculation
                    df[f'{col}_days_since_ref'] = (
                        (df[col] - reference_date).dt.total_seconds() / (24 * 3600)
                    ).astype('float32')
                    
                    # Cyclical features
                    df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[col].dt.month/12).astype('float32')
                    df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[col].dt.month/12).astype('float32')
                    df[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * df[col].dt.dayofweek/7).astype('float32')
                    df[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * df[col].dt.dayofweek/7).astype('float32')
                    df[f'{col}_hour_sin'] = np.sin(2 * np.pi * df[col].dt.hour/24).astype('float32')
                    df[f'{col}_hour_cos'] = np.cos(2 * np.pi * df[col].dt.hour/24).astype('float32')
                    
                    cols_to_drop.append(col)
        
        return df.drop(columns=cols_to_drop, errors='ignore')


    def _encode_categorical(self, df: pd.DataFrame, cols: List[str] = None,
    method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical variables"""
        if cols is None or len(cols) == 0:
            return df
        cols = [col for col in cols if col in df.select_dtypes(include=['object']).columns.tolist() and not pd.api.types.is_datetime64_any_dtype(df[col])]

        if method == 'onehot':
            # One-hot encoding using pd.get_dummies
            return pd.get_dummies(df, columns=cols)
        elif method == 'ordinal':
            # Ordinal encoding (convert to category first if it's an object)
            for col in cols:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype('category')
                df[col] = df[col].cat.codes
            return df

        elif method == 'numerical':
            # Numerical encoding using OrdinalEncoder from sklearn
            return self._encode_numerical(df, cols)

        else:
            # Default return of the dataframe if method is unknown
            return df
    
    def _encode_numerical(self, df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
        """Numerically encode categorical columns using OrdinalEncoder, handling NaNs before encoding"""
        if cols is None:
            cols = df.select_dtypes(include=['object']).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[(
                'ordinal', OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                    encoded_missing_value=-2  # Handle NaN values
                ), cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False  # Cleaner column names
        )

        # Apply the transformation to the specified columns
        encoded_data = preprocessor.fit_transform(df)

        # Convert the transformed data back into a DataFrame with proper column names
        df_encoded = pd.DataFrame(
            encoded_data,
            columns=preprocessor.get_feature_names_out(),
            index=df.index
        )
        

        return df_encoded

class Imputer(PipelineStep):
    """Handles missing value imputation with multiple strategies"""

    def process(self, data: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        config = config or {}
        df = data.copy()

        # Track missing values before any imputation
        self.logger.info("Missing values before imputation:")
        self.logger.info(df.isna().mean())
    
        strategy = config.get('strategy', 'median')
        print("HASCOLUMNS",df.dtypes)
        # Get numeric columns excluding datetime columns
        # df = df.apply(lambda col: pd.to_numeric(col, errors='ignore') if col.dtypes == 'object' else col)
        numeric_cols = [col for col in df.select_dtypes(include='number').columns]
        # Get categorical columns (object/string columns)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if strategy == 'simple':
            df = self._simple_impute(df, numeric_cols, categorical_cols, config)
        elif strategy == 'knn':
            df = self._knn_impute(df, numeric_cols, config)
        elif strategy == 'iterative':
            df = self._iterative_impute(df, numeric_cols, config)
        else:
            self.logger.warning(f"Unknown imputation strategy: {strategy}. Using 'simple' as fallback.")
            df = self._simple_impute(df, numeric_cols, categorical_cols, config)

        # Track missing values after imputation
        self.logger.info("Missing values after imputation:")
        self.logger.info(df.isna().mean())
        return df

    def _simple_impute(self, df: pd.DataFrame, numeric_cols: List[str], 
                      categorical_cols: List[str], config: Dict) -> pd.DataFrame:
        """Simple imputation using mean/median/most_frequent"""
        numeric_strategy = config.get('numeric_stratery', 'median')  # Note: typo in config key
        categorical_strategy = config.get('categorical_strategy', 'most_frequent')

        # Impute numeric columns
        if numeric_cols:
            num_imputer = SimpleImputer(strategy=numeric_strategy)
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

        # Impute categorical columns
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

        return df

    def _knn_impute(self, df: pd.DataFrame, numeric_cols: List[str], config: Dict) -> pd.DataFrame:
        """KNN imputation for numeric columns"""
        n_neighbors = config.get('n_neighbors', 5)
        if numeric_cols:
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
        return df

    def _iterative_impute(self, df: pd.DataFrame, numeric_cols: List[str], config: Dict) -> pd.DataFrame:
        """Iterative imputation for numeric columns"""
        max_iter = config.get('max_iter', 10)
        random_state = config.get('random_state', 21)
        estimator = config.get('estimator', BayesianRidge())

        print("df",df)
        print(numeric_cols)

        if 'label' in df.columns:
            y = df['label'].apply(lambda x: 1 if x == 'bot' else 0)
        if isinstance(estimator, RandomForestRegressor):
            # Handle case where 'label' column exists
            if y is not None:
                x = df[numeric_cols]
                print("IMPUTERR",x.columns, y)
                estimator.fit(X=x, y=y)

        if numeric_cols:
            it_imputer = IterativeImputer(
                estimator=estimator,
                max_iter=max_iter,
                random_state=random_state
            )
            df[numeric_cols] = it_imputer.fit_transform(df[numeric_cols])
        
        return df

