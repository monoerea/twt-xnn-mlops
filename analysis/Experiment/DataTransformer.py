from abc import ABC, abstractmethod
import ast
import re
from typing import Union, Dict, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
class TransformerStrategy(ABC):
    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    @abstractmethod
    def transform(self, transformer: Optional['TransformerStrategy'] = None, **kwargs) -> Union[pd.DataFrame, Dict]:
        pass
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
import warnings
class DataImputer(TransformerStrategy):
    def __init__(self, df: pd.DataFrame = None, strategy: str = 'iterative',
                 estimator=None, **imputer_kwargs):
        """
        Args:
            df: Input DataFrame
            strategy: 'iterative' (default) or 'simple' imputation
            estimator: Custom estimator for iterative imputation
            imputer_kwargs: Additional arguments for the imputer
        """
        super().__init__(df)
        self.strategy = strategy
        self.estimator = estimator
        self.imputer_kwargs = imputer_kwargs
        self._validate_strategy()

    def _validate_strategy(self):
        valid_strategies = ['iterative', 'simple']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of {valid_strategies}")

    def _get_imputer(self) -> Union[IterativeImputer, SimpleImputer]:
        """Return configured imputer with error handling"""
        try:
            if self.strategy == 'iterative':
                estimator = self.estimator or BayesianRidge()
                return IterativeImputer(estimator=estimator, **self.imputer_kwargs)
            return SimpleImputer(**self.imputer_kwargs)
        except Exception as e:
            warnings.warn(f"Imputer initialization failed: {str(e)}")
            return SimpleImputer(strategy='mean')

    def transform(self, df: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """Handle NaN values with robust error recovery"""
        if df is not None:
            self.df = df.copy()
        if self.df.isnull().sum().sum() == 0:
            return self.df  # Early return if no missing values

        numeric_cols = self.df.select_dtypes(include='number').columns
        if len(numeric_cols) == 0:
            return self.df
        try:
            imputer = self._get_imputer()
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
        except Exception as e:
            warnings.warn(f"Imputation failed: {str(e)}. Using median imputation as fallback")
            # Emergency fallback
            self.df[numeric_cols] = self.df[numeric_cols].fillna(
                self.df[numeric_cols].median()
            )
        return self.df
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MultiDateTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date='2000-01-01',
                 extract_cyclical=True,
                 extract_derived=True):
        # Ensure reference_date matches the timezone awareness of your data
        self.reference_date = pd.to_datetime(reference_date).tz_localize(None)  # Make naive
        self.extract_cyclical = extract_cyclical
        self.extract_derived = extract_derived
        self.datetime_cols_ = None

    def fit(self, X, y=None):
        # Convert potential datetime columns and standardize timezones
        datetime_candidates = [col for col in X.columns if 'created_at' in col.lower()]
        for col in datetime_candidates:
            if not pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = pd.to_datetime(X[col])
            # Make all datetime columns timezone-naive
            X[col] = X[col].dt.tz_localize(None)

        self.datetime_cols_ = [col for col in datetime_candidates if pd.api.types.is_datetime64_any_dtype(X[col])]

    def transform(self, X):
        X = X.copy()
        for col in self.datetime_cols_:
            # Basic temporal features
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_quarter'] = X[col].dt.quarter
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_day'] = X[col].dt.day
            X[f'{col}_dayofweek'] = X[col].dt.dayofweek
            X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
            X[f'{col}_days_since_ref'] = (X[col] - self.reference_date).dt.days

            # Cyclical encoding if enabled
            if self.extract_cyclical:
                X[f'{col}_month_sin'] = np.sin(2 * np.pi * X[col].dt.month/12)
                X[f'{col}_month_cos'] = np.cos(2 * np.pi * X[col].dt.month/12)
                X[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * X[col].dt.dayofweek/7)
                X[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * X[col].dt.dayofweek/7)

            # Derived temporal relationships
            if self.extract_derived and len(self.datetime_cols_) > 1:
                for other_col in self.datetime_cols_:
                    if col != other_col:
                        delta = X[col] - X[other_col]
                        X[f'{col}_minus_{other_col}_days'] = delta.dt.total_seconds()/(24*3600)
        return X.drop(columns=self.datetime_cols_)
class DataTransformer():
    def __init__(self):
        pass
    def transform(self):
        pass
    def generate_report(self):
        """Generate a customizable report"""

def extract_hashtags(hashtag_column):
    """Extract hashtag texts from formatted string data."""
    hashtags = []
    try:
        # Handle both string representations and actual lists
        if isinstance(hashtag_column, str):
            data = ast.literal_eval(hashtag_column)
        else:
            data = hashtag_column
        
        # Process single dict or list of dicts
        if isinstance(data, dict):
            hashtags.append(data['text'])
        elif isinstance(data, list):
            hashtags.extend([item['text'] for item in data if isinstance(item, dict)])
    except (ValueError, SyntaxError, KeyError, TypeError):
        pass
    return hashtags if hashtags else None
if __name__ == '__main__':
    data =pd.read_csv("data/processed/flattened_status.csv", low_memory=False)
    df = pd.DataFrame(data).drop_duplicates()
    initial = df
    y = df['label'].apply(lambda x: 1 if x == 'bot' else 2)

    #hashtags
    hashtag_columns = [col for col in df.columns if 'hashtag' in col.lower()]

    for col in hashtag_columns:
        df[col] = df[col].apply(
        lambda x: extract_hashtags(x)[0]
        if extract_hashtags(x) and len(extract_hashtags(x)) > 0
        else None
        )
        print(df[col])

    datetime_candidates = [col for col in df.columns if 'created_at' in col.lower()]
    df[datetime_candidates] = df[datetime_candidates].apply(pd.to_datetime, errors='coerce')
    date_transformer = MultiDateTimeTransformer()
    date_transformer.fit(X=df)
    df = date_transformer.transform(df)
    # print("Generated datetime features:")
    # print([col for col in df.columns if any(orig_col in col for orig_col in date_transformer.datetime_cols_)])

    source_cols = [col for col in df.columns if re.search(r'source', col, re.IGNORECASE)]
    pattern = r'>Twitter for (Android|iPhone)<|>Twitter (Web App)<|>(True Anthem)<'
    df[source_cols] = (
        df[source_cols]
        .apply(lambda col: col.str.extract(pattern)[0].fillna('other')))

    df[[col for col in df.columns if 'url' in col.lower()]] = df.filter(like='url', axis=1).notna()

    remove_patterns = ['id','text','media','screen_name','mentions','description']
    pattern = re.compile('|'.join(remove_patterns), flags=re.IGNORECASE)

    # Get columns to remove
    to_remove = [col for col in df.columns if pattern.search(col)]

    # Select target columns (numeric with 5-95% nulls)
    target_cols = [
        col for col in df.select_dtypes(include='number').columns
        if 0.05 < df[col].isnull().mean() < 0.95
    ]

    # Remove unwanted columns
    df = df[df.columns.difference(to_remove)]
    initial = initial[initial.columns.difference(to_remove)]
    df.to_csv('analysis/result/preencoding.csv')
    #########################################
    from DataInspector import DataInspector
    inspector = DataInspector(
        df=df,
        thresholds=[0.5, 0.95],
        exclude_columns=['id']
    )

    print("=== DEFAULT REPORT ===")
    output_dir = 'analysis/result/transformed'
    strategies = [
            ('basic_info', {}),
            ('mcar', {'columns': None, 'pct': 0.05}),
            ('ttest', {'target_cols': None, 'group_col': None, }),
            ('outlier',{'threshold':5}),
            ('missing_heatmap', {'filename': f"{output_dir}/missing_heatmap.png"})
        ]
    inspector.generate_report(output_dir=output_dir, strategies=strategies)
    ###################
    print([col for col in initial.columns if 'created_at' in col])


    categorical_cols = df.select_dtypes(include=['object','bool']).columns.tolist()

    print("Columns being encoded:", categorical_cols)
    print("Sample data before:\n", df[categorical_cols].head())

    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                encoded_missing_value=-2  # Handle NaN values
            ), categorical_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False  # Cleaner column names
    )

    encoded_data = preprocessor.fit_transform(df)
    df_encoded = pd.DataFrame(
        encoded_data,
        columns=preprocessor.get_feature_names_out(),
        index=df.index
    )
    new_columns = preprocessor.get_feature_names_out()
    print("\nSample data after encoding:")
    print(df_encoded[categorical_cols].head())
    print("\nData types after encoding:")
    print(df_encoded[categorical_cols].dtypes)

    df = df.drop(columns=categorical_cols, errors='ignore')

    df = df.join(df_encoded[new_columns[:len(categorical_cols)]])
    to_remove = df.columns[df.isna().mean() > 0.995]
    df = df.drop(columns=to_remove)
    df.to_csv('analysis/result/test.csv')
    X = df[df.columns.difference(['label'])]

    estimator = RandomForestRegressor(
        n_estimators=50,
        max_samples=0.5,
        max_features=0.8,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=21
    )
    complete_cases = y.notna()
    X_complete = X[complete_cases]
    y_complete = y[complete_cases]

    print("Training data shape:", X_complete.shape)
    estimator.fit(X=X_complete, y=y_complete)

    # 3. Create and apply imputer
    imputer = DataImputer(estimator=estimator)

    # Handle completely empty columns first
    numeric_cols = df.select_dtypes(include='number').columns
    print(numeric_cols)
    empty_cols = numeric_cols[df[numeric_cols].isna().all()]
    if not empty_cols.empty:
        print(f"Filling empty columns: {list(empty_cols)}")
        df[empty_cols].fillna(df.median())
    try:
        df_clean = imputer.transform(df)
        df_clean.to_csv('analysis/result/test_impute.csv')
    except Exception as e:
        warnings.warn(f"Imputation failed: {e}. Using simple imputer fallback")
        df_clean = pd.DataFrame(
            SimpleImputer(strategy='median').fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        df_clean.to_csv('analysis/result/test_impute_fallback.csv')