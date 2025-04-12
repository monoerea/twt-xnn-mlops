import os
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Pipeline import Pipeline
from Transformers import Cleaner, FeatureEngineer, Imputer
from Analyzers import DataProfiler


def test(df):
    columns = [
    "quoted_status.favorite_count",
    "quoted_status.retweet_count",
    "quoted_status.user.favourites_count",
    "quoted_status.user.followers_count",
    "quoted_status.user.friends_count",
    "quoted_status.user.listed_count",
    "quoted_status.user.statuses_count",
    "retweet_count",
    "retweeted",
    "retweeted_status.favorite_count",
    "retweeted_status.quoted_status.favorite_count",
    "retweeted_status.quoted_status.retweet_count",
    "retweeted_status.quoted_status.user.favourites_count",
    "retweeted_status.quoted_status.user.followers_count",
    "retweeted_status.quoted_status.user.friends_count",
    "retweeted_status.quoted_status.user.listed_count",
    "retweeted_status.quoted_status.user.statuses_count",
    "retweeted_status.retweet_count",
    "retweeted_status.user.favourites_count",
    "retweeted_status.user.followers_count",
    "retweeted_status.user.friends_count",
    "retweeted_status.user.listed_count",
    "retweeted_status.user.statuses_count",
    "truncated"
]
    column_data_types = {col: df[col].dtype for col in columns if col in df.columns}

    # Print the data types
    for col, dtype in column_data_types.items():
        print(f"Column: {col}, Data Type: {dtype}", df[col].isna().mean())

def main():
    # Load and prepare data
    data = pd.read_csv("data/processed/flattened_status.csv", low_memory=False)
    data = pd.DataFrame(data).drop_duplicates()
    remove_patterns = ['id','text','media','screen_name','mentions','description']
    pattern = re.compile('|'.join(remove_patterns), flags=re.IGNORECASE)
    # Get columns to remove
    to_remove = [col for col in data.columns if pattern.search(col)]
    data = data[data.columns.difference(to_remove)]
    print(data[data.columns[data.columns.str.contains('created_at', case=False, na=False)].tolist()].dtypes)
    numeric_cols = data.select_dtypes(include=['number']).columns
    print(numeric_cols)

    # Create and configure pipeline
    pipeline = Pipeline("DataProcessingPipeline")
    # Add cleaning step
    pipeline.add_step(Cleaner())
    pipeline.add_step(Imputer())
    # Add feature engineering step
    pipeline.add_step(FeatureEngineer())
    pipeline.add_step(FeatureEngineer())

    pipeline.add_step(Imputer())
    # Add analysis step
    pipeline.add_step(DataProfiler())
    # Run pipeline with configuration
    processed_data = pipeline.run(data, {
        'Cleaner': {
            'missing_threshold': 0.75,
            'outlier_method': 'iqr',
            'outlier_threshold': 0.05
        },
        'Impute': {
            'strategy': 'simple',# Options: 'simple', 'knn', 'iterative',
            'numeric_stratery':'median',
            'numeric_columns': numeric_cols
        },
        'FeatureEngineerDT': {
            'process_datetime': True,
        },
        'FeatureEngineer': {
            'encode_categorical': True,
            'categorical_columns': data.select_dtypes(include=['object','bool']).columns,
            'encoding_method': 'numerical'
        },
        # 'Imputer': {
        #     'strategy': 'iterative',  # Options: 'simple', 'knn', 'iterative',
        #     'numeric_cols': numeric_cols,
        #     'max_iter': 15,
        #     'random_state':21,
        #     'estimator': RandomForestRegressor(
        #                 n_estimators=10,
        #                 max_samples=0.5,
        #                 max_features=0.8,
        #                 min_samples_leaf=10,
        #                 n_jobs=-1,
        #                 random_state=21
        #                 )
        # },
        'DataProfiler': {
            'output_dir': 'analysis/report/graphs',
            'analyze_correlations': True,
            'analyze_distributions': True
        }
    })
    print("Pipeline completed!")
    print("Final data shape:", processed_data.shape)
    # Save processed data
    test(data)
    has_null = processed_data.isna().mean()==0
    print(has_null)
    if all(has_null):
        processed_data.to_csv('data/processed/_cleaned_data.csv', index=False)
    else:
        print("Has null values.")
if __name__ == "__main__":
    main()