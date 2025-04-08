import os
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd
from analysis.Pipeline.DataInspector import DataInspector
from analysis.Pipeline.Transformers import Cleaner, Imputer, FeatureEngineer
from analysis.Pipeline.Analyzers import DataProfiler
from analysis.Pipeline.Pipeline import Pipeline

def load_data():

    beers_url = "https://raw.githubusercontent.com/nickhould/craft-beers-dataset/master/data/processed/beers.csv"
    breweries_url = "https://raw.githubusercontent.com/nickhould/craft-beers-dataset/master/data/processed/breweries.csv"

    beers = pd.read_csv(beers_url)
    breweries = pd.read_csv(breweries_url)


    beers_and_breweries = pd.merge(beers,
                                breweries,
                                how='inner',
                                left_on="brewery_id",
                                right_on="id",
                                sort=True,
                                suffixes=('_beer', '_brewery'))
    return beers_and_breweries


def inspect_data(df):
    inspector = DataInspector(
        df=df,
        thresholds=[0.0, 0.99],
        exclude_columns=['id', 'brewery_id'],
    )

    print("=== DEFAULT REPORT ===")
    output_dir = 'Week3/analysis/data_inpection'
    strategies = [
            ('basic_info', {}),
            ('mcar', {'columns': None, 'pct': 0.05}),
            ('ttest', {'target_cols': None, 'group_col': None, }),
            ('outlier',{'threshold':5}),
            ('missing_heatmap', {'filename': f"{output_dir}/missing_heatmap.png"})
        ]
    inspector.generate_report(output_dir=output_dir, strategies=strategies)

def data_analysis(df):

    filepath = 'Week3/analysis/report'

    numeric_cols = df.select_dtypes(include=['number']).columns
    categorial_columns = df.select_dtypes(include=['object','bool']).columns
    print("=== NUMERIC COLUMNS ===\n", numeric_cols)
    print("=== CATEGORICAL COLUMNS ===\n", categorial_columns)

    remove_columns = False
    if remove_columns == True:
        remove_patterns = ['id']
        pattern = re.compile('|'.join(remove_patterns), flags=re.IGNORECASE)
        to_remove = [col for col in df.columns if pattern.search(col)]
        data = data[data.columns.difference(to_remove)]

    pipeline = Pipeline("DataProcessingPipeline")
    pipeline.add_step(Cleaner())
    pipeline.add_step(Imputer())
    pipeline.add_step(FeatureEngineer())
    pipeline.add_step(Imputer())
    pipeline.add_step(DataProfiler())
    processed_data = pipeline.run(df, {
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
        'FeatureEngineer': {
            'encode_categorical': True,
            'categorical_columns': df.select_dtypes(include=['object','bool']).columns,
            'encoding_method': 'ordinal'
        },
        'Imputer': {
            'strategy': 'simple',  # Options: 'simple', 'knn', 'iterative',
            'numeric_cols': numeric_cols,
            # 'n_neighbors': np.sqrt(len(numeric_cols)).astype(int),
        },
        'DataProfiler': {
            'output_dir': f'{filepath}',
            'analyze_correlations': True,
            'analyze_distributions': True
        }
    })
    print("Pipeline completed!")
    print("Final df shape:", processed_data.shape)

    has_null = processed_data.isna().mean()==0
    print(has_null)
    if all(has_null):
        processed_data.to_csv(f'{filepath}/cleaned_data.csv', index=False)
    else:
        print("Has null values.")
def main():
    df = load_data()
    inspect_data(df)
    data_analysis(df)

if __name__ == '__main__':
    main()