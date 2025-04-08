import os
import re
import sys

from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from analysis.Pipeline.DataInspector import DataInspector
from Week3.pipeline import Pipeline
from Week3.transformers import CategoricalEncoder

import pandas as pd

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
    beers_and_breweries.to_csv('Week3/analysis/beers_and_breweries.csv', index=False)
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
    print(df.head())
    object_columns = df.select_dtypes(include=['object']).columns
    print("Object columns:", object_columns)
    estimator = RandomForestRegressor(
                                        n_estimators=10,
                                        max_depth=5,
                                        random_state=21,
                                        n_jobs=-1
                                        )
    encoder  = CategoricalEncoder()

    df = encoder.fit_transform(df, {'ordinal': object_columns})
    print(df[object_columns].head())
    pipeline = Pipeline(
        steps=[
            {
                'MissingValueRemover': {
                        'name': 'missing_value_remover',
                        'strategy': 'remove_missing',
                        'minimum': 0.75,
                        'maximum': 1,
                        'to_iterate': False
                    },
                }
            ,
            {
                'DataImputer': {
                        'name': 'data_transformer',
                        'strategy': 'iterative',
                        'estimator': estimator,
                        'params': {
                            'max_iter': 10,
                            'imputation_order': 'ascending',
                            'skip_complete': False,
                        },
                    },
                },
            {
                'DataScaler': {
                        'name': 'log_scaler',
                        'strategy': 'log',
                        'params': {
                            'base': 10,
                        },
                    },
                },
            {
                'CorrelationHeatmap':{
                    'strategy':'pearson',
                    }}
        ]
    )
    df = pipeline.fit_transform(df)
    print(df)
    df.to_csv('Week3/analysis/processed_data.csv', index=False)

def main():
    df = load_data()
    data_analysis(df)

if __name__ == '__main__':
    main()