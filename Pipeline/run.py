import os
import re
import sys

from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from analysis.Pipeline.DataInspector import DataInspector
from Pipeline.pipeline import Pipeline
from Pipeline.transformers import CategoricalEncoder

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
    beers_and_breweries.to_csv('data/raw/beers_and_breweries.csv', index=False)
    return beers_and_breweries


def inspect_data(df):
    inspector = DataInspector(
        df=df,
        thresholds=[0.0, 0.99],
        exclude_columns=['id', 'brewery_id'],
    )

    print("=== DEFAULT REPORT ===")
    output_dir = 'analysis/report/week_3/data_inspection'
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
    output_dir = 'analysis/report/week_3'
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
                'DataProfiler': {
                    'name': 'analyze_correlations',
                    'output_dir':output_dir,
                    'strategy':'analyze_correlations',
                }
            },
            {
                'DataProfiler': {
                    'name': 'distributions',
                    'output_dir':output_dir,
                    'strategy':'analyze_distributions',
                }
            },
            {
                'DataProfiler': {
                    'name': 'group_by',
                    'output_dir':output_dir,
                    'strategy':'group_by',
                    'features':['style']
                }
            },
            {
                'DataProfiler': {
                    'name': 'one_sample',
                    'output_dir':output_dir,
                    'strategy':'one_sample',
                }
            },
            {
                'CorrelationHeatmap':{
                    'name': 'CorrelationHeatmap',
                    'strategy':'pearson',
                    'output_dir':output_dir
                    }
            },
        ]
    )
    df = pipeline.fit_transform(df)
    print(df)
    df.to_csv('data/processed/week_3_beers_and_breweries.csv', index=False)

def main():
    df = load_data()
    inspect_data(df=df)
    data_analysis(df)

if __name__ == '__main__':
    main()