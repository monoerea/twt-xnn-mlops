import os
import re
import sys

from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from analysis.Pipeline.DataInspector import DataInspector
from Pipeline.pipeline import Pipeline
from Pipeline.transformers import CategoricalEncoder, TargetEncoder

import pandas as pd

def load_data_breweries():

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
    print(beers_and_breweries.columns)
    beers_and_breweries.drop(columns=['Unnamed: 0_brewery','Unnamed: 0_beer', 'id_brewery'] , inplace=True, axis=1)
    beers_and_breweries.to_csv('data/raw/beers_and_breweries.csv', index=False)
    return beers_and_breweries

def load_data_twitter():
    data = pd.read_csv("data/processed/flattened_status.csv", low_memory=False)
    data = pd.DataFrame(data).drop_duplicates()
    # remove_patterns = ['id','text','media','screen_name','mentions','description']
    # pattern = re.compile('|'.join(remove_patterns), flags=re.IGNORECASE)
    # # Get columns to remove
    # to_remove = [col for col in data.columns if pattern.search(col)]
    # data = data[data.columns.difference(to_remove)]
    # print(data[data.columns[data.columns.str.contains('created_at', case=False, na=False)].tolist()].dtypes)
    return data
def inspect_data(df, output_dir='analysis/report/week_4/data_inspection'):
    remove_patterns = ['id','text','media','screen_name','mentions','description']
    pattern = re.compile('|'.join(remove_patterns), flags=re.IGNORECASE)
    to_skip = [col for col in df.columns if pattern.search(col)]
    inspector = DataInspector(
        df=df,
        thresholds=[0.0, 0.99],
        exclude_columns=to_skip,
    )

    print("=== DEFAULT REPORT ===")
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

    pipeline = Pipeline(
        steps=[
            {
                'CategoricalEncoder': {
                        'name': 'categorical_encoder',
                        'strategy': 'target',
                        'columns':  df.select_dtypes(include=['object']).columns.to_list(),
                    },
            },
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
                        'estimator': RandomForestRegressor(
                                        n_estimators=10,
                                        max_depth=5,
                                        random_state=21,
                                        n_jobs=-1
                                        ),
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
                    'features':df.columns.to_list()
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
def data_analysis_twitter(df):
    pipeline = Pipeline(steps=[
        {'MissingValueRemover': {
                'name':'missing_value_remover',
                'strategy':'remove_missing',
                'minimum': 0.75,
                'maximum': .95,
                'to_iterate': False}
         },])
    df = pipeline.fit_transform(df)
    print(df.head())
def main():
    df = load_data_twitter()
    inspect_data(df=df, output_dir='analysis/report/week_4/data_inspection')
    data_analysis_twitter(df)

if __name__ == '__main__':
    main()