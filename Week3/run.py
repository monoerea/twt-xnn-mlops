import os
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from analysis.Pipeline.DataInspector import DataInspector
from Week3.pipeline import Pipeline

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
    pipeline = Pipeline(
        steps=[
            {
                'MissingValueHandler': {
                        'strategy': 'removed_missing',
                        'minimum': 0.75,
                        'maximum': 1,
                        'to_iterate': False
                    }
                }
            ,
        ]
    )
    df = pipeline.fit_transform(df)
    print(df)

def main():
    df = load_data()
    data_analysis(df)

if __name__ == '__main__':
    main()