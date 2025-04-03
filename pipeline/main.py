import numpy as np
import pandas as pd
import os
from Pipeline import Pipeline
from Analyzers import (
    DataProfiler, 
    DynamicAnalyzer, 
    MissingValueAnalyzer, 
    CorrelationAnalysisStrategy,
    TimeSeriesAnalysisStrategy,
    AnalysisReport
)
from Transformers import (
    DataCleaner,
    DynamicTransformer,
    MissingValueStrategy,
    OutlierHandlingStrategy,
    TextCleaningStrategy
)

def save_cleaned_data(df, base_filename="cleaned_status"):
    """
    Save cleaned data with incremental numbering based on file size
    Returns the path to the saved file
    """
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the next available file number
    counter = 1
    while True:
        # Format number with leading zeros (e.g., 001k for 1k)
        size_suffix = f"{counter:03d}k"
        filename = f"{base_filename}_{size_suffix}.csv"
        filepath = os.path.join(output_dir, filename)
        
        if not os.path.exists(filepath):
            break
        counter += 1
    
    # Save the file
    df.to_csv(filepath, index=False)
    print(f"\nSaved cleaned data to: {filepath}")
    return filepath

def main():
    # Create a data processing pipeline with both transformation and analysis steps
    pipeline = Pipeline(name="DataProcessingPipeline")
    
    # Register common strategies
    pipeline.register_step(DataCleaner, "data_cleaner")
    pipeline.register_step(DynamicTransformer, "dynamic_transformer")
    
    # Add transformation steps
    pipeline.add_step(
        "data_cleaner",
        strategies=[
            "MissingValueStrategy",
            "OutlierHandlingStrategy",
            "TextCleaningStrategy"
        ],
        column_patterns={
            "TextCleaningStrategy": ".*text.*|.*description.*"
        }
    )
    
    pipeline.add_step(
        "dynamic_transformer",
        operations=[
            {
                "type": "impute",
                "columns": None,
                "strategy": "median"
            },
            {
                "type": "encode",
                "columns": ["category"],
                "method": "onehot"
            },
            {
                "type": "datetime",
                "columns": ["date"],
                "extract_components": True
            }
        ]
    )
    
    # Add analysis steps
    pipeline.add_step(
        DataProfiler(
            strategies=[
                "CorrelationAnalysisStrategy",
                "TimeSeriesAnalysisStrategy"
            ],
            output_dir="reports/profiler"
        )
    )
    
    pipeline.add_step(
        MissingValueAnalyzer(
            output_dir="reports/missing_values",
            run_littles_test=True
        )
    )
    
    pipeline.add_step(
        DynamicAnalyzer(
            operations=[
                {"type": "distribution", "columns": None, "plot": True},
                {"type": "outliers", "method": "iqr", "threshold": 1.5, "plot": True}
            ],
            output_dir="reports/dynamic_analysis"
        )
    )
    
    # Load your data
    try:
        data = pd.read_csv('data/processed/flattened_status.csv')
        print("Successfully loaded data from 'data/processed/flattened_status.csv'")
    except FileNotFoundError:
        print("File 'data/processed/flattened_status.csv' not found. Using sample data instead.")
        # Create a sample dataframe for demonstration
        data = pd.DataFrame({
            'id': range(1, 101),
            'value': np.random.normal(50, 15, 100),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'text_data': ['Sample text ' + str(i) for i in range(100)],
            'missing_col': [np.nan if i % 10 == 0 else i for i in range(100)]
        })
        # Add some outliers
        data.loc[[5, 15, 25], 'value'] = [150, -30, 200]
    
    # Execute the pipeline
    try:
        processed_data = pipeline.run(data)
        print(f"\nPipeline executed successfully. Final data shape: {processed_data.shape}")
        
        # Save the cleaned data
        cleaned_filepath = save_cleaned_data(processed_data)
        
        # Generate a comprehensive report
        report = AnalysisReport(output_dir="final_report")
        report_path = report.generate(
            processed_data,
            title="Comprehensive Data Analysis Report",
            include_sections=[
                "basic_stats", 
                "missing_values", 
                "distributions",
                "correlations", 
                "outliers"
            ]
        )
        print(f"\nReport generated at: {report_path}")
        
        # Print summary of saved data
        file_size = os.path.getsize(cleaned_filepath) / 1024  # Size in KB
        print("\nFinal cleaned data summary:")
        print(f"- File: {cleaned_filepath}")
        print(f"- Size: {file_size:.2f} KB")
        print(f"- Rows: {len(processed_data)}")
        print(f"- Columns: {len(processed_data.columns)}")
        
    except Exception as e:
        print(f"\nError during pipeline execution: {str(e)}")

if __name__ == "__main__":
    main()