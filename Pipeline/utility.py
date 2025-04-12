from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_categorical_frequency(data, cat_col, figsize=(10, 6)):
    """
    Plot frequency distribution of categorical data with optimal label visibility.
    
    Parameters:
    - data: DataFrame containing the data
    - cat_col: Name of the categorical column to plot
    - figsize: Base figure size (width, height), width expands for more categories
    
    Returns:
    - matplotlib Figure object
    """
    n_unique = data[cat_col].nunique()
    max_categories = int(np.sqrt(n_unique)) if n_unique > 0 else 0

    if max_categories < 2:
        print(f"Skipping {cat_col} (too few unique categories: {n_unique})")
        return None

    # Get top categories
    top_cats = data[cat_col].value_counts().nlargest(max_categories).index
    filtered = data[data[cat_col].isin(top_cats)]
    
    # Dynamic width calculation (min 1 inch per category)
    dynamic_width = max(figsize[0], len(top_cats) * 1.2)
    fig, ax = plt.subplots(figsize=(dynamic_width, figsize[1]))
    
    # Create countplot
    sns.countplot(x=cat_col, data=filtered, order=top_cats, ax=ax)
    
    # Title and labels
    ax.set_title(f"Top {max_categories} Categories in '{cat_col}' (of {n_unique} total)")
    
    # Smart label rotation and alignment
    if len(top_cats) > 5:
        # For many categories: 45° rotation with right alignment
        ax.set_xticklabels(ax.get_xticklabels(), 
                          rotation=45, 
                          ha='right',
                          rotation_mode='anchor')
    else:
        # For few categories: horizontal with center alignment
        ax.set_xticklabels(ax.get_xticklabels(), 
                          ha='center')

    # Add percentage labels
    total = len(filtered)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                f'{height/total:.1%}',
                ha='center', 
                va='bottom',
                fontsize=9)

    # Adjust layout with extra padding for rotated labels
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add top padding
    
    return fig
if __name__ == '__main__':
    from run import load_data, data_analysis
    df= load_data()
    # df = data_analysis(df)

    for col in df.select_dtypes(include=['object']).columns:
        fig = plot_categorical_frequency(df, col, figsize=(10, 6))
        fig.savefig(f"analysis/report/week_3/frequency/frequency_by_{col}.png")