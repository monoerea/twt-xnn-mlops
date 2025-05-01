from collections import Counter
import numpy as np
import pytest
from day_one import main as training_data
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from collections import Counter

def main():
    iris = load_iris()
    variables = training_data()
    test_main(variables, iris)

def test_main(variables, iris):
    expected_shapes = {
        "X_train": variables["X_train"].shape,
        "X_test": variables["X_test"].shape,
        "y_train": variables["y_train"].shape,
        "y_test": variables["y_test"].shape
    }

    for name, array in variables.items():
        print(f"Testing {name}: {array.shape if hasattr(array, 'shape') else 'N/A'}")

        if "train" in name:
            target_key = "X_train" if "X" in name else "y_train"
        elif "test" in name:
            target_key = "X_test" if "X" in name else "y_test"
        else:
            continue
        # print(f"Expected {name} shape: {expected_shapes[target_key]}")
        assert array.shape == expected_shapes[target_key], f"{name} shape mismatch"

        if np.issubdtype(array.dtype, np.floating):
            assert not np.any(np.isnan(array)), f"{name} contains NaNs"

        if "scaled" in name:
            assert np.allclose(np.mean(array, axis=0), 0, atol=1e-2) and np.allclose(np.std(array, axis=0), 1, atol=1e-2), f"{name} not scaled correctly"

        if "y" in name:

            distribution = Counter(array)
            total_samples = len(array)
            print(f"Class distribution for {name}: {distribution}")

            assert all(distribution[class_] / total_samples == Counter(iris.target)[class_] / len(iris.target) for class_ in distribution), f"{name} class distribution mismatch"

if __name__ == "__main__":
    main()