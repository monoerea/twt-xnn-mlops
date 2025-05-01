import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def perform_task():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {k: v for k, v in locals().items() if hasattr(v, 'shape') and not k.startswith("__") and not isinstance(v, StandardScaler)}

def check_shape(name, array, expected_shapes):
    """Check if the shape of the variable matches the expected shape."""
    if "train" in name:
        target_key = "X_train" if "X" in name else "y_train"
    elif "test" in name:
        target_key = "X_test" if "X" in name else "y_test"
    else:
        target_key = None
    expected_shape = expected_shapes[target_key]
    if array.shape == expected_shape:
        print(f"{name} shape matches the expected value: {expected_shape}")
    else:
        print(f"Shape mismatch for {name}. Expected {expected_shape}, got {array.shape}")

def check_nans(name, array):
    if np.any(np.isnan(array)):
        print(f"NaN found in {name}")
    else:
        print(f"No NaN values found in {name}.")

def check_scaling(name, array):
    if "scaled" in name:
        mean_value = np.mean(array, axis=0)
        std_value = np.std(array, axis=0)
        mean_check = np.allclose(mean_value, 0, atol=1e-1)
        std_check = np.allclose(std_value, 1, atol=1e-1)

        if mean_check and std_check:
            print(f"{name} is properly scaled. Mean: {mean_value}, Std: {std_value}")
        else:
            print(f"Scaling issue in {name}: Mean = {mean_value}, Std = {std_value}")
    else:
        print(f"No scaling check needed for {name}.")

def perform_test(variables):
    expected_shapes = {
        "X_train": variables["X_train"].shape,
        "X_test": variables["X_test"].shape,
        "y_train": variables["y_train"].shape,
        "y_test": variables["y_test"].shape
    }
    for name, array in variables.items():
        if not hasattr(array, 'shape'):
            continue

        print(f"\n--- Testing {name} ---")
        check_shape(name, array, expected_shapes)
        check_nans(name, array)
        check_scaling(name, array)

    print("\nAll tests completed.")

def main():
    variables = perform_task()
    perform_test(variables)
if __name__ == "__main__":
    main()
