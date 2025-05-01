


def load_dataset():
    """Load the diabetes dataset."""
    from sklearn.datasets import load_iris
    dataset = load_iris()
    X, y = dataset.data, dataset.target
    return X, y, dataset
def split_data(X, y):
    from sklearn.model_selection import train_test_split
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    from sklearn.discriminant_analysis import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(model, X_train_scaled, y_train):
    model.fit(X_train_scaled, y_train)
    return model

def print_results(tree, dataset):
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plot_tree(tree, filled=True, feature_names=dataset.feature_names)
    plt.show()

def main():
    X,y, dataset = load_dataset()
    X_train, X_test, y_train, _ = split_data(X,y)
    X_train_scaled, _ = scale_data(X_train, X_test)

    from sklearn.tree import DecisionTreeClassifier
    intial_model = DecisionTreeClassifier(max_depth=3)
    model = train_model(intial_model, X_train_scaled, y_train)
    print_results(model, dataset)
if __name__ == "__main__":
    main()





