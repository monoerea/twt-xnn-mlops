from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
def load_dataset():
    """Load the diabetes dataset."""
    from sklearn.datasets import load_breast_cancer
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target
    return X, y
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

def train_model(model, X_train_scaled, y_train, X_test_scaled):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return y_pred

def print_results(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, classification_report
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def main():
    X,y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    from sklearn.linear_model import LogisticRegression
    intial_model = LogisticRegression()
    y_pred = train_model(intial_model, X_train_scaled, y_train, X_test_scaled)
    print_results(y_test, y_pred)
if __name__ == "__main__":
    main()