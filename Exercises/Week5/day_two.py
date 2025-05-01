
def load_dataset():
    """Load the diabetes dataset."""
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
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

def plot_results(y_test, y_pred):
    import plotly.express as px
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'True Values', 'y': 'Predicted Values'}, title='True vs Predicted Values')
    fig.add_shape(type="line", x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test), line=dict(color="red", width=2))
    fig.update_layout(xaxis_title='True Values', yaxis_title='Predicted Values')
    fig.show()

def print_results(y_test, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, mean_squared_log_error, r2_score, root_mean_squared_error
    print("RMSE:", root_mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("Explained Variance:", explained_variance_score(y_test, y_pred))
    print("MSLE:", mean_squared_log_error(y_test, y_pred))

def main():
    X,y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    from sklearn.linear_model import LinearRegression
    intial_model = LinearRegression()
    y_pred = train_model(intial_model, X_train_scaled, y_train, X_test_scaled)
    print_results(y_test, y_pred)
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()