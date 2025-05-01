from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
def load_dataset():
    """Load the diabetes dataset."""
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    return X, y
def split_data(X, y):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
def scale_data(X_train, X_test):
    """Scale the data using StandardScaler."""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("RMSE:", root_mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    return y_pred
def plot_results(y_test, y_pred):
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'True Values', 'y': 'Predicted Values'}, title='True vs Predicted Values')
    fig.add_shape(type="line", x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test), line=dict(color="red", width=2))
    fig.update_layout(xaxis_title='True Values', yaxis_title='Predicted Values')
    fig.show()
def print_results(y_test, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, mean_squared_log_error
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("Explained Variance:", explained_variance_score(y_test, y_pred))
    print("MSLE:", mean_squared_log_error(y_test, y_pred))
    plot_results(y_test, y_pred)
def main():
    X,y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    intial_model = LinearRegression()
    y_pred = train_model(intial_model, X_train_scaled, y_train, X_test_scaled, y_test)
    print_results(y_test, y_pred)
    param_grid = {
    'standardscaler__with_mean': [True, False],
    'standardscaler__with_std': [True, False],
    'linearregression__fit_intercept': [True, False],
    'linearregression__copy_X': [True, False],
    'linearregression__n_jobs': [-1]
    }
    pipe = make_pipeline(StandardScaler(), LinearRegression(), memory='cache')
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error', 'explained_variance'],
        n_jobs=-1,
        cv=5,
        verbose=3,
        refit='r2',
        error_score='raise',
        return_train_score=True
        )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)
    print("Best cross-validation score: ", grid_search.best_score_)

    lr_best_params = {
    key.replace('linearregression__', ''): value
    for key, value in grid_search.best_params_.items()
    if key.startswith('linearregression__')
    }
    intial_model = LinearRegression().set_params(**lr_best_params)
    y_pred = train_model(intial_model, X_train_scaled, y_train, X_test_scaled, y_test)
    print_results(y_test, y_pred)

if __name__ == "__main__":
    main()