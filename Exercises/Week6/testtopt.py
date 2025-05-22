import pandas as pd
from tpot import TPOTClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Minimal working example
data = pd.read_csv("Exercises/Week5/imputed_data.csv")

y = data['label']
X = data.drop(columns='label')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tpot = TPOTClassifier(
    generations=2,
    population_size=5,
    verbosity=2,
    random_state=42,
    config_dict={
        'sklearn.ensemble.RandomForestClassifier': {},
        'sklearn.svm.SVC': {},
        'sklearn.linear_model.LogisticRegression': {}
    },
    template='Classifier'  # Forces simple structure
)

tpot.fit(X_train, y_train)
print(f"Test score: {tpot.score(X_test, y_test)}")