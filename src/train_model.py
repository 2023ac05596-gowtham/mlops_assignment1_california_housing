import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import os

# Load the dataset
data = pd.read_csv("data/processed/california_housing.csv")
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set MLflow experiment
mlflow.set_experiment("CaliforniaHousingModels")

# Dictionary to store model results
results = {}

# Train and log Linear Regression
with mlflow.start_run(run_name="LinearRegression") as run_lr:
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(
        lr, "model", registered_model_name="LinearRegression_California"
    )

    print(f"LinearRegression - RMSE: {rmse:.2f}, R2: {r2:.2f}")
    results["LinearRegression"] = {
        "rmse": rmse,
        "model": lr,
        "run_id": run_lr.info.run_id,
    }

# Train and log Decision Tree
with mlflow.start_run(run_name="DecisionTree") as run_dt:
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param("model", "DecisionTree")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(
        dt, "model", registered_model_name="DecisionTree_California"
    )

    print(f"DecisionTree - RMSE: {rmse:.2f}, R2: {r2:.2f}")
    results["DecisionTree"] = {
        "rmse": rmse,
        "model": dt,
        "run_id": run_dt.info.run_id,
    }

# Save models locally for DVC tracking
os.makedirs("models", exist_ok=True)
joblib.dump(
    results["LinearRegression"]["model"],
    "models/LinearRegression.pkl"
)
joblib.dump(results["DecisionTree"]["model"], "models/DecisionTree.pkl")

# Pick best model based on lowest RMSE
best_model_name = min(results, key=lambda k: results[k]["rmse"])
print(
    (
        f"\n✅ Best Model: {best_model_name} "
        f"(Run ID: {results[best_model_name]['run_id']})"
    )
)
print(
    "📁 Models saved to: models/LinearRegression.pkl, "
    "models/DecisionTree.pkl"
)
