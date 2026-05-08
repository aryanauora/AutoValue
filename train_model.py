import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def main():
    print("Loading data...")
    # Load dataset
    df = pd.read_csv("car_details_final.csv")
    
    # Extract Make and Model from Name
    df['make'] = df['name'].apply(lambda x: str(x).split()[0])
    df['model'] = df['name'].apply(lambda x: " ".join(str(x).split()[1:]))
    
    # Save the original dataset for comparable cars lookups in the app
    joblib.dump(df, "dataset.joblib")
    print("Saved original dataset to dataset.joblib")

    # Features and Target
    # Features: make, year, miles_driven, fuel, seller_type, transmission, owner, engine, max_power, mileage_mpg, seats
    categorical_features = ['make', 'fuel', 'seller_type', 'transmission', 'owner']
    numeric_features = ['year', 'miles_driven', 'engine', 'max_power', 'mileage_mpg', 'seats']
    
    # Drop any rows with missing values in the specific columns
    df = df.dropna(subset=categorical_features + numeric_features + ['selling_price_usd'])
    
    X = df[categorical_features + numeric_features]
    y = df['selling_price_usd']
    
    # Split the data 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # --- Linear Regression Baseline ---
    print("\nTraining Linear Regression Baseline...")
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', LinearRegression())])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)
    print(f"Linear Regression RMSE: ${lr_rmse:,.2f}")
    print(f"Linear Regression R²: {lr_r2:.4f}")
    
    # --- Random Forest Regressor ---
    print("\nTraining Random Forest Regressor...")
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)
    print(f"Random Forest RMSE: ${rf_rmse:,.2f}")
    print(f"Random Forest R²: {rf_r2:.4f}")
    
    # Serialize the Random Forest model
    print("\nSerializing Random Forest model to model.joblib...")
    joblib.dump(rf_pipeline, "model.joblib")
    print("Done!")

if __name__ == "__main__":
    main()
