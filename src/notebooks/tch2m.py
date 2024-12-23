import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('../data/PROCESS/processed_tch_prediction_data_2Mesv7.csv')

# Create a mask for ZAFRA 23-24
mask_23_24 = data['ZAFRA'] == '23-24'

# Split the data into training and testing sets
X_train = data[~mask_23_24].drop('TCH', axis=1)
y_train = data[~mask_23_24]['TCH']
X_test = data[mask_23_24].drop('TCH', axis=1)
y_test = data[mask_23_24]['TCH']

# Drop unnecessary columns
X_train = X_train.drop(columns=['ABS_IDCOMP', 'ZAFRA', 'fecha', 'rendimiento'])
X_test = X_test.drop(columns=['ABS_IDCOMP', 'ZAFRA', 'fecha', 'rendimiento'])

print("\nZAFRA Split:")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Define column groups
column_group_0 = ['area', 'ETP_Q2_mean', 'ETP_Q2_sum', 'ETP_Q2_std', 'ETP_Q2_integral',
                  'Radiacion (MJ/m2)_Q2_mean', 'Radiacion (MJ/m2)_Q2_sum', 'Radiacion (MJ/m2)_Q2_std', 'Radiacion (MJ/m2)_Q2_integral',
                  'Amplitud Térmica_Q2_mean', 'Amplitud Térmica_Q2_sum', 'Amplitud Térmica_Q2_std', 'Amplitud Térmica_Q2_integral',
                  'R0_Q2_mean', 'R0_Q2_sum', 'R0_Q2_std', 'R0_Q2_integral', 'temperatura_Q2_mean', 'temperatura_Q2_sum', 'temperatura_Q2_std', 'temperatura_Q2_integral',
                  'humedad relativa_Q2_mean', 'humedad relativa_Q2_sum', 'humedad relativa_Q2_std', 'humedad relativa_Q2_integral',
                  'humedad relativa minima_Q2_mean', 'humedad relativa minima_Q2_sum', 'humedad relativa minima_Q2_std', 'humedad relativa minima_Q2_integral',
                  'humedad relativa maxima_Q2_mean', 'humedad relativa maxima_Q2_sum', 'humedad relativa maxima_Q2_std', 'humedad relativa maxima_Q2_integral',
                  'precipitacion_Q2_mean', 'precipitacion_Q2_sum', 'precipitacion_Q2_std', 'precipitacion_Q2_integral',
                  'velocidad viento_Q2_mean', 'velocidad viento_Q2_sum', 'velocidad viento_Q2_std', 'velocidad viento_Q2_integral',
                  'velocidad viento minima_Q2_mean', 'velocidad viento minima_Q2_sum', 'velocidad viento minima_Q2_std', 'velocidad viento minima_Q2_integral',
                  'velocidad viento maxima_Q2_mean', 'velocidad viento maxima_Q2_sum', 'velocidad viento maxima_Q2_std', 'velocidad viento maxima_Q2_integral',
                  'direccion viento_Q2_mean', 'direccion viento_Q2_sum', 'direccion viento_Q2_std', 'direccion viento_Q2_integral',
                  'NDVI_POND_Q2_mean', 'NDVI_POND_Q2_sum', 'NDVI_POND_Q2_std', 'NDVI_POND_Q2_integral',
                  'LAI_POND_Q2_mean', 'LAI_POND_Q2_sum', 'LAI_POND_Q2_std', 'LAI_POND_Q2_integral',
                  'PRODUCCION_PON_Q2_mean', 'PRODUCCION_PON_Q2_sum', 'PRODUCCION_PON_Q2_std', 'PRODUCCION_PON_Q2_integral',
                  'ANOMALIAS_POND_Q2_mean', 'ANOMALIAS_POND_Q2_sum', 'ANOMALIAS_POND_Q2_std', 'ANOMALIAS_POND_Q2_integral',
                  'HUMEDAD_POND_Q2_mean', 'HUMEDAD_POND_Q2_sum', 'HUMEDAD_POND_Q2_std', 'HUMEDAD_POND_Q2_integral']

column_group_1 = ['ETP_Q2_mean', 'ETP_Q2_std',
                  'Radiacion (MJ/m2)_Q2_mean', 'Radiacion (MJ/m2)_Q2_std',
                  'Amplitud Térmica_Q2_mean', 'Amplitud Térmica_Q2_std',
                  'R0_Q2_mean', 'R0_Q2_std',
                  'temperatura_Q2_mean', 'temperatura_Q2_std',
                  'humedad relativa_Q2_mean', 'humedad relativa_Q2_std',
                  'humedad relativa minima_Q2_mean', 'humedad relativa minima_Q2_std',
                  'humedad relativa maxima_Q2_mean', 'humedad relativa maxima_Q2_std',
                  'precipitacion_Q2_mean', 'precipitacion_Q2_std',
                  'velocidad viento_Q2_mean', 'velocidad viento_Q2_std',
                  'velocidad viento minima_Q2_mean', 'velocidad viento minima_Q2_std',
                  'velocidad viento maxima_Q2_mean', 'velocidad viento maxima_Q2_std',
                  'direccion viento_Q2_mean', 'direccion viento_Q2_std',
                  'NDVI_POND_Q2_mean', 'NDVI_POND_Q2_std',
                  'LAI_POND_Q2_mean', 'LAI_POND_Q2_std',
                  'PRODUCCION_PON_Q2_mean', 'PRODUCCION_PON_Q2_std',
                  'ANOMALIAS_POND_Q2_mean', 'ANOMALIAS_POND_Q2_std',
                  'HUMEDAD_POND_Q2_mean', 'HUMEDAD_POND_Q2_std']

column_group_2 = ['prod_mad', 'estacion', 'variedad', 'numero_corte', 'sist_riego', 'region', 'estrato', 'cuadrante', 'PRODUCTO_ACTUAL']

column_group_3 = ['tipo_cosecha']

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num0', SimpleImputer(strategy='mean'), column_group_0),
        ('num1', SimpleImputer(strategy='mean'), column_group_1),
        ('cat2', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), column_group_2),
        ('cat3', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder())
        ]), column_group_3),
    ],
    remainder='drop'
)

# Define individual pipelines
pipeline_0 = Pipeline(steps=[
    ('preproc', StandardScaler(with_mean=False, with_std=False)),
    ('model', XGBRegressor(
        colsample_bytree=0.5,
        eta=0.3,
        max_depth=7,
        n_estimators=800,
        reg_alpha=2.291666666666667,
        reg_lambda=0.8333333333333334,
        subsample=0.5,
        verbosity=0
    ))
])

pipeline_1 = Pipeline(steps=[
    ('preproc', StandardScaler(with_mean=False, with_std=False)),
    ('model', XGBRegressor(
        colsample_bytree=0.9,
        eta=0.5,
        gamma=0.01,
        learning_rate=0.5,
        max_depth=8,
        n_estimators=50,
        reg_alpha=1.3541666666666667,
        reg_lambda=1.6666666666666667,
        subsample=1,
        verbosity=0
    ))
])

pipeline_2 = Pipeline(steps=[
    ('preproc', MaxAbsScaler()),
    ('model', XGBRegressor(
        max_depth=6,
        verbosity=0
    ))
])

# Define the ensemble model
ensemble = VotingRegressor(
    estimators=[
        ('model_0', pipeline_0),
        ('model_1', pipeline_1),
        ('model_2', pipeline_2)
    ],
    weights=[0.5714285714285714, 0.2857142857142857, 0.14285714285714285]
)

# Build the final model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('ensemble', ensemble)
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse}")
print(f"Test R2: {r2}")

# Use SHAP to explain the model
# Fit the preprocessor and get feature names
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()

# Get transformed data
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Create a DataFrame for transformed test data
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

# Fit pipeline_0 individually
pipeline_0.fit(X_train_transformed, y_train)

# Create SHAP explainer
explainer = shap.Explainer(pipeline_0.named_steps['model'])

# Compute SHAP values
shap_values = explainer(X_test_transformed)

# Plot SHAP summary plot
shap.summary_plot(shap_values, X_test_transformed_df, feature_names=feature_names, max_display=20)
plt.show()

# Plot predicted vs true values
plt.figure(figsize=(10, 6))
sns.lineplot(x=y_test, y=y_pred, label='Average Predicted Value', color='blue')
sns.lineplot(x=y_test, y=y_test, label='Ideal', linestyle='--', color='green')

# Create a histogram to show bin counts
sns.histplot(y_test, bins=30, alpha=0.5, label='Bin Count', color='blue')

plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Predicted vs. True')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()