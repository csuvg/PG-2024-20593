from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import FeatureUnion
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn_pandas import DataFrameMapper

# Read data
data = pd.read_csv('../data/PROCESS/processed_tch_prediction_data_2Mesv7.csv')

# Create a mask for ZAFRA 23-24
mask_23_24 = data['ZAFRA'] == '23-24'

# Split data into training and testing
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
column_group_3 = ['tipo_cosecha']
column_group_2 = ['prod_mad', 'estacion', 'variedad', 'numero_corte',
                  'sist_riego', 'region', 'estrato', 'cuadrante', 'PRODUCTO_ACTUAL']

column_group_1 = [
    ['ETP_Q2_mean'], ['ETP_Q2_std'], [
        'Radiacion (MJ/m2)_Q2_mean'], ['Radiacion (MJ/m2)_Q2_std'],
    ['Amplitud Térmica_Q2_mean'], ['Amplitud Térmica_Q2_std'], [
        'R0_Q2_mean'], ['R0_Q2_std'],
    ['temperatura_Q2_mean'], ['temperatura_Q2_std'], [
        'humedad relativa_Q2_mean'], ['humedad relativa_Q2_std'],
    ['humedad relativa minima_Q2_mean'], ['humedad relativa minima_Q2_std'], [
        'humedad relativa maxima_Q2_mean'],
    ['humedad relativa maxima_Q2_std'], [
        'precipitacion_Q2_mean'], ['precipitacion_Q2_std'],
    ['velocidad viento_Q2_mean'], ['velocidad viento_Q2_std'], [
        'velocidad viento minima_Q2_mean'],
    ['velocidad viento minima_Q2_std'], ['velocidad viento maxima_Q2_mean'], [
        'velocidad viento maxima_Q2_std'],
    ['direccion viento_Q2_mean'], ['direccion viento_Q2_std'], [
        'NDVI_POND_Q2_mean'], ['NDVI_POND_Q2_std'],
    ['LAI_POND_Q2_mean'], ['LAI_POND_Q2_std'], [
        'PRODUCCION_PON_Q2_mean'], ['PRODUCCION_PON_Q2_std'],
    ['ANOMALIAS_POND_Q2_mean'], ['ANOMALIAS_POND_Q2_std'], [
        'HUMEDAD_POND_Q2_mean'], ['HUMEDAD_POND_Q2_std']
]
column_group_1_flat = [col[0]
                       for col in column_group_1 if col[0] in X_train.columns]

column_group_0 = [
    ['area'], ['ETP_Q2_mean'], ['ETP_Q2_sum'], [
        'ETP_Q2_std'], ['ETP_Q2_integral'],
    ['Radiacion (MJ/m2)_Q2_mean'], ['Radiacion (MJ/m2)_Q2_sum'], ['Radiacion (MJ/m2)_Q2_std'],
    ['Radiacion (MJ/m2)_Q2_integral'], ['Amplitud Térmica_Q2_mean'], ['Amplitud Térmica_Q2_sum'],
    ['Amplitud Térmica_Q2_std'], ['Amplitud Térmica_Q2_integral'], [
        'R0_Q2_mean'], ['R0_Q2_sum'],
    ['R0_Q2_std'], ['R0_Q2_integral'], ['temperatura_Q2_mean'], [
        'temperatura_Q2_sum'], ['temperatura_Q2_std'],
    ['temperatura_Q2_integral'], ['humedad relativa_Q2_mean'], [
        'humedad relativa_Q2_sum'],
    ['humedad relativa_Q2_std'], ['humedad relativa_Q2_integral'], [
        'humedad relativa minima_Q2_mean'],
    ['humedad relativa minima_Q2_sum'], ['humedad relativa minima_Q2_std'], [
        'humedad relativa minima_Q2_integral'],
    ['humedad relativa maxima_Q2_mean'], [
        'humedad relativa maxima_Q2_sum'], ['humedad relativa maxima_Q2_std'],
    ['humedad relativa maxima_Q2_integral'], [
        'precipitacion_Q2_mean'], ['precipitacion_Q2_sum'],
    ['precipitacion_Q2_std'], ['precipitacion_Q2_integral'], [
        'velocidad viento_Q2_mean'],
    ['velocidad viento_Q2_sum'], ['velocidad viento_Q2_std'], [
        'velocidad viento_Q2_integral'],
    ['velocidad viento minima_Q2_mean'], [
        'velocidad viento minima_Q2_sum'], ['velocidad viento minima_Q2_std'],
    ['velocidad viento minima_Q2_integral'], [
        'velocidad viento maxima_Q2_mean'], ['velocidad viento maxima_Q2_sum'],
    ['velocidad viento maxima_Q2_std'], [
        'velocidad viento maxima_Q2_integral'], ['direccion viento_Q2_mean'],
    ['direccion viento_Q2_sum'], ['direccion viento_Q2_std'], [
        'direccion viento_Q2_integral'],
    ['NDVI_POND_Q2_mean'], ['NDVI_POND_Q2_sum'], [
        'NDVI_POND_Q2_std'], ['NDVI_POND_Q2_integral'],
    ['LAI_POND_Q2_mean'], ['LAI_POND_Q2_sum'], [
        'LAI_POND_Q2_std'], ['LAI_POND_Q2_integral'],
    ['PRODUCCION_PON_Q2_mean'], [
        'PRODUCCION_PON_Q2_sum'], ['PRODUCCION_PON_Q2_std'],
    ['PRODUCCION_PON_Q2_integral'], [
        'ANOMALIAS_POND_Q2_mean'], ['ANOMALIAS_POND_Q2_sum'],
    ['ANOMALIAS_POND_Q2_std'], [
        'ANOMALIAS_POND_Q2_integral'], ['HUMEDAD_POND_Q2_mean'],
    ['HUMEDAD_POND_Q2_sum'], ['HUMEDAD_POND_Q2_std'], ['HUMEDAD_POND_Q2_integral']
]
column_group_0_flat = [col[0]
                       for col in column_group_0 if col[0] in X_train.columns]

column_group_2_final = [
    col for col in column_group_2 if col in X_train.columns]
column_group_3_final = [
    col for col in column_group_3 if col in X_train.columns]

# Define mappers
# Mapper 0: Numeric columns with SimpleImputer(strategy='mean')
mapper_0 = DataFrameMapper(
    [(column_group_0_flat, [SimpleImputer(strategy='mean')])],
    input_df=True,
    df_out=True
)

# Mapper 1: Numeric columns with SimpleImputer(strategy='mean', add_indicator=True)
mapper_1 = DataFrameMapper(
    [(column_group_1_flat, [SimpleImputer(strategy='mean', add_indicator=True)])],
    input_df=True,
    df_out=True
)

# Mapper 2: Categorical columns with SimpleImputer(strategy='most_frequent') and OneHotEncoder
mapper_2 = DataFrameMapper(
    [(column_group_2_final, [SimpleImputer(strategy='most_frequent'),
      OneHotEncoder(handle_unknown='ignore', sparse_output=False)])],
    input_df=True,
    df_out=False
)

# Mapper 3: Text columns with SimpleImputer(strategy='most_frequent') and OrdinalEncoder
mapper_3 = DataFrameMapper(
    [(column_group_3_final, [SimpleImputer(strategy='most_frequent'), OrdinalEncoder()])],
    input_df=True,
    df_out=True
)

# Combine mappers into FeatureUnion

feature_union = FeatureUnion(
    transformer_list=[
        ('mapper_0', mapper_0),
        ('mapper_1', mapper_1),
        ('mapper_2', mapper_2),
        ('mapper_3', mapper_3)
    ]
)

# Build the pipeline
pipeline = Pipeline([
    ('features', feature_union),
    ('model', XGBRegressor(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.3,
        subsample=0.5,
        colsample_bytree=0.5,
        reg_alpha=2.291666666666667,
        reg_lambda=0.8333333333333334,
        objective='reg:squarederror',
        random_state=42
    ))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse}")
print(f"Test R2 score: {r2}")

# Use SHAP to explain the model

# Get the transformed training data
X_train_transformed = pipeline.named_steps['features'].transform(X_train)
X_test_transformed = pipeline.named_steps['features'].transform(X_test)

# Get feature names


def get_feature_names(feature_union):
    feature_names = []
    for name, transformer in feature_union.transformer_list:
        if isinstance(transformer, DataFrameMapper):
            # Each transformer in features is (columns, transformers)
            for columns, transformers_list in transformer.features:
                # Initialize current feature names with input columns
                if isinstance(columns, list):
                    current_feature_names = columns.copy()
                else:
                    current_feature_names = [columns]
                for transformer_item in transformers_list:
                    if isinstance(transformer_item, SimpleImputer):
                        if transformer_item.add_indicator:
                            # For each column with missing values, an indicator is added
                            # Get indices of columns with missing values
                            missing_indices = transformer_item.indicator_.features_
                            missing_cols = [current_feature_names[i]
                                            for i in missing_indices]
                            indicator_features = [
                                f"{col}_missing_indicator" for col in missing_cols]
                            current_feature_names.extend(indicator_features)
                    elif isinstance(transformer_item, OneHotEncoder):
                        # Get feature names from OneHotEncoder
                        if hasattr(transformer_item, 'categories_'):
                            categories = transformer_item.categories_
                            col_feature_names = []
                            for col, cats in zip(columns, categories):
                                col_feature_names.extend(
                                    [f"{col}_{cat}" for cat in cats])
                            current_feature_names = col_feature_names
                    else:
                        # Other transformers do not change feature names
                        pass
                feature_names.extend(current_feature_names)
        else:
            pass  # Handle other cases if necessary
    return feature_names


feature_names = get_feature_names(pipeline.named_steps['features'])

# Ensure that the number of feature names matches the transformed data
print(f"Number of features in transformed data: {X_test_transformed.shape[1]}")
print(f"Number of feature names: {len(feature_names)}")

# Create a SHAP explainer
explainer = shap.TreeExplainer(pipeline.named_steps['model'])

# Compute SHAP values
shap_values = explainer.shap_values(X_test_transformed)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
