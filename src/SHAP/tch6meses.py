import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, Normalizer, MaxAbsScaler,
                                   OneHotEncoder)
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('../data/PROCESS/processed_tch_prediction_data_6Mesv7.csv')

# Crear una máscara para ZAFRA 23-24
mask_23_24 = data['ZAFRA'] == '23-24'

# Dividir los datos en entrenamiento y prueba
X_train = data[~mask_23_24].drop('TCH', axis=1)
y_train = data[~mask_23_24]['TCH']
X_test = data[mask_23_24].drop('TCH', axis=1)
y_test = data[mask_23_24]['TCH']

# Eliminar columnas innecesarias
X_train = X_train.drop(columns=['ABS_IDCOMP', 'ZAFRA', 'fecha', 'rendimiento'])
X_test = X_test.drop(columns=['ABS_IDCOMP', 'ZAFRA', 'fecha', 'rendimiento'])

print("\nZAFRA Split:")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(
    include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=['object', 'category']).columns.tolist()

print("\nNumerical columns:")
print(numerical_cols)
print("\nCategorical columns:")
print(categorical_cols)

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    # Scaler will be applied in individual pipelines
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create a base preprocessor pipeline
preprocessor_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Fit and transform the training data
X_train_preprocessed = preprocessor_pipeline.fit_transform(X_train)
X_test_preprocessed = preprocessor_pipeline.transform(X_test)

# Get feature names after preprocessing
feature_names = preprocessor_pipeline['preprocessor'].get_feature_names_out()

# Define individual pipelines with their respective preprocessors and models

# Pipeline 0
pipeline_0 = Pipeline(steps=[
    ('scaler', StandardScaler(with_mean=False)),
    ('model', LGBMRegressor(
        boosting_type='gbdt',
        colsample_bytree=0.4,
        learning_rate=0.1789484210526316,
        max_bin=1023,
        max_depth=8,
        min_child_samples=198,
        min_child_weight=0.001,
        min_split_gain=0.6842105263157894,
        n_estimators=600,
        num_leaves=255,
        reg_alpha=0.6,
        reg_lambda=0.825,
        subsample=0.7,
        subsample_for_bin=200000,
        subsample_freq=1,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    ))
])

# Pipeline 1
pipeline_1 = Pipeline(steps=[
    ('normalizer', Normalizer(norm='l1')),
    ('model', LGBMRegressor(
        boosting_type='gbdt',
        colsample_bytree=0.2,
        learning_rate=0.042113157894736845,
        max_bin=63,
        max_depth=9,
        min_child_samples=73,
        min_child_weight=0.001,
        min_split_gain=0.3684210526315789,
        n_estimators=800,
        num_leaves=63,
        reg_alpha=0.3,
        reg_lambda=0.6,
        subsample=0.75,
        subsample_for_bin=200000,
        subsample_freq=8,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    ))
])

# Pipeline 2
pipeline_2 = Pipeline(steps=[
    ('scaler', StandardScaler(with_mean=False)),
    ('model', LGBMRegressor(
        boosting_type='gbdt',
        colsample_bytree=0.6,
        learning_rate=0.16842263157894738,
        max_bin=1023,
        max_depth=9,
        min_child_samples=38,
        min_child_weight=0.001,
        min_split_gain=0.7368421052631579,
        n_estimators=100,
        num_leaves=255,
        reg_alpha=0,
        reg_lambda=0.75,
        subsample=0.9,
        subsample_for_bin=200000,
        subsample_freq=6,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    ))
])

# Pipeline 3
pipeline_3 = Pipeline(steps=[
    ('maxabs_scaler', MaxAbsScaler()),
    ('model', XGBRegressor(
        verbosity=0,
        n_jobs=-1,
        random_state=42
    ))
])

# Fit individual pipelines
pipeline_0.fit(X_train_preprocessed, y_train)
pipeline_1.fit(X_train_preprocessed, y_train)
pipeline_2.fit(X_train_preprocessed, y_train)
pipeline_3.fit(X_train_preprocessed, y_train)

# Create the ensemble model
ensemble = VotingRegressor(
    estimators=[
        ('model_0', pipeline_0),
        ('model_1', pipeline_1),
        ('model_2', pipeline_2),
        ('model_3', pipeline_3)
    ],
    weights=[0.46153846153846156, 0.23076923076923078,
             0.07692307692307693, 0.23076923076923078]
)

# Fit the ensemble model
ensemble.fit(X_train_preprocessed, y_train)

# Make predictions with the ensemble model
y_pred = ensemble.predict(X_test_preprocessed)

# Evaluate the ensemble model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nEnsemble Model Evaluation:")
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")

# Generate SHAP explanations for all pipelines
pipelines = [pipeline_0, pipeline_1, pipeline_2, pipeline_3]
pipeline_names = ['pipeline_0', 'pipeline_1', 'pipeline_2', 'pipeline_3']

for i, pipeline in enumerate(pipelines):
    print(f"\nGenerating SHAP explanations for {pipeline_names[i]}...")

    # Extract the model and scaler (if any)
    model = pipeline.named_steps['model']
    scaler = pipeline.named_steps.get('scaler') or pipeline.named_steps.get(
        'normalizer') or pipeline.named_steps.get('maxabs_scaler')

    # Transform the data as the model sees it
    if scaler:
        X_train_scaled = scaler.transform(X_train_preprocessed)
        X_test_scaled = scaler.transform(X_test_preprocessed)
    else:
        X_train_scaled = X_train_preprocessed
        X_test_scaled = X_test_preprocessed

    # Create the SHAP explainer (without check_additivity)
    explainer = shap.TreeExplainer(
        model,
        data=X_train_scaled,
        feature_names=feature_names
    )

    # Compute SHAP values for test data with check_additivity=False
    shap_values = explainer.shap_values(X_test_scaled, check_additivity=False)

    # Plot SHAP summary plot
    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=feature_names,
        #  show=False
    )
    # plt.title(f"SHAP Summary Plot for {pipeline_names[i]}")
    # plt.tight_layout()
    # plt.savefig(f'shap_summary_{pipeline_names[i]}.png')
    # plt.close()

    print(f"SHAP summary plot saved as shap_summary_{pipeline_names[i]}.png")
