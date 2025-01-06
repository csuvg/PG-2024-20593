import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('../data/PROCESS/processed_tch_prediction_data_4Mesv7.csv')
# data = pd.read_csv('../data/UP/TEST_2Mesv7.csv')

# Create mask for ZAFRA 23-24
mask_23_24 = data['ZAFRA'] == '23-24'

# Split data into training and test sets
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

# Prepare data: remove rows with NaNs


def prepare_data(X, y):
    data = X.copy()
    data['TCH'] = y
    data = data.dropna()
    y = data['TCH']
    X = data.drop('TCH', axis=1)
    return X, y


X_train, y_train = prepare_data(X_train, y_train)
X_test, y_test = prepare_data(X_test, y_test)

# Define column groups based on your data
categorical_columns_cv = [
    'prod_mad', 'estacion', 'variedad', 'numero_corte', 'sist_riego',
    'region', 'estrato', 'cuadrante', 'PRODUCTO_ACTUAL'
]

categorical_columns_le = [
    'tipo_cosecha'
]

# Automatically select numerical and other categorical columns
numerical_columns = selector(dtype_include=np.number)(X_train)
numerical_columns = [
    col for col in numerical_columns if col not in categorical_columns_cv + categorical_columns_le]

# Define transformers for each group
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_onehot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

categorical_ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])

# Combine transformers into ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat_onehot', categorical_onehot_transformer, categorical_columns_cv),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_columns_le),
    ],
    remainder='drop'  # Or 'passthrough' if you prefer
)

# Define the first pipeline without the scaler
pipeline_0 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        colsample_bytree=0.9,
        eta=0.5,
        gamma=0.01,
        learning_rate=0.5,
        max_depth=8,
        n_estimators=50,
        reg_alpha=1.3541666666666667,
        reg_lambda=1.6666666666666667,
        verbosity=0,
        random_state=0
    ))
])

# Define the second pipeline
pipeline_1 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', MaxAbsScaler()),
    ('regressor', XGBRegressor(
        colsample_bytree=1,
        gamma=0,
        learning_rate=0.3,
        max_depth=6,
        n_estimators=100,
        reg_alpha=0,
        reg_lambda=1,
        verbosity=0,
        random_state=0
    ))
])

# Combine pipelines into a VotingRegressor
voting_regressor = VotingRegressor(
    estimators=[
        ('model_0', pipeline_0),
        ('model_1', pipeline_1)
    ],
    weights=[0.5714285714285714, 0.42857142857142855]
)

# Fit the model
voting_regressor.fit(X_train, y_train)

# Access the fitted pipeline_0 from the voting regressor
fitted_pipeline_0 = voting_regressor.named_estimators_['model_0']

# Function to get preprocessed data


def get_preprocessed_data(pipeline, X):
    preprocessor = pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(X)
    return X_transformed


# Get preprocessed data using the fitted pipeline
X_train_transformed = get_preprocessed_data(fitted_pipeline_0, X_train)
X_test_transformed = get_preprocessed_data(fitted_pipeline_0, X_test)

print("Shape of X_test_transformed:", X_test_transformed.shape)

# Get feature names after preprocessing
feature_names = fitted_pipeline_0.named_steps['preprocessor'].get_feature_names_out(
)
print("Number of feature names:", len(feature_names))

# Evaluate the model
y_pred = voting_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")


explainer = shap.Explainer(fitted_pipeline_0.named_steps['regressor'])

# Create DataFrame with transformed features
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

# Calculate SHAP values
shap_values = explainer(X_test_transformed)

# Plot SHAP summary and save the figure
shap.summary_plot(shap_values, X_test_transformed_df,
                  feature_names=feature_names, show=False)
plt.savefig('model_0_shap_4m.png')  # Guardamos el gráfico
plt.close()  # Cerramos la figura para evitar sobreposición

# Repeat for the second pipeline
fitted_pipeline_1 = voting_regressor.named_estimators_['model_1']
X_train_transformed_1 = get_preprocessed_data(fitted_pipeline_1, X_train)
X_test_transformed_1 = get_preprocessed_data(fitted_pipeline_1, X_test)
feature_names_1 = fitted_pipeline_1.named_steps['preprocessor'].get_feature_names_out(
)
explainer_1 = shap.Explainer(fitted_pipeline_1.named_steps['regressor'])
X_test_transformed_df_1 = pd.DataFrame(
    X_test_transformed_1, columns=feature_names_1)
shap_values_1 = explainer_1(X_test_transformed_1)
shap.summary_plot(shap_values_1, X_test_transformed_df_1,
                  feature_names=feature_names_1, show=False)
plt.savefig('model_1_shap_4m.png')  # Guardamos el gráfico
plt.close()
