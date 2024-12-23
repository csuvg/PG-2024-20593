import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Import scikit-learn and other required packages
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('../data/PROCESS/processed_tch_prediction_data_8Mesv7.csv')

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

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print("\nNumerical columns:", numerical_cols)
print("\nCategorical columns:", categorical_cols)


# Define data transformation using ColumnTransformer
def generate_data_transformation_config(numerical_cols, categorical_cols):
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean', missing_values=np.nan)),
        # ('scaler', StandardScaler(with_mean=False, with_std=False))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        sparse_threshold=0  # Ensures the output is a sparse matrix if any transformer outputs sparse
    )
    return preprocessor


# Define preprocessor configurations (you can adjust scalers as needed)
def generate_preprocessor_config_0():
    from sklearn.preprocessing import StandardScaler
    preproc = StandardScaler(
        copy=True,
        with_mean=False,
        with_std=False
    )
    return preproc


def generate_preprocessor_config_1():
    from sklearn.preprocessing import StandardScaler
    preproc = StandardScaler(
        copy=True,
        with_mean=False,
        with_std=False
    )
    return preproc


def generate_preprocessor_config_2():
    from sklearn.preprocessing import MaxAbsScaler
    preproc = MaxAbsScaler(
        copy=True
    )
    return preproc


# Define algorithm configurations
def generate_algorithm_config_0():
    from xgboost import XGBRegressor
    algorithm = XGBRegressor(
        colsample_bytree=0.6,
        eta=0.3,
        max_depth=8,
        max_leaves=3,
        reg_alpha=1.25,
        reg_lambda=0.10416666666666667,
        subsample=0.7,
        n_estimators=100,
        verbosity=0,
        objective='reg:squarederror',
    )
    return algorithm


def generate_algorithm_config_1():
    from xgboost import XGBRegressor
    algorithm = XGBRegressor(
        colsample_bytree=0.9,
        eta=0.5,
        gamma=0.01,
        max_depth=8,
        max_leaves=0,
        reg_alpha=1.3541666666666667,
        reg_lambda=1.6666666666666667,
        subsample=1,
        n_estimators=50,
        verbosity=0,
        objective='reg:squarederror',
    )
    return algorithm


def generate_algorithm_config_2():
    from xgboost import XGBRegressor
    algorithm = XGBRegressor(
        colsample_bytree=1,
        eta=0.3,
        max_depth=6,
        reg_alpha=0,
        reg_lambda=1,
        subsample=1,
        n_estimators=100,
        verbosity=0,
        objective='reg:squarederror',
    )
    return algorithm


def generate_algorithm_config():
    from sklearn.ensemble import VotingRegressor
    pipeline_0 = Pipeline(
        steps=[('preproc', generate_preprocessor_config_0()), ('model', generate_algorithm_config_0())])
    pipeline_1 = Pipeline(
        steps=[('preproc', generate_preprocessor_config_1()), ('model', generate_algorithm_config_1())])
    pipeline_2 = Pipeline(
        steps=[('preproc', generate_preprocessor_config_2()), ('model', generate_algorithm_config_2())])
    algorithm = VotingRegressor(
        estimators=[
            ('model_0', pipeline_0),
            ('model_1', pipeline_1),
            ('model_2', pipeline_2),
        ],
        weights=[0.4666666666666667, 0.3333333333333333, 0.2]
    )
    return algorithm


def build_model_pipeline(numerical_cols, categorical_cols):
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(
        steps=[
            ('featurization', generate_data_transformation_config(numerical_cols, categorical_cols)),
            ('ensemble', generate_algorithm_config()),
        ]
    )
    return pipeline


def train_model(X_train, y_train, numerical_cols, categorical_cols):
    model_pipeline = build_model_pipeline(numerical_cols, categorical_cols)
    model_pipeline.fit(X_train, y_train)
    return model_pipeline


# Train the model
model = train_model(X_train, y_train, numerical_cols, categorical_cols)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# SHAP explanations
# Get transformed data
X_train_featurized = model.named_steps['featurization'].transform(X_train)
X_test_featurized = model.named_steps['featurization'].transform(X_test)

# Extract base models
ensemble = model.named_steps['ensemble']
estimators = ensemble.estimators_


# Prepare feature names after transformation
def get_feature_names(column_transformer):
    feature_names = []
    for name, transformer, features in column_transformer.transformers_:
        if transformer == 'drop':
            continue
        if hasattr(transformer, 'get_feature_names_out'):
            names = transformer.get_feature_names_out(features)
        else:
            names = features
        feature_names.extend(names)
    return feature_names


feature_names = get_feature_names(model.named_steps['featurization'])

# Generate SHAP values for each base model
for idx, (name, pipeline) in enumerate(ensemble.named_estimators_.items()):
    print(f"\nGenerating SHAP values for {name}")
    preproc = pipeline.named_steps['preproc']
    model_step = pipeline.named_steps['model']

    # Preprocess the data
    X_test_preprocessed = preproc.transform(X_test_featurized)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model_step)
    shap_values = explainer.shap_values(X_test_preprocessed)

    # SHAP summary plot
    shap.summary_plot(shap_values, X_test_preprocessed,
                      feature_names=feature_names,
                      #plot_type="bar"
                      )
    # plt.title(f"SHAP Summary for {name}")
    # plt.show()

# SHAP for the ensemble model
print("\nGenerating SHAP values for the ensemble model")

# Since SHAP doesn't directly support VotingRegressor, we can approximate the SHAP values
# by averaging the SHAP values from the individual models, weighted by their importance

# # Initialize shap_values_ensemble as zero array
# shap_values_ensemble = np.zeros_like(X_test_featurized)
#
# for idx, (name, pipeline) in enumerate(ensemble.named_estimators_.items()):
#     weight = ensemble.weights[idx]
#     preproc = pipeline.named_steps['preproc']
#     model_step = pipeline.named_steps['model']
#     X_test_preprocessed = preproc.transform(X_test_featurized)
#
#     explainer = shap.TreeExplainer(model_step)
#     shap_values = explainer.shap_values(X_test_preprocessed)
#
#     shap_values_ensemble += weight * shap_values
#
# # SHAP summary plot for ensemble
# shap.summary_plot(shap_values_ensemble, X_test_featurized, feature_names=feature_names, plot_type="bar")
# plt.title("SHAP Summary for Ensemble Model")
# plt.show()
