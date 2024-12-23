# Import necessary libraries
from sklearn.pipeline import FeatureUnion
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('../data/PROCESS/processed_tch_prediction_data_8Mesv7.csv')
# data = pd.read_csv('../data/UP/TEST_2Mesv7.csv')

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

# Function to remove NaN rows


def remove_nan_rows_in_X_y(X, y, sample_weights=None, is_timeseries=False, target_column='TCH'):
    """Removes rows with NaN in X or y."""
    X = X.copy()
    y = y.copy()
    mask = pd.isnull(y)
    if mask.any():
        X = X[~mask]
        y = y[~mask]
        if sample_weights is not None:
            sample_weights = sample_weights[~mask]
    mask = X.isnull().any(axis=1)
    if mask.any():
        X = X[~mask]
        y = y[~mask]
        if sample_weights is not None:
            sample_weights = sample_weights[~mask]
    return X, y, sample_weights


# Prepare the data
X_train, y_train, sample_weights_train = remove_nan_rows_in_X_y(
    X_train, y_train)
X_test, y_test, sample_weights_test = remove_nan_rows_in_X_y(X_test, y_test)

# Define column groups (as per your original code)
column_group_3 = ['tipo_cosecha']
column_group_2 = ['prod_mad', 'estacion', 'variedad', 'numero_corte',
                  'sist_riego', 'region', 'estrato', 'cuadrante', 'PRODUCTO_ACTUAL']

# Assuming all other columns are numerical and belong to column_group_0 or column_group_1
all_columns = X_train.columns.tolist()
used_columns = column_group_2 + column_group_3
remaining_columns = [col for col in all_columns if col not in used_columns]

# For simplicity, we'll divide the remaining columns evenly between group 0 and group 1
half = len(remaining_columns) // 2
column_group_0 = remaining_columns[:half]
column_group_1 = remaining_columns[half:]

# Define mappers with corrected column wrapping


def get_mapper_0(column_names):
    return DataFrameMapper(
        [([col], [SimpleImputer(strategy='mean')]) for col in column_names],
        input_df=True, df_out=True)


def get_mapper_1(column_names):
    return DataFrameMapper(
        [([col], [SimpleImputer(strategy='mean', add_indicator=True)])
         for col in column_names],
        input_df=True, df_out=True)


def get_mapper_2(column_names):
    return DataFrameMapper(
        [([col], [
            SimpleImputer(strategy='most_frequent'),
            OrdinalEncoder(handle_unknown='use_encoded_value',
                           unknown_value=-1)
        ]) for col in column_names],
        input_df=True, df_out=True)


def get_mapper_3(column_names):
    return DataFrameMapper(
        [([col], [
            SimpleImputer(strategy='most_frequent'),
            OrdinalEncoder(handle_unknown='use_encoded_value',
                           unknown_value=-1)
        ]) for col in column_names],
        input_df=True, df_out=True)


# Define feature union

feature_union = FeatureUnion([
    ('mapper_0', get_mapper_0(column_group_0)),
    ('mapper_1', get_mapper_1(column_group_1)),
    ('mapper_2', get_mapper_2(column_group_2)),
    ('mapper_3', get_mapper_3(column_group_3)),
])

# Define models and pipelines
# Pipeline 0
preproc_0 = StandardScaler(with_mean=False, with_std=False)
model_0 = XGBRegressor(
    colsample_bytree=0.6,
    learning_rate=0.4,
    gamma=0.1,
    max_depth=10,
    n_estimators=50,
    reg_lambda=0.625,
    subsample=0.8,
    verbosity=0,
    random_state=0
)
pipeline_0 = Pipeline([
    ('preproc', preproc_0),
    ('model', model_0)
])

# Pipeline 1
preproc_1 = StandardScaler(with_mean=False, with_std=False)
model_1 = XGBRegressor(
    colsample_bytree=0.9,
    learning_rate=0.5,
    gamma=0.01,
    max_depth=8,
    n_estimators=50,
    reg_alpha=1.3541666666666667,
    reg_lambda=1.6666666666666667,
    verbosity=0,
    random_state=0
)
pipeline_1 = Pipeline([
    ('preproc', preproc_1),
    ('model', model_1)
])

# Pipeline 2
preproc_2 = StandardScaler(with_mean=False, with_std=True)
model_2 = LGBMRegressor(
    colsample_bytree=0.6,
    learning_rate=0.16842263157894738,
    max_bin=1023,
    max_depth=9,
    min_child_samples=38,
    min_split_gain=0.7368421052631579,
    n_estimators=100,
    num_leaves=255,
    reg_lambda=0.75,
    subsample=0.9,
    subsample_freq=6,
    random_state=0
)
pipeline_2 = Pipeline([
    ('preproc', preproc_2),
    ('model', model_2)
])

# Pipeline 3
preproc_3 = MaxAbsScaler()
model_3 = XGBRegressor(
    max_depth=6,
    n_estimators=100,
    verbosity=0,
    random_state=0
)
pipeline_3 = Pipeline([
    ('preproc', preproc_3),
    ('model', model_3)
])

# Pipeline 4
preproc_4 = StandardScaler(with_mean=False, with_std=False)
model_4 = XGBRegressor(
    colsample_bytree=0.5,
    max_depth=9,
    n_estimators=50,
    reg_alpha=0.3125,
    reg_lambda=1.5625,
    subsample=0.5,
    verbosity=0,
    random_state=0
)
pipeline_4 = Pipeline([
    ('preproc', preproc_4),
    ('model', model_4)
])

# Assemble the ensemble
estimators = [
    ('model_0', pipeline_0),
    ('model_1', pipeline_1),
    ('model_2', pipeline_2),
    ('model_3', pipeline_3),
    ('model_4', pipeline_4)
]

weights = [0.4, 0.3, 0.1, 0.1, 0.1]

ensemble = VotingRegressor(estimators=estimators, weights=weights)

# Build the full pipeline
full_pipeline = Pipeline([
    ('featurization', feature_union),
    ('ensemble', ensemble)
])

# Fit the model
print("\nTraining the model...")
full_pipeline.fit(X_train, y_train)

# Transform the test data
print("\nTransforming the test data...")
featurization = full_pipeline.named_steps['featurization']
X_test_featurized = featurization.transform(X_test)

# Get feature names for SHAP plots


def get_feature_names(feature_union):
    feature_names = []
    for name, transformer in feature_union.transformer_list:
        if isinstance(transformer, DataFrameMapper):
            for feature in transformer.features:
                col_name = feature[0]
                if isinstance(col_name, list):
                    feature_names.extend(col_name)
                else:
                    feature_names.append(col_name)
        else:
            print(f"Transformer {name} does not provide feature names")
    return feature_names


feature_names = get_feature_names(featurization)

# Generate SHAP values for each pipeline in the ensemble
# Generate SHAP values for each pipeline in the ensemble
print("\nGenerating SHAP values for each model in the ensemble...")

for idx, (name, pipeline) in enumerate(ensemble.named_estimators_.items()):
    print(f"\nGenerating SHAP values for {name}")
    preproc = pipeline.named_steps['preproc']
    model_step = pipeline.named_steps['model']

    # Preprocess the data
    X_test_preprocessed = preproc.transform(X_test_featurized)

    # Create SHAP explainer
    explainer = shap.Explainer(model_step)
    shap_values = explainer(X_test_preprocessed)

    # Save the SHAP values
    shap_file_name = f"model_{idx}_shap_10m.npy"
    np.save(shap_file_name, shap_values.values)
    print(f"SHAP values saved to {shap_file_name}")

    # Optionally, save the expected values
    expected_value_file_name = f"model_{idx}_expected_value_10m.npy"
    np.save(expected_value_file_name, shap_values.base_values)
    print(f"Expected values saved to {expected_value_file_name}")

    # SHAP summary plot
    shap.summary_plot(shap_values, X_test_preprocessed,
                      feature_names=feature_names,
                      show=False)
    plt.title(f"SHAP Summary Plot for {name}")
    plt.show()


# After fitting the full pipeline
print("\nExtracting encoding mappings...")

featurization = full_pipeline.named_steps['featurization']

for name, transformer in featurization.transformer_list:
    if name in ['mapper_2', 'mapper_3']:
        mapper = transformer
        for feature in mapper.features:
            # Assuming feature[0] is a list with one column name
            col_name = feature[0][0]
            steps = feature[1]        # List of transformers
            # Find the OrdinalEncoder in the steps
            for step in steps:
                if isinstance(step, OrdinalEncoder):
                    encoder = step
                    categories = encoder.categories_[
                        0]  # Categories for this feature
                    # Map categories to codes
                    mapping = dict(zip(categories, range(len(categories))))
                    # Save to a file
                    mapping_df = pd.DataFrame(
                        list(mapping.items()), columns=[col_name, 'code'])
                    mapping_file_name = f"encoding_mapping_{col_name}.csv"
                    mapping_df.to_csv(mapping_file_name, index=False)
                    print(
                        f"Saved encoding mapping for {col_name} to {mapping_file_name}")
