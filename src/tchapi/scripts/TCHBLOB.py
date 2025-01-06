from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import BlobBlock
import base64
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
from dotenv import load_dotenv

load_dotenv("../environment/.env")


# Azure Blob Storage configuration
CONNECT_STR = os.getenv('AZURE_STORAGE_CONNECTION_STRING')



# Set random seeds for reproducibility
seed_ = 935115
np.random.seed(seed_)
MODEL = 'AZUREML_'

# Load the data
data = pd.read_csv('./data/PROCESS/encoded_tch_prediction_data_zafrav3.2.csv')
data = pd.read_csv('./data/PROCESS/TEST.csv')

#rename prod_mad_Bispiribac 
data = data.rename(columns={'prod_mad_Bispiribac ': 'prod_mad_Bispiribac'})

# Limitar a zafra 2020
data = data[data['ZAFRA'] == '23-24']

data['ZAFRA'] = '24-25'

# Prepare data for prediction
features = data.drop(['TCH', 'ABS_IDCOMP', 'ZAFRA'], axis=1)
print(features.columns)

# Define the directory where models are saved
models_dir = './models'

# List all subdirectories in models_dir that start with 'CNN_'
existing_dirs = [
    d for d in os.listdir(models_dir)
    if os.path.isdir(os.path.join(models_dir, d)) and d.startswith(MODEL)
]

# Extract the numbers from existing CNN_# directories
numbers = []
for d in existing_dirs:
    try:
        num = int(d.split('_')[1])
        numbers.append(num)
    except ValueError:
        pass  # Ignore directories that don't fit the pattern

# If there are no existing directories, raise an error
if not numbers:
    raise Exception("No model directories found in 'models'.")

# Determine the latest model number
latest_num = max(numbers)

# Construct the path to the latest model directory
latest_model_dir = os.path.join(models_dir, f'{MODEL}{latest_num}')
print(f"Loading models from directory: {latest_model_dir}")
# Load the Azure ML model from model.pkl
model_path = os.path.join(latest_model_dir, 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# If the model is a pipeline that includes preprocessing, you can directly use it
# Otherwise, you may need to apply preprocessing steps (e.g., scaling) here
# Since we don't have scaler.pkl, we assume the model includes preprocessing

for step_name, step in model.steps:
    print(step_name)
    if step_name == 'datatransformer':
        data_transformer = step
        break
expected_features = data_transformer.get_column_names_and_types

# Step 1: Extract expected feature names from the model
expected_features_dict = data_transformer.get_column_names_and_types
expected_features = list(expected_features_dict.keys())

# Step 2: Get the columns from your input features DataFrame
input_features = features.columns.tolist()

# Step 3: Find columns that are in input but not expected by the model
extra_columns = set(input_features) - set(expected_features)

# Step 4: Find columns that are expected by the model but not in input
missing_columns = set(expected_features) - set(input_features)

# Step 5: Print the extra and missing columns
print("Extra columns in input data not expected by the model:")
print(extra_columns)

print("\nMissing columns in input data expected by the model:")
print(missing_columns)


# Make predictions
predicted_labels = model.predict(features)


# Create a DataFrame with 'ABS_IDCOMP', 'ZAFRA', 'TCHPRED'
min_pred = predicted_labels.min()
max_pred = predicted_labels.max()

# Generate random predictions between min and max of the predicted labels
random_pred_4meses = np.random.uniform(low=min_pred, high=max_pred, size=len(data))
random_pred_2meses = np.random.uniform(low=min_pred, high=max_pred, size=len(data))

# Create a DataFrame with 'ABS_IDCOMP', 'ZAFRA', 'TCHPRED'
results = pd.DataFrame({
    'ABS_IDCOMP': data['ABS_IDCOMP'].astype(str),
    'ZAFRA': data['ZAFRA'],
    'TCHPRED_8Meses': np.nan,
    'TCHPRED_6Meses': predicted_labels,
    'TCHPRED_4Meses': random_pred_4meses,
    'TCHPRED_2Meses': random_pred_2meses,
})
# Load the GeoDataFrame
world = gpd.read_file("./data/SHAPES/shapefile.shp")

# Ensure matching data types for merge
world['unidad_01'] = world['unidad_01'].astype(str)

# Merge with the DataFrame 'results'
merged = world.merge(results, left_on='unidad_01',
                     right_on='ABS_IDCOMP', how='left')

# Create the final GeoDataFrame
horizons = [2, 4, 6, 8]
columns = ['geometry', 'ABS_IDCOMP', 'ZAFRA'] + \
    [f'TCHPRED_{h}Meses' for h in horizons]

gdf = merged[columns].rename(columns={'ABS_IDCOMP': 'id', 'ZAFRA': 'zafra'})

# Export to GeoJSON
gdf.to_file('outputv5.geojson', driver='GeoJSON')



# Name of the container and blob (file)
container_name = "tchgeopandas"  # Use lowercase letters, numbers, and hyphens
blob_name = "outputv5.geojson"
local_file_path = "outputv5.geojson"

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)

# Create a container if it doesn't exist
container_client = blob_service_client.get_container_client(container_name)
try:
    container_client.create_container()
except Exception as e:
    print(f"Container already exists. {e}")

# Create a blob client
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)


block_list = []
block_size = 4 * 1024 * 1024  # 4 MB

try:
    with open(local_file_path, 'rb') as file:
        idx = 0
        while True:
            data = file.read(block_size)
            if not data:
                break
            idx += 1
            block_id = base64.b64encode(str(idx).zfill(6).encode()).decode()
            block_list.append(BlobBlock(block_id=block_id))
            blob_client.stage_block(block_id=block_id, data=data)
            print(f"Staged block {idx}")

    # Commit the blocks
    blob_client.commit_block_list(block_list)
    print(f"Uploaded {blob_name} to container {container_name}.")
except Exception as e:
    print(f"An error occurred during upload: {e}")