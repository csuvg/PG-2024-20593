import os
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
# Connect to Azure Cosmos DB and upload the data
from azure.cosmos import CosmosClient, exceptions
import json
import uuid
import API.scripts.config as config


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
columns = ['geometry', 'ABS_IDCOMP', 'ZAFRA','unidad_01','id'] + \
    [f'TCHPRED_{h}Meses' for h in horizons]

gdf = merged[columns].rename(columns={ 'ZAFRA': 'zafra'})

# Export to GeoJSON
gdf.to_file('outputv4.geojson', driver='GeoJSON')

data = json.loads(gdf.to_json())

HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']
DATABASE_ID = config.settings['database_id']
CONTAINER_ID = config.settings['container_id']


# Initialize the Cosmos client
client = CosmosClient(HOST, credential=MASTER_KEY)

# Get the database and container clients
database = client.get_database_client(DATABASE_ID)
container = database.get_container_client(CONTAINER_ID)

# Upload each feature in the GeoJSON as a document in the container
for feature in data['features']:
    try:
        doc = feature['properties']
        doc['geometry'] = feature['geometry']
        
        if 'id' not in doc:
            doc['id'] = str(uuid.uuid4())
        
        # Optionally, set the partition key if your container uses one
        # doc['partitionKey'] = your_partition_key_value
        
        # Create the item in Cosmos DB
        print("INDEX OF #")
        container.upsert_item(body=doc)
        
    except exceptions.CosmosResourceExistsError:
        print(f"Item already exists: {feature['properties']['id']}")