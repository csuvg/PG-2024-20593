import os
import base64
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import BlobBlock
from dotenv import load_dotenv

load_dotenv("../environment/.env")


# Azure Blob Storage configuration
CONNECT_STR = os.getenv('AZURE_STORAGE_CONNECTION_STRING')


# Set random seeds for reproducibility
seed_ = 935115
np.random.seed(seed_)

# Directories and model setup
models_dir = '../models'
data_dir = '../data/PROCESS'
MODEL_PREFIX = 'TCH'
TIME_HORIZONS = [2, 4, 6, 8, 10]

# Prepare results list
results_list = []

# Iterate through the time horizons
for horizon in TIME_HORIZONS:
    # Construct the model directory name
    model_dir = f'{MODEL_PREFIX}{horizon}Meses_1'
    model_path = os.path.join(models_dir, model_dir, 'model.pkl')
    data_path = os.path.join(data_dir, f'/processed_tch_prediction_data_{horizon}Mesv7.csv')
    data_path = f'{data_dir}/processed_tch_prediction_data_{horizon}Mesv7.csv'
    
    # Check if model and data file exist
    if not os.path.exists(model_path):
        raise Exception(f"Model file not found: {model_path}")
    if not os.path.exists(data_path):
        raise Exception(f"Data file not found: {data_path}")

    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load the data
    data = pd.read_csv(data_path)

    # Rename columns if necessary
    if 'prod_mad_Bispiribac ' in data.columns:
        data = data.rename(columns={'prod_mad_Bispiribac ': 'prod_mad_Bispiribac'})

    # Limitar a zafra 2020
    data = data[data['ZAFRA'] == '23-24']
    data['ZAFRA'] = '24-25'

    # Prepare data for prediction
    features = data.drop(['TCH', 'ABS_IDCOMP', 'ZAFRA','rendimiento','fecha'], axis=1)

        # Extract expected feature names from the model's data transformer step
    for step_name, step in model.steps:
        if step_name == 'datatransformer':
            data_transformer = step
            break

    expected_features_dict = data_transformer.get_column_names_and_types
    expected_features = list(expected_features_dict.keys())

    # Get the columns from your input features DataFrame
    input_features = features.columns.tolist()

    # Find columns that are in input but not expected by the model
    extra_columns = set(input_features) - set(expected_features)

    # Find columns that are expected by the model but not in input
    missing_columns = set(expected_features) - set(input_features)

    # Print the extra and missing columns
    print(f"Processing {model_dir} with data from {data_path}")
    print("Extra columns in input data not expected by the model:")
    print(extra_columns)
    print("\nMissing columns in input data expected by the model:")
    print(missing_columns)

    # Make predictions
    predicted_labels = model.predict(features)

    # Save results in the list
    results = pd.DataFrame({
        'ABS_IDCOMP': data['ABS_IDCOMP'].astype(str),
        'ZAFRA': data['ZAFRA'],
        f'TCHPRED_{horizon}Meses': predicted_labels
    })
    results_list.append(results)

# Merge all prediction results
final_results = results_list[0]
for i in range(1, len(results_list)):
    final_results = pd.merge(final_results, results_list[i], on=['ABS_IDCOMP', 'ZAFRA'], how='outer')


final_results.to_csv("../data/final_results.csv", index=False)

# Load the GeoDataFrame
world = gpd.read_file("../data/SHAPES/shapefile.shp")

# Ensure matching data types for merge
world['unidad_01'] = world['unidad_01'].astype(str)

# Merge with the DataFrame 'final_results'
merged = world.merge(final_results, left_on='unidad_01', right_on='ABS_IDCOMP', how='left')

# Create the final GeoDataFrame
columns = ['geometry', 'ABS_IDCOMP', 'ZAFRA'] + [f'TCHPRED_{h}Meses' for h in TIME_HORIZONS]
gdf = merged[columns].rename(columns={'ABS_IDCOMP': 'id', 'ZAFRA': 'zafra'})

# Export to GeoJSON
output_geojson_path = '../data/outputv6.geojson'
gdf.to_file(output_geojson_path, driver='GeoJSON')

# Plot Predicted vs Real for the 6-month model as an example
data_6m = pd.read_csv(os.path.join(data_dir, 'TEST_6Mesv7.csv'))
predicted_labels_6m = final_results['TCHPRED_6Meses'].dropna().values[:len(data_6m)]

plt.figure(figsize=(12, 6))
plt.scatter(data_6m['TCH'], predicted_labels_6m, alpha=0.5, color='blue', label='Predicciones')
plt.scatter(data_6m['TCH'], data_6m['TCH'], alpha=0.5, color='red', label='Valores reales')
plt.plot([data_6m['TCH'].min(), data_6m['TCH'].max()], [data_6m['TCH'].min(), data_6m['TCH'].max()], 'r', lw=2)
plt.xlabel('Real TCH')
plt.ylabel('Predicted TCH / Real TCH')
plt.title('Predicted vs Actual Values (6 Months)')
plt.legend()

# Save the plot
output_plot_path = '../data/predicted_vs_actual_6m.png'
plt.savefig(output_plot_path)
plt.show()

output_geojson_path, output_plot_path



# Name of the container and blob (file)
container_name = "tchgeopandas"  # Use lowercase letters, numbers, and hyphens
blob_name = "outputv6.geojson"

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
    with open(output_geojson_path, 'rb') as file:
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