import os
import logging
from datetime import datetime as dt
from flask import Blueprint, jsonify
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import json
import pandas as pd
import requests  # Para poder solicitar contenido desde GitHub

load_dotenv("./environment/.env")

# Blueprint
geojson_api = Blueprint('geojson_api', __name__)

# Azure Blob Storage configure
LOG_ROUTE = './logs/activity/'
CONNECT_STR = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = "tchgeopandas"  # Replace with your container name
BLOB_NAME = "outputv6.geojson"   # Replace with your blob name
RAW_URL = "https://raw.githubusercontent.com/Jack200133/tchapi/master/data/outputv6.geojson"


if not os.path.exists(LOG_ROUTE):
    os.makedirs(LOG_ROUTE)

log_filename = os.path.join(
    LOG_ROUTE,
    "application_" + dt.now().strftime('%d_%m_%Y') + ".log"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@geojson_api.route('/get_geojson', methods=['GET'])
def get_geojson():
    logger.info("Received request to get GeoJSON file.")
    try:
        # Verificar si existe la cadena de conexión
        if not CONNECT_STR:
            # Si no existe, obtenemos el archivo desde GitHub
            logger.info("Obteniendo archivo desde GitHub...")
            response = requests.get(RAW_URL)
            if response.status_code == 200:
                geojson_content = json.loads(response.text)
                logger.info("GeoJSON file obtenido desde GitHub.")
                return jsonify(geojson_content)
            else:
                logger.error(f"Error en GitHub.{response.status_code}")
                return jsonify(
                    {'error':
                     'No se pudo obtener el archivo desde GitHub'}), 500

        # Si CONNECT_STR existe, usamos Azure
        logger.info("Descargando desde Azure Blob Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(
            CONNECT_STR)
        logger.info("Azure BlobServiceClient created successfully.")

        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME,
            blob=BLOB_NAME
        )
        logger.info(
            f"BlobClient for container '{CONTAINER_NAME}' "
            f"and blob '{BLOB_NAME}' obtained successfully."
        )

        # Descargar el blob a memoria
        downloader = blob_client.download_blob()
        geojson_bytes = downloader.readall()
        logger.info("GeoJSON file downloaded successfully from Azure")

        # Parsear el contenido a un objeto de Python
        geojson_content = json.loads(geojson_bytes)
        logger.info("GeoJSON content parsed successfully.")

        return jsonify(geojson_content)

    except Exception as e:
        logger.error(f"Error occurred while fetching GeoJSON file: {str(e)}")
        return jsonify({'error': str(e)}), 500


@geojson_api.route('/get_tch_mean', methods=['GET'])
def get_tch_mean():
    logger.info("Received request to get TCH mean values.")
    try:
        # Load the final results CSV
        final_results_path = "./data/final_results.csv"
        if not os.path.exists(final_results_path):
            raise FileNotFoundError(f"File not found: {final_results_path}")

        final_results = pd.read_csv(final_results_path)
        logger.info("Final results CSV loaded successfully.")

        # Group by 'ZAFRA' and calculate the mean for each prediction horizon
        grouped_results = final_results.groupby('ZAFRA').mean().reset_index()

        # Convert the results to a dictionary
        tch_mean_values = grouped_results.to_dict(orient='records')
        logger.info("TCH mean values calculated successfully.")

        # Return the TCH mean values as a JSON response
        return jsonify(tch_mean_values)

    except Exception as e:
        logger.error(
            f"Error occurred while fetching TCH mean values: {str(e)}")
        return jsonify({'error': str(e)}), 500
