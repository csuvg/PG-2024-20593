# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

data_sample = PandasParameterType(pd.DataFrame({"area": pd.Series([0.0], dtype="float64"), "prod_mad": pd.Series(["example_value"], dtype="object"), "estacion": pd.Series(["example_value"], dtype="object"), "variedad": pd.Series(["example_value"], dtype="object"), "numero_corte": pd.Series([0], dtype="int64"), "sist_riego": pd.Series(["example_value"], dtype="object"), "tipo_cosecha": pd.Series(["example_value"], dtype="object"), "region": pd.Series(["example_value"], dtype="object"), "estrato": pd.Series(["example_value"], dtype="object"), "cuadrante": pd.Series(["example_value"], dtype="object"), "PRODUCTO_ACTUAL": pd.Series(["example_value"], dtype="object"), "ETP_Q2_mean": pd.Series([0.0], dtype="float64"), "ETP_Q2_sum": pd.Series([0.0], dtype="float64"), "ETP_Q2_std": pd.Series([0.0], dtype="float64"), "ETP_Q2_integral": pd.Series([0.0], dtype="float64"), "Radiacion (MJ/m2)_Q2_mean": pd.Series([0.0], dtype="float64"), "Radiacion (MJ/m2)_Q2_sum": pd.Series([0.0], dtype="float64"), "Radiacion (MJ/m2)_Q2_std": pd.Series([0.0], dtype="float64"), "Radiacion (MJ/m2)_Q2_integral": pd.Series([0.0], dtype="float64"), "Amplitud T\u00e9rmica_Q2_mean": pd.Series([0.0], dtype="float64"), "Amplitud T\u00e9rmica_Q2_sum": pd.Series([0.0], dtype="float64"), "Amplitud T\u00e9rmica_Q2_std": pd.Series([0.0], dtype="float64"), "Amplitud T\u00e9rmica_Q2_integral": pd.Series([0.0], dtype="float64"), "R0_Q2_mean": pd.Series([0.0], dtype="float64"), "R0_Q2_sum": pd.Series([0.0], dtype="float64"), "R0_Q2_std": pd.Series([0.0], dtype="float64"), "R0_Q2_integral": pd.Series([0.0], dtype="float64"), "temperatura_Q2_mean": pd.Series([0.0], dtype="float64"), "temperatura_Q2_sum": pd.Series([0.0], dtype="float64"), "temperatura_Q2_std": pd.Series([0.0], dtype="float64"), "temperatura_Q2_integral": pd.Series([0.0], dtype="float64"), "humedad relativa_Q2_mean": pd.Series([0.0], dtype="float64"), "humedad relativa_Q2_sum": pd.Series([0.0], dtype="float64"), "humedad relativa_Q2_std": pd.Series([0.0], dtype="float64"), "humedad relativa_Q2_integral": pd.Series([0.0], dtype="float64"), "humedad relativa minima_Q2_mean": pd.Series([0.0], dtype="float64"), "humedad relativa minima_Q2_sum": pd.Series([0.0], dtype="float64"), "humedad relativa minima_Q2_std": pd.Series([0.0], dtype="float64"), "humedad relativa minima_Q2_integral": pd.Series([0.0], dtype="float64"), "humedad relativa maxima_Q2_mean": pd.Series([0.0], dtype="float64"), "humedad relativa maxima_Q2_sum": pd.Series([0.0], dtype="float64"), "humedad relativa maxima_Q2_std": pd.Series([0.0], dtype="float64"), "humedad relativa maxima_Q2_integral": pd.Series([0.0], dtype="float64"), "precipitacion_Q2_mean": pd.Series([0.0], dtype="float64"), "precipitacion_Q2_sum": pd.Series([0.0], dtype="float64"), "precipitacion_Q2_std": pd.Series([0.0], dtype="float64"), "precipitacion_Q2_integral": pd.Series([0.0], dtype="float64"), "velocidad viento_Q2_mean": pd.Series([0.0], dtype="float64"), "velocidad viento_Q2_sum": pd.Series([0.0], dtype="float64"), "velocidad viento_Q2_std": pd.Series([0.0], dtype="float64"), "velocidad viento_Q2_integral": pd.Series([0.0], dtype="float64"), "velocidad viento minima_Q2_mean": pd.Series([0.0], dtype="float64"), "velocidad viento minima_Q2_sum": pd.Series([0.0], dtype="float64"), "velocidad viento minima_Q2_std": pd.Series([0.0], dtype="float64"), "velocidad viento minima_Q2_integral": pd.Series([0.0], dtype="float64"), "velocidad viento maxima_Q2_mean": pd.Series([0.0], dtype="float64"), "velocidad viento maxima_Q2_sum": pd.Series([0.0], dtype="float64"), "velocidad viento maxima_Q2_std": pd.Series([0.0], dtype="float64"), "velocidad viento maxima_Q2_integral": pd.Series([0.0], dtype="float64"), "direccion viento_Q2_mean": pd.Series([0.0], dtype="float64"), "direccion viento_Q2_sum": pd.Series([0.0], dtype="float64"), "direccion viento_Q2_std": pd.Series([0.0], dtype="float64"), "direccion viento_Q2_integral": pd.Series([0.0], dtype="float64"), "NDVI_POND_Q2_mean": pd.Series([0.0], dtype="float64"), "NDVI_POND_Q2_sum": pd.Series([0.0], dtype="float64"), "NDVI_POND_Q2_std": pd.Series([0.0], dtype="float64"), "NDVI_POND_Q2_integral": pd.Series([0.0], dtype="float64"), "LAI_POND_Q2_mean": pd.Series([0.0], dtype="float64"), "LAI_POND_Q2_sum": pd.Series([0.0], dtype="float64"), "LAI_POND_Q2_std": pd.Series([0.0], dtype="float64"), "LAI_POND_Q2_integral": pd.Series([0.0], dtype="float64"), "PRODUCCION_PON_Q2_mean": pd.Series([0.0], dtype="float64"), "PRODUCCION_PON_Q2_sum": pd.Series([0.0], dtype="float64"), "PRODUCCION_PON_Q2_std": pd.Series([0.0], dtype="float64"), "PRODUCCION_PON_Q2_integral": pd.Series([0.0], dtype="float64"), "ANOMALIAS_POND_Q2_mean": pd.Series([0.0], dtype="float64"), "ANOMALIAS_POND_Q2_sum": pd.Series([0.0], dtype="float64"), "ANOMALIAS_POND_Q2_std": pd.Series([0.0], dtype="float64"), "ANOMALIAS_POND_Q2_integral": pd.Series([0.0], dtype="float64"), "HUMEDAD_POND_Q2_mean": pd.Series([0.0], dtype="float64"), "HUMEDAD_POND_Q2_sum": pd.Series([0.0], dtype="float64"), "HUMEDAD_POND_Q2_std": pd.Series([0.0], dtype="float64"), "HUMEDAD_POND_Q2_integral": pd.Series([0.0], dtype="float64")}))
input_sample = StandardPythonParameterType({'data': data_sample})

result_sample = NumpyParameterType(np.array([0.0]))
output_sample = StandardPythonParameterType({'Results':result_sample})
sample_global_parameters = StandardPythonParameterType(1.0)

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script_v2')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('Inputs', input_sample)
@input_schema('GlobalParameters', sample_global_parameters, convert_to_provided_type=False)
@output_schema(output_sample)
def run(Inputs, GlobalParameters=1.0):
    data = Inputs['data']
    result = model.predict(data)
    return {'Results':result.tolist()}
