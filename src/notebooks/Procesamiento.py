# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%
# LEER CSV
weather_data  = pd.read_csv('../data/CLIMA/GT_CLIMA.csv', parse_dates=['Fecha'])
index_data = pd.read_csv('../data/INDICES/GT_INDICES_NAX.csv', parse_dates=['FECHA'], low_memory=False)
harvest_data    = pd.read_csv('../data/BASE_ENTRENAMIENTO_VALIDACION_GT_NIv2.csv', parse_dates=['fecha', 'fi', 'ff'])

# %%
print(harvest_data.iloc[4])

# %%
# Variables numéricas
numeric_vars = ['area', 'numero_corte', 'edad_proyectada', 'rendimiento','TCH']
numeric_vars_indice = ['NDVI_POND','LAI_POND','PRODUCCION_PON','ANOMALIAS_POND','HUMEDAD_POND']
numeric_vars_clima = ['ETP','Radiacion (MJ/m2)','Amplitud Térmica',
'R0','temperatura','humedad relativa',  'humedad relativa minima', 'humedad relativa maxima',
'precipitacion', 'velocidad viento','velocidad viento minima',
'velocidad viento maxima', 'direccion viento']

# %% [markdown]
# # NA TREATMENT

# %%
weather_delet = ['radiacion',
'radiacion promedio',
'temperatura minima',
'temperatura maxima',
'radiacion promedio',
'presion atmosferica maxima',
'presion atmosferica minima',  
'presion atmosferica',
'mojadura',
'Unnamed: 31']
   
indice_delet = ['AGUA_POND','NITROGENADO_POND','ESTATUS_COSECHA','MADURACION_POND']

harvest_delet = ['edad_act','fecha_mad','semana_corte']

weather_data = weather_data.drop(columns=weather_delet)
index_data = index_data.drop(columns=indice_delet)
harvest_data = harvest_data.drop(columns=harvest_delet)


# %%
weather_cols_nulls = [
    'temperatura',
    'humedad relativa maxima',
    'humedad relativa minima',
    'humedad relativa',
    'velocidad viento',
    'velocidad viento maxima',
    'direccion viento',
    'velocidad viento minima'
]

index_cols_nulls = [
    'ANOMALIAS_POND',
    'LAI_POND',
    'HUMEDAD_POND',
    'PRODUCCION_PON',
    'NDVI_POND'
]



# %%
for col in weather_cols_nulls:
    weather_data[col] = weather_data[col].fillna(weather_data[col].mean())

for col in index_cols_nulls:
    index_data[col] = index_data[col].fillna(index_data[col].mean())


weather_data.columns

# %%
def preprocess_data(harvest_row, months=6):
    # Get the start date (fi) and calculate the prediction date (months before ff)
    start_date = harvest_row['fi']
    prediction_date = harvest_row['ff'] - pd.Timedelta(days=months * 30)

    # Filter weather data for the relevant period
    relevant_weather = weather_data[(weather_data['Fecha'] >= start_date) & 
                                    (weather_data['Fecha'] <= prediction_date) & 
                                    (weather_data['Cuadrante'] == harvest_row['cuadrante'])]

    # Calculate stats for each 2-month interval and for the full period
    weather_stats = {}
    intervals = [2, 4, 6, 8, 10]
    for i in intervals:
        if i <= months:
            interval_data = relevant_weather[relevant_weather['Fecha'] <= start_date + pd.Timedelta(days=i * 30)]
            for column in numeric_vars_clima:
                weather_stats[f'{column}_Q{i}_mean'] = interval_data[column].mean()
                weather_stats[f'{column}_Q{i}_sum'] = interval_data[column].sum()
                weather_stats[f'{column}_Q{i}_std'] = interval_data[column].std()

                # Sort the DataFrame by date
                interval_data = interval_data.sort_values(by='Fecha')

                # Perform the integration using trapezoid method assuming width is 1 day
                integral = np.trapz(interval_data[column])
                weather_stats[f'{column}_Q{i}_integral'] = integral

    # Process index data
    relevant_index = index_data[(index_data['FECHA'] >= start_date) & 
                                (index_data['FECHA'] <= prediction_date) & 
                                (index_data['ABS_IDCOMP'] == harvest_row['terreno'])]

    index_stats = {}
    for i in intervals:
        if i <= months:
            interval_index = relevant_index[relevant_index['FECHA'] <= start_date + pd.Timedelta(days=i * 30)]
            for column in numeric_vars_indice:
                index_stats[f'{column}_Q{i}_mean'] = interval_index[column].mean()
                index_stats[f'{column}_Q{i}_sum'] = interval_index[column].sum()
                index_stats[f'{column}_Q{i}_std'] = interval_index[column].std()

                integral = np.trapz(interval_index[column])
                index_stats[f'{column}_Q{i}_integral'] = integral

    # Combine all features
    features = {
        'rendimiento': harvest_row['rendimiento'],
        'fecha': harvest_row['fecha'],
        'TCH': harvest_row['TCH'],
        'ABS_IDCOMP': harvest_row['terreno'],
        'ZAFRA': harvest_row['zafra'],
        'area': harvest_row['area'],
        'prod_mad': harvest_row['prod_mad'],
        'estacion': harvest_row['estacion'],
        'variedad': harvest_row['variedad'],
        'numero_corte': harvest_row['numero_corte'],
        'sist_riego': harvest_row['sist_riego'],
        'tipo_cosecha': harvest_row['tipo_cosecha'],
        'region': harvest_row['region'],
        'estrato': harvest_row['estrato'],
        'cuadrante': harvest_row['cuadrante'],
        'PRODUCTO_ACTUAL': harvest_row['producto_actual'],
        **weather_stats,
        **index_stats
    }

    return features

# %%

# Process all harvest data and save datasets for different months
processed_data_all = {}
intervals = [2, 4, 6, 8, 10]
for months in intervals:
    processed_data = []
    for _, row in harvest_data.iterrows():
        features = preprocess_data(row, months=months)
        processed_data.append(features)

    # Create a DataFrame with processed features
    final_data = pd.DataFrame(processed_data)

    # Normalize data
    def normalize_final_data(data):
        sum_columns = [column for column in data.columns if '_sum' in column]
        integral_columns = [column for column in data.columns if '_integral' in column]
        columns_to_normalize = sum_columns + integral_columns

        # Initialize and fit the MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data[columns_to_normalize])

        # Create a new DataFrame with normalized values
        normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize, index=data.index)

        # Update data with normalized values
        data[columns_to_normalize] = normalized_df

        return data

    final_data = normalize_final_data(final_data)
    processed_data_all[months] = final_data

    # Save to CSV
    final_data.to_csv(f'../data/PROCESS/processed_tch_prediction_data_{months}Mesv7.csv', index=False)

# %%

# Process all harvest data
processed_data = []
for _, row in harvest_data.iterrows():
    features = preprocess_data(row)
    #features['rendimiento'] = harvest_row['rendimiento']  # Target variable
    processed_data.append(features)

# Create a DataFrame with processed features
final_data = pd.DataFrame(processed_data)
#features = preprocess_data(harvest_row, weather_data, index_data.iloc[4])

# %%
def normalize_final_data(data):
    sum_columns = [column for column in data.columns if '_sum' in column]
    integral_columns = [column for column in data.columns if '_integral' in column]
    columns_to_normalize = sum_columns + integral_columns
    
    # Initialize and fit the MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data[columns_to_normalize])
    
    # Create a new DataFrame with normalized values
    normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize, index=data.index)
    
    # Update data with normalized values
    data[columns_to_normalize] = normalized_df
    
    return data

final_data = normalize_final_data(final_data)
final_data

# %%
final_data.to_csv('../data/PROCESS/processed_tch_prediction_datav6.csv', index=False)
 

# %%
print(harvest_data.iloc[3])

# %%
print(harvest_data.iloc[4])

# %%
row = harvest_data.iloc[4]

start_date = row['fi']
prediction_date = row['ff'] - pd.Timedelta(days=180)
relevant_weather = weather_data[(weather_data['Fecha'] >= start_date) & 
                                (weather_data['Fecha'] <= prediction_date) & 
                                (weather_data['Cuadrante'] == row['cuadrante'])]

relevant_weather



# %%
a = preprocess_data(row)
print(a)

# %%
import matplotlib.pyplot as plt

# Asegúrate de que la columna 'fecha' esté en formato de fecha
final_data['fecha'] = pd.to_datetime(final_data['fecha'])
final_data= final_data.sort_values(by='fecha')
# Configurar la figura y las líneas
plt.figure(figsize=(10, 6))

# Graficar 'TCH' vs 'fecha'
plt.plot(final_data['fecha'], final_data['TCH'], label='TCH', color='b', marker='o')

# Graficar 'rendimiento' vs 'fecha' en la misma gráfica
plt.plot(final_data['fecha'], final_data['rendimiento'], label='Rendimiento', color='g', marker='x')

# Configurar etiquetas y título
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Serie de Tiempo: Fecha vs TCH y Rendimiento')

# Añadir leyenda
plt.legend()

# Rotar las etiquetas de fecha para una mejor visualización
plt.xticks(rotation=45)

# Mostrar la gráfica
plt.tight_layout()
plt.show()



