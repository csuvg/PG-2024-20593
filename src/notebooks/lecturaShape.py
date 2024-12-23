# %%
import geopandas
import pandas as pd

# Ajustar la configuración de pandas
pd.set_option('display.max_columns', None)

# Load the data
world = geopandas.read_file("../data/SHPAPEPRUEBA/shapefile.shp")


# %%
world.columns

# %%
world.describe()

# %%
world

# %%
world.tail()

# %%
# Agrupar por 'unidad_01' y agregar 'id' y 'geometry' como listas
grouped = world.groupby('unidad_01').agg({
    'id': list,
    'geometry': list
}).reset_index()

# Mostrar el DataFrame resultante
print(grouped)


# %%
df = pd.read_csv('../API/outputv2.csv')
df

# %%
world.explore()


