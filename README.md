# Sistema Predictivo del Tonelaje de Caña por Hectárea (TCH)

## Descripción del Proyecto

Este proyecto implementa un sistema avanzado de **predicción del tonelaje de caña por hectárea (TCH)** utilizando técnicas de aprendizaje automático. El sistema integra datos climatológicos, índices vegetativos derivados de imágenes satelitales y registros históricos de cosecha para **generar predicciones precisas** del rendimiento de los cultivos de caña de azúcar.

El proyecto se divide en dos componentes principales:

1. **Ambiente de entrenamiento** para desarrollo y evaluación de modelos.  
2. **API REST** para despliegue en producción.

---

## 1. Características Principales

- **Predicción del TCH** con diferentes horizontes temporales (2-10 meses).  
- **Integración de múltiples fuentes de datos**: clima, satélite e históricos de cosecha.  
- **Análisis de interpretabilidad** mediante SHAP.  
- **API REST** para consumo de predicciones en entornos productivos.  
- **Pipeline automatizado de MLOps** para la orquestación de entrenamiento y despliegue.

---

## 2. Instrucciones de Instalación

En esta sección se detallan los pasos necesarios para configurar y ejecutar el proyecto, tanto en el ambiente de entrenamiento (Python local) como en el ambiente de producción (Docker para la API).

### 2.1 Requisitos Previos

1. **Ambiente de Entrenamiento (Python local)**  
   - Python **3.11** o superior.  
   - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) instalado (en caso de que se utilicen GPUs para acelerar el entrenamiento).  
   - [Git](https://git-scm.com/).  
   - [pip](https://pip.pypa.io/en/stable/) (administrador de paquetes de Python).

2. **Ambiente de Producción (Docker)**  
   - [Docker](https://www.docker.com/) instalado y funcionando.  
   - Docker Compose (generalmente incluido en las versiones más recientes de Docker Desktop).

---

### 2.2 Instalación y Ejecución en el Ambiente de Entrenamiento

1. **Clona el repositorio**  
   ```bash
   https://github.com/csuvg/PG-2024-20593.git
   ```

2. Instala las dependencias
Dentro de la carpeta principal del proyecto, ejecuta:
   ```bash
   cd PG-2024-20593/src/
   pip install -r requirements.txt
   ```
3. Uso de Notebooks
Una vez instaladas las dependencias, podrás acceder a los notebooks localizados en la carpeta `/src/notebooks`. Estos notebooks contienen:

   - Procesos de exploración de datos.
   - Entrenamiento y evaluación de modelos.
   - Ejecución de análisis de interpretabilidad con SHAP.

### 2.3 Instalación y Ejecución de la API en Producción
1. **Construye la imagen Docker**  
Dentro de la carpeta principal del proyecto, ejecuta:
   ```bash
   cd PG-2024-20593/src/tchapi
   docker build -t tchapi .
   ```
   Este comando creará una imagen de Docker con la etiqueta tchapi, lista para su despliegue.

2. **Arranca la API con Docker Compose**  
Asegúrate de tener un archivo `docker-compose.yml` configurado correctamente (debe incluir el servicio que utilice la imagen `tchapi`). Luego, ejecuta:
   ```bash
   docker compose up -d
   ```
   Esto levantará el contenedor en background (modo daemon). La API estará disponible en la ruta y puerto especificados en el docker-compose.yml (por defecto, podría ser http://localhost:5000).


### 3. Demo
En la carpeta `/demo/` encontrarás un video demostrativo que muestra el funcionamiento del proyecto. El video se almacena directamente en el repositorio para asegurar su disponibilidad permanente. Puedes acceder a él directamente [aquí](/demo/Demo.mp4) o utilizando la siguiente ruta relativa:

```
/demo/Demo.mp4
```


### 4. Informe
El informe final de este proyecto de graduación se encuentra en formato PDF y está almacenado en la carpeta` /docs/`. Puedes acceder a él directamente [aquí](docs/Informe.pdf) o utilizando la siguiente ruta relativa:

```
/docs/Informe.pdf
```

## 5. Contacto
Para más información o consultas, por favor contacta a:
- **Nombre:** Juan Angel Carrera Soto
- **Email:** car20593@uvg.edu.gt
- **GitHub:** https://github.com/jack200133