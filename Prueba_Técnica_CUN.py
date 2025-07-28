#============================================
# Prueba Técnica CUN
#============================================
# Resolución de la prueba técnica para el cargo de Analista Operativo
#============================================
# Autores: Daniel Manosalva
# Fecha: 2025-07-26
#============================================
#===========Iniciamos cargando las librerías necesarias para el análisis de datos y la visualización de los mismos.===========
import pandas as pd
import requests
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine,text
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import openpyxl
import random
import sys

####################################################################################
#===========Realizamos una función para cargar los datasets de la prueba.===========
# 1.1 ================== SIMULACIÓN DE DATOS DE VENTAS ==================
fecha_inicio = datetime(2025, 1, 1)
fecha_fin = datetime(2025, 6, 30)
days = (fecha_fin - fecha_inicio).days + 1
dates = [fecha_inicio + timedelta(days=i) for i in range(days)]

#Lista de productos y regiones
productos = ["Impermeable", "Paraguas", "Botas", "Chaqueta", "Pantalón"]
region = ["Bogotá", "Medellín", "Cali", "Barranquilla"]

# Diccionario de product_id por producto
ids_producto = {
    "Impermeable": 101,
    "Paraguas": 102,
    "Botas": 103,
    "Chaqueta": 104,
    "Pantalón": 105
}

# Lista de categorías posibles
categorias = ["Hombre", "Mujer", "Niño", "Unisex"]

# Generación del conjunto de datos
data = []
id_counter = 1
for date in dates:
    for producto in productos:
        region = random.choice(region)
        precio = round(random.uniform(20, 80), 2)
        ventas = np.random.randint(5, 50)
        id_producto = ids_producto[producto]
        categoria = random.choice(categorias)
        data.append([id_counter, date.strftime("%Y-%m-%d"), producto, id_producto, categoria, region, ventas, precio])
        id_counter += 1

df_ventas = pd.DataFrame(data, columns=["id", "fecha", "producto", "id_producto", "categoria","region", "ventas", "precio"])

# Guardar CSV
df_ventas.to_csv("01.Ventas_historicas.csv", index=False, encoding="utf-8")
print("Archivo 'ventas_historicas.csv' generado con éxito.")
print("Datos de ventas simulados:")
print(df_ventas.head())


##########################################################################
# 1.2 ================== EXTRACCIÓN DE DATOS CLIMÁTICOS ==================
url = "https://archive-api.open-meteo.com/v1/archive"
parametros = {
    "latitude": 4.61,
    "longitude": -74.08,
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
    "timezone": "America/Bogota",
    "start_date": fecha_inicio.strftime("%Y-%m-%d"),
    "end_date": fecha_fin.strftime("%Y-%m-%d"),
}

response = requests.get(url, params=parametros)

if response.status_code == 200:
    clima_data = response.json().get("daily", {})
    df_clima = pd.DataFrame(clima_data)
    df_clima['time'] = pd.to_datetime(df_clima['time'])
    df_clima.rename(columns={'time': 'date'}, inplace=True)
    df_clima.to_csv("1.2.Clima_ciudad.csv", index=False)
    print("Archivo 'clima_ciudad.csv' generado con éxito.")   
else:
    # Manejo de errores en la solicitud
    print("Error en la solicitud:")
    print("Código:", response.status_code)
    print("Contenido:", response.text)
    raise Exception(f"Error al obtener datos del clima: {response.status_code}")
print(df_clima.head())
df_clima.info()

##################################################################
# 1.3 ================== TRANSFORMACIÓN Y UNIÓN ==================
df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
df_clima['date'] = pd.to_datetime(df_clima['date'])
# Unir los DataFrames de ventas y clima por fecha
df_merged = pd.merge(df_ventas, df_clima, left_on="fecha", right_on="date", how="inner")
# Eliminar columnas innecesarias
df_merged.drop(columns=["date"], inplace=True)
# Renombrar columnas para mayor claridad
df_merged.rename(columns={
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "precipitation_sum": "precipitation"
}, inplace=True)

# Verificar el DataFrame resultante
print("Datos combinados de ventas y clima:")
print(df_merged.head())

###################################################################################
# 1.4 ================== PREDICCIÓN DE VENTAS ==================
from sklearn.metrics import r2_score
#Asegurarse de no tener nulos
df_merged.dropna(subset=["temp_max", "temp_min", "precipitation", "ventas"], inplace=True)

# Crear columna vacía para predicción por producto
df_merged["sales_prediction_por_producto"] = np.nan

# Entrenar modelo por producto
productos = df_merged["producto"].unique()

for producto in productos:
    # Filtrar datos del producto
    df_producto = df_merged[df_merged["producto"] == producto]
    
    # Definir variables X e y
    X = df_producto[["temp_max", "temp_min", "precipitation"]]
    y = df_producto["ventas"]
    
    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Realizar predicción
    predicciones = modelo.predict(X).round(0).astype(int)
    
    # Asignar al DataFrame original
    df_merged.loc[df_producto.index, "sales_prediction_por_producto"] = predicciones
    
    # Mostrar R² para cada modelo
    r2 = r2_score(y, predicciones)
    print(f"R² para '{producto}': {r2:.4f}")
    
# Guardar el DataFrame combinado con predicciones
df_merged.to_csv("02.Ventas_clima_predicciones.csv", index=False, encoding="utf-8")
print("Archivo 'ventas_clima_predicciones.csv' generado con éxito.")
print("Datos combinados con predicciones:")
print(df_merged.head())
df_merged.info()
#######################################################################################
# 1.5 ================== CARGAR LOS DATOS TRASNFORMADOS A POSTGRESQL ==================
# Configuración de la conexión a PostgreSQL

# === Configuración de conexión a PostgreSQL ===
# 1.5.1. Función para limpiar texto
import re
import unicodedata
def limpiar_texto_avanzado(texto):
    if isinstance(texto, str):
        texto = unicodedata.normalize("NFKC", texto)
        texto = re.sub(r"[^\x00-\x7F\u00A0-\u00FF]+", "", texto)
    return texto

def limpiar_columnas_texto(df):
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].apply(limpiar_texto_avanzado)
    return df

# 1.5.2. Conexión PostgreSQL
usuario = "postgres"
contraseña = "010618Dani*"
host = "localhost"
puerto = "5432"
base_datos = "postgres"
engine = create_engine(f"postgresql+psycopg2://{usuario}:{contraseña}@{host}:{puerto}/{base_datos}")

# 1.5.3. Limpiar y cargar
df_merged = limpiar_columnas_texto(df_merged)

try:
    df_merged.to_sql("sales_predictions", engine, if_exists="replace", index=False)
    print("Datos cargados correctamente en la tabla 'sales_predictions'.")

    with engine.connect() as conn:
        resultado = conn.execute(text("SELECT * FROM sales_predictions LIMIT 5;"))
        for fila in resultado:
            print(fila)

except Exception as e:
    print("ERROR al cargar los datos en PostgreSQL:")
    print(str(e))
    
# 2 ================== CARGAR LOS DATOS A SUPABASE ==================
# 2.1 Conexión a Supabase
password = "010618Dani*"
url = f"postgresql://postgres.jogmxluulwzghopomzcc:{password}@aws-0-sa-east-1.pooler.supabase.com:5432/postgres"
engine_supabase = create_engine(url)

# 2.2 Crear tabla en Supabase si no existe
query_crear_tabla = """
CREATE TABLE IF NOT EXISTS sales_predictions (
    id SERIAL PRIMARY KEY,
    fecha DATE,
    id_producto INT,
    ventas INT,
    precio FLOAT,
    categoria VARCHAR(50),
    temp_min FLOAT,
    sales_prediction_por_producto FLOAT
);
"""
with engine_supabase.connect() as conn:
    conn.execute(text(query_crear_tabla))
    print("Tabla 'sales_predictions' verificada o creada correctamente en Supabase.")
# Guardar consulta de creación de tabla

# 2.3 Crear DataFrame para la tabla de creación
df_crear_tabla = pd.DataFrame({
    "query": [query_crear_tabla],
    "tabla": ["sales_predictions"]
})  
# Guardar DataFrame de creación de tabla
df_crear_tabla.to_csv("03.Query_crear_tabla_supabase.csv", index=False, encoding="utf-8")



# 2.4 Subir DataFrame principal a Supabase
df_merged.to_sql("sales_predictions", engine_supabase, if_exists="replace", index=False)
print("Tabla 'sales_predictions' cargada correctamente en Supabase.")

# 2.5 Consulta SQL para calcular el error promedio
query_error = """
SELECT 
    categoria,
    AVG(ABS(ventas - sales_prediction_por_producto)) AS error_promedio
FROM 
    sales_predictions
GROUP BY 
    categoria;
"""

# 2.5 Ejecutar consulta y guardar en Supabase
df_error = pd.read_sql_query(query_error, engine_supabase)
df_error.to_sql("error_promedio_categoria", engine_supabase, if_exists="replace", index=False)
print("Tabla 'error_promedio_categoria' también respaldada en Supabase.")

# ======Generar archivo CSV de los dataframe=====
# Asegurar que la primera fila sea el encabezado======
df_merged = df_merged.rename(columns=lambda x: x.strip())
df_error = df_error.rename(columns=lambda x: x.strip())
df_crear_tabla = df_crear_tabla.rename(columns=lambda x: x.strip())
df_crear_tabla.columns = [col.replace(" ", "_") for col in df_crear_tabla.columns]
# Guardar DataFrames como CSV
df_crear_tabla.to_csv("05.Query_crear_tabla_supabase.csv", index=False, encoding="utf-8")
df_merged.to_csv("04.Ventas_clima_predicciones_supabase.csv", index=False, encoding="utf-8")
df_error.to_csv("06.Error_promedio_categoria_supabase.csv", index=False, encoding="utf-8")
print("Archivos CSV generados para Supabase.")

