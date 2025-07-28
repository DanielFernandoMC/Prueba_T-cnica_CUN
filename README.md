# Prueba Técnica CUN - Analista Operativo
Fecha: 26 de julio de 2025
Elaborado por **Daniel Manosalva**

### Resumen
Este proyecto demuestra la capacidad para analizar patrones de ventas a partir de factores climáticos, generar predicciones mediante modelos estadísticos, almacenar los resultados en la nube y visualizarlos a través de un tablero interactivo. Está diseñado como una prueba técnica orientada a la toma de decisiones operativas basadas en datos.

### Descripción del Objetivo General
Simular, analizar y predecir ventas con base en datos climatológicos, almacenarlos en PostgreSQL y Supabase, y visualizar los resultados en un tablero interactivo en Looker Studio.

### 1. Tecnologías utilizadas
**Python 3.13**
Pandas, NumPy, Matplotlib, Seaborn
Scikit-learn (para modelos de regresión)
SQLAlchemy + psycopg2,
**PostgreSQL local
Supabase (PostgreSQL en la nube)
Looker Studio (visualización de datos)**

### 2. Instrucciones de ejecución
**I. Instalar dependencias**
pip install -r requirements.txt
pip install pandas numpy matplotlib seaborn scikit-learn requests sqlalchemy psycopg2 openpyxl streamlit
**II. Ejecutar el script principal**
python **Prueba_Técnica_CUN.py**

### 3.Acceso a la base de datos local
#### PostgreSQL local debe estar corriendo en:
Usuario: postgres
Contraseña: 010618Dani*
Puerto: 5432
Base de datos: postgres

### 4. Acceso a Supabase
postgresql://postgres.jogmxluulwzghopomzcc:010618Dani*@aws-0-sa-east-1.pooler.supabase.com:5432/postgres

### 5. Archivos generados relevantes para la visualización
**04.Ventas_clima_predicciones_supabase.csv:** Dataset final con predicciones para cargar en Supabase
**05.Query_crear_tabla_supabase.csv:** Consulta SQL de estructura
**06.Error_promedio_categoria_supabase.csv:** Resultado de la consulta de error promedio por categoría

### 6. Tablero en Looker Studio
El tablero permite visualizar de forma interactiva los resultados del análisis
Enlace directo al informe:

**https://lookerstudio.google.com/reporting/e60f4d08-731a-4684-b960-db740c921e6f**

**Componentes del tablero:**

**Título del informe:** Análisis de Ventas Reales vs. Predichas
**Gráfico de líneas:** Ventas reales vs. predichas por fecha
**Tabla:** Error promedio por categoría
**Mapa de calor:** Ventas por categoría y temperatura
**Filtros interactivos:** Por fecha y categoría

### 7. Resultados Relevantes
I. Se alcanzó un coeficiente R² superior al 0.8 para varios productos, lo que indica una buena capacidad predictiva.
II. Las predicciones se respaldan automáticamente en Supabase, asegurando trazabilidad.
III. El tablero interactivo permite a cualquier líder visualizar tendencias y errores sin necesidad de conocimientos técnicos.




