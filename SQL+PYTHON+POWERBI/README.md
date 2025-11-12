# SQL-PYTHON-POWERBI
Flujo de trabajo de un científico de datos (**ETL + EDA + MODELADO**), desde la exploración y preparación de datos con SQL, pasando por el modelado de Machine Learning con Python, hasta la visualización con Power BI. Se ultilizara la base de datos Sakila de MySql. 

**Objetivo general**: Nuestro objetivo será predecir el "valor" de un cliente o analizar patrones de alquiler para optimizar inventarios y promociones.

**Objetivo de SQL**: Extraer y preparar un dataset que contenga información de clientes y sus historiales de alquiler, incluyendo el valor total gastado y la frecuencia.

**Objetivo de Python (ML)**: Cargar los datos preparados por SQL con ayudada de sqlalchemy y construir un modelo de clasificación para predecir si un cliente es un "cliente VIP" (definido por el gasto total) basado en sus características.

**Objetivo de Power BI**: Crear un dashboard interactivo que visualice los resultados del análisis y las predicciones, permitiendo explorar los datos.

**Descripción de la base de datos**:La base de datos de ejemplo Sakila es una base de datos ficticia diseñada para representar una tienda de alquiler de DVD. Sus tablas incluyen, entre otras, las de película, categoría de película, actor, cliente, alquiler, pago e inventario. En concreto se realizara un filtro para obtener los usuarios Vip.

En concreto la carpeta "SQL-PYTHON-POWERBI" son 3 archivos en los que reafirmos mis conocimientos obtenicos tanto en el servicio social hecho en el Insituto Nacional de Medicina Genomica en Mexico, y en mi cursos de la carrera de Fisica en la UNAM. 

El analisis se hizo con aprendisaje supervisado, en concreto, **logistic regression** y **Suport Vector Machine(svm)**

Primeramente se utilizo Sql para exploración y Preparación de Datos con SQL (Notebook SQL). Uso de CTEs, VIEWs, JOINs, COALESCE...

Con python se hace un Modelado de Machine Learning (En Jupyter Notebook). Cargaremos los datos preparados por SQL en Python con ayuda de sqlalchemy, realizaremos un análisis exploratorio de datos (EDA) rápido, preprocesaremos los datos y entrenaremos un modelo de clasificación para predecir si un cliente es VIP. Se entrenaron dos modelos, utilizando solo la prediccion obtenida de Logistic Regression. Tambien guardamos las predicciones obtenidas de python en mysql para despues poder conectarnos a sql con Power Bi.

Por ultimo se hizo una Visualización de Resultados con Power BI, en el cual nos Conectamos a la base de datos MySQL.

Visualizaciones del Dashboard: Total de Clientes y Porcentaje VIP, Distribución de Clientes por País(En mapa coroplético), Gasto Total por País, Distribución del Gasto Total (Histograma), Comparación de Métricas entre VIP y No VIP, Tabla de Clientes con Predicciones y Segmentación de Predicciones por Distrito/Ciudad
