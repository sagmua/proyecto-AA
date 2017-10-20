# Proyecto de Aprendizaje Automático

### _Realizado por Samuel Cardenete Rodríguez y Juan José Sierra González_

En este proyecto se comparan modelos lineales y no lineales para un mismo problema de regresión.
El problema escogido para esta práctica es Housing (https://archive.ics.uci.edu/ml/machine-learning-databases/housing/), un problema de regresión que estima el valor de las viviendas de los suburbios de Boston en base a distintos parámetros.

En el proyecto se propone realizar una serie de modelos lineales para encontrar empíricamente el que parece ser el mejor, evaluando múltiples tuplas de características con las que predecir la variable respuesta. A continuación, se evalúan cuatro modelos no lineales (Redes Neuronales, Support Vector Machine, Boosting y Random Forest), analizando cómo optimizar cada uno de ellos y sus parámetros, y comparándolos entre sí y con el mejor modelo lineal encontrado.

Los datos son preprocesados utilizando el paquete _regsubsets_ y _PCA_, gracias al cual somos capaces de decidir qué características es mejor juntar para obtener un buen modelo. Para cada uno de los algoritmos se hace uso de 5-fold Cross-validation, una forma potente de asegurar que la tasa de error obtenida es significativa, y se muestra el código en R que se ha usado para hacer cada una de las operaciones.

Finalmente, se arroja una conclusión sobre cuál es el mejor modelo, o el más recomendable, para el problema de regresión escogido, y se aportan los argumentos que hacen que se tome esa decisión.
