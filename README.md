# Distributed IDS with Apache Spark — TFG UEM 2026

**Detección de Intrusiones a Gran Escala utilizando Machine Learning Distribuido y Big Data**

> Trabajo de Fin de Grado · Universidad Europea de Madrid · Grado en Ingeniería Matemática aplicada al análisis de datos 
> Autor: **Eduardo Morillas Rodríguez** · Curso 2025–2026

---

## Descripción

Pipeline de procesamiento de datos y aprendizaje automático distribuido implementado en **Apache Spark (PySpark)** para la clasificación multiclase de tráfico de red a gran escala sobre el dataset **CSE-CIC-IDS 2018** (16,2 millones de flujos y 15 clases de tráfico). 

El sistema implementa una arquitectura en capas tipo Medallion (Bronce, Plata y Oro) y compara el rendimiento y la escalabilidad de dos algoritmos de Ensemble Learning: **Random Forest** y **Gradient Boosted Trees (One-vs-Rest)**.

### Resultados principales (evaluados en Test con distribución real)

| Modelo | F1 ponderado | Accuracy | Tiempo de ejecución |
|:---|:---:|:---:|---:|
| Random Forest (200 árboles, maxDepth=10) | **0,8955** | 0,8305 | ~35 min (TVS) |
| GBT One-vs-Rest (100 iter, maxDepth=8) | **0,9601** | 0,9384 | ~44,5 h (Tuning manual) |

---

## Estructura del repositorio

```
tfg_ids/
├── config.py                     # Configuración de rutas, SparkSession y paleta gráfica
├── nb0_configuracion.ipynb       # Pruebas iniciales y verificación del entorno Spark
├── nb1_ingesta.ipynb             # Capa Bronce: Ingesta de CSVs distribuidos y tipado explícito
├── nb2_eda.ipynb                 # Análisis Exploratorio de Datos (distribución, correlación, PCA)
├── nb3_limpieza.ipynb            # Capa Plata: Saneamiento de infinitos, duplicados y ventanas TCP
├── nb4_feature_engineering.ipynb # Capa Oro: VectorAssembler y selección por correlación/MDI
├── nb5_entrenamiento.py          # Script de entrenamiento de Random Forest y GBT One-vs-Rest
├── nb6_evaluacion.ipynb          # Notebook de evaluación: curvas ROC, PR, matrices y comparativa
├── nb6_evaluacion.py             # Export del notebook de evaluación
├── nb7_escabilidad.py            # Script de pruebas de escalabilidad (volumen, Strong Scaling)
├── slurm_nb5.sh                  # Script de ejecución del job SLURM para el entrenamiento (NB5)
├── slurm_nb7.sh                  # Script de ejecución del job SLURM para la escalabilidad (NB7)
├── logs/                         # Ficheros de log y métricas detalladas en JSON
│   ├── nb5_265.log               # Log de ejecución del entrenamiento (RF y GBT OVR)
│   ├── nb5_metrics.json          # Métricas de entrenamiento guardadas en JSON
│   ├── nb7_314.log               # Log de ejecución de las pruebas de escalabilidad
│   └── nb7_metrics.json          # Métricas de escalabilidad (Amdahl, velocidad) en JSON
├── figures/                      # Gráficos e informes de salida (EDA, matrices de confusión)
└── .gitignore                    # Reglas de exclusión para Git (datos, temporales y modelos pesados)
```

---

## Requisitos del entorno

### Software

* **Python:** 3.10
* **Apache Spark:** 3.5.x
* **PySpark:** 3.5.x
* **Java JDK:** 11 o 17
* **Gestor de colas:** SLURM (opcional, para entornos HPC)

### Instalación de dependencias de Python

```bash
pip install pyspark==3.5.0 pandas numpy scikit-learn matplotlib seaborn statsmodels
```

### Configuración del Hardware recomendado

Los modelos pesados (especialmente GBT OVR) requieren una cantidad sustancial de recursos para evitar errores de OOM (Out Of Memory):
* **RAM mínima:** 64 GB (Recomendado 125 GB para el Driver de Spark en ejecuciones locales).
* **Almacenamiento:** Mínimo 20 GB de espacio en disco (SSD/NVMe recomendado para gestionar el almacenamiento intermedio de _shuffles_).

---

## Preparación de los datos

1. Descarga el dataset oficial **CSE-CIC-IDS 2018** de la web de la [UNB](https://www.unb.ca/cic/datasets/ids-2018.html).
2. Crea el directorio de datos crudos: `data/raw/`.
3. Coloca en él los 10 archivos CSV del dataset original.
4. Ajusta la variable `BASE_PATH` en el script `config.py` para apuntar a la ruta raíz de tu carpeta.

---

## Ejecución del pipeline

El pipeline está diseñado de forma secuencial, donde cada etapa lee la salida de formato Parquet generada por la anterior:

```
nb1_ingesta ──> nb2_eda ──> nb3_limpieza ──> nb4_feature_engineering ──> nb5_entrenamiento ──> nb6_evaluacion
                                                                   └──> nb7_escalabilidad
```

### Ejecución interactiva
Para ejecutar localmente las fases interactivas de análisis y exploración:
```bash
jupyter lab
```

### Lanzamiento en el clúster HPC (SLURM)
Los scripts bash `slurm_nb5.sh` y `slurm_nb7.sh` automatizan la asignación de recursos y ejecutan los scripts de entrenamiento y escalabilidad en segundo plano:

```bash
# Lanzar entrenamiento de modelos
sbatch slurm_nb5.sh

# Lanzar pruebas de escalabilidad
sbatch slurm_nb7.sh
```

---

## Cita en publicaciones académicas

Si este proyecto te ha resultado útil en tu investigación o trabajo, por favor cítalo utilizando el siguiente formato APA 7:

```text
Morillas Rodríguez, E. (2026). Distributed IDS with Apache Spark: detección de
intrusiones a gran escala mediante Ensemble Learning [Software]. GitHub.
https://github.com/[USUARIO]/[NOMBRE-REPO]
```

---

## Licencia

Este proyecto está bajo la Licencia MIT. Para más detalles, ver el archivo `LICENSE`.
