# ============================================================
# config.py — Configuración Compartida del TFG
# ============================================================
# Detección de Intrusiones a Gran Escala utilizando
# Machine Learning Distribuido y Big Data
#
# Autor: Eduardo Morillas Rodríguez
# Curso: 2025-2026
#
# INFRAESTRUCTURA REAL — LORCA Computing Center (UEM)
# ──────────────────────────────────────────────────────────
# Nodo: 	eespcachcpro01 (grupo High Computing Research)
# CPU:  	Intel Xeon Gold 6338 @ 2.0 GHz
# Cores:	32 físicos / 64 hilos lógicos
# RAM:  	125 GB (≈100 GB disponibles)
# GPU:  	NVIDIA RTX 3080 Ti, 12 GB GDDR6X
# CUDA: 	13.1  ·  Driver: 590.48.01
# Disco:	20 GB cuota personal (SSD/NVMe compartida 300 GB)
# SLURM:	low_intensity (nodos 01-02), high_intensity (03)
# Acceso:   JupyterHub (10.151.52.x) · SSH · FortiClient VPN
# SO:   	Ubuntu / Debian
# ──────────────────────────────────────────────────────────
#
# DISTRIBUCIÓN DE MEMORIA (125 GB):
# ┌────────────────────────────────────────────────┐
# │ Spark JVM (driver)      	80 GB          	│ ← Procesamiento principal
# │ Python workers (PySpark)   ~20 GB          	│ ← toPandas, sklearn, t-SNE
# │ SO + servicios          	~5 GB          	│ ← Linux, SLURM, SSH
# │ Caché disco / headroom 	~20 GB          	│ ← Lectura Parquet, buffer
# └────────────────────────────────────────────────┘
#
# USO: En cada notebook → from config import *
# ============================================================


import os
import matplotlib.pyplot as plt
import matplotlib


# ============================================================
# PATHS DEL PROYECTO
# ============================================================


BASE_PATH = os.path.join(os.path.expanduser("~"), "tfg_ids")


DATA_RAW_PATH 	= os.path.join(BASE_PATH, "data", "raw")
DATA_PARQUET_PATH = os.path.join(BASE_PATH, "data", "parquet",
                                "cse_cic_ids_2018")
DATA_CLEAN_PATH   = os.path.join(BASE_PATH, "data", "clean")
DATA_FINAL_PATH   = os.path.join(BASE_PATH, "data", "final")
MODELS_PATH   	= os.path.join(BASE_PATH, "models")
FIGURES_PATH  	= os.path.join(BASE_PATH, "figures")
ANNEX_PATH    	= os.path.join(FIGURES_PATH, "anexo")
LOGS_PATH     	= os.path.join(BASE_PATH, "logs")


ALL_PATHS = [
    DATA_RAW_PATH, DATA_PARQUET_PATH, DATA_CLEAN_PATH,
    DATA_FINAL_PATH, MODELS_PATH, FIGURES_PATH, ANNEX_PATH, LOGS_PATH,
]


for _path in ALL_PATHS:
    os.makedirs(_path, exist_ok=True)




# ============================================================
# VARIABLES GLOBALES
# ============================================================


SEED                	= 42
TRAIN_RATIO         	= 0.8
TEST_RATIO          	= 0.2
CORRELATION_THRESHOLD   = 0.95
PCA_VARIANCE_THRESHOLD  = 0.95
NULL_THRESHOLD_DROP_COL = 0.50
NULL_THRESHOLD_DROP_ROW = 0.05
VIF_THRESHOLD       	= 10


# Umbral de aviso: archivos con >10% de labels inválidos se destacan
# en el informe de auditoría. NO se excluyen archivos enteros;
# solo se eliminan las filas con labels no reconocidos.
WARNING_CORRUPTION_PCT = 10


# Hardware del nodo (para documentación y NB7)
NODE_NAME   	= "eespcachcpro01"
CPU_MODEL   	= "Intel Xeon Gold 6338"
PHYSICAL_CORES  = 32
LOGICAL_THREADS = 64
RAM_TOTAL_GB	= 125
GPU_MODEL   	= "NVIDIA RTX 3080 Ti"
GPU_VRAM_GB 	= 12
DISK_QUOTA_GB   = 20




# ============================================================
# CATÁLOGO OFICIAL DE LABELS — CSE-CIC-IDS 2018
# ============================================================


VALID_LABELS = [
    "Benign",
    "Bot",
    "Brute Force -Web",
    "Brute Force -XSS",
    "DDoS attacks-LOIC-HTTP",
    "DDoS attack-HOIC",
    "DDoS attack-LOIC-UDP",
    "DoS attacks-GoldenEye",
    "DoS attacks-Hulk",
    "DoS attacks-SlowHTTPTest",
    "DoS attacks-Slowloris",
    "FTP-BruteForce",
    "Infilteration",
    "SQL Injection",
    "SSH-Bruteforce",
]




# ============================================================
# PALETA DE COLORES UNIFICADA
# ============================================================


COLOR_PALETTE = {
    "Benign":                 	"#2ecc71",
    "Bot":                    	"#e74c3c",
    "Brute Force -Web":       	"#3498db",
    "Brute Force -XSS":      	"#9b59b6",
    "DDoS attacks-LOIC-HTTP":	"#e67e22",
    "DDoS attack-HOIC":      	"#f39c12",
    "DDoS attack-LOIC-UDP":  	"#d35400",
    "DoS attacks-GoldenEye": 	"#1abc9c",
    "DoS attacks-Hulk":      	"#16a085",
    "DoS attacks-SlowHTTPTest":  "#2980b9",
    "DoS attacks-Slowloris": 	"#8e44ad",
    "FTP-BruteForce":        	"#c0392b",
    "Infilteration":         	"#7f8c8d",
    "SQL Injection":         	"#f1c40f",
    "SSH-Bruteforce":        	"#34495e",
}




# ============================================================
# ESTILO GLOBAL DE MATPLOTLIB
# ============================================================


matplotlib.rcParams["figure.figsize"]   = (14, 8)
matplotlib.rcParams["figure.dpi"]   	= 150
matplotlib.rcParams["font.size"]    	= 12
matplotlib.rcParams["axes.titlesize"]   = 16
matplotlib.rcParams["axes.labelsize"]   = 14
plt.style.use("seaborn-v0_8-whitegrid")




# ============================================================
# FUNCIONES AUXILIARES
# ============================================================


def save_figure(fig, name, dpi=150):
    """Guarda una figura matplotlib en FIGURES_PATH como PNG."""
    filepath = os.path.join(FIGURES_PATH, f"{name}.png")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  💾 Figura guardada: {filepath}")


def save_annex_figure(fig, name, dpi=150):
    """Guarda una figura en ANNEX_PATH (material complementario del TFG)."""
    filepath = os.path.join(ANNEX_PATH, f"{name}.png")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  💾 Figura guardada (Anexo): {filepath}")


def get_size_mb(path):
    """Devuelve el tamaño de un archivo o directorio en MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 ** 2)
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 2)


def print_dataframe_info(df, name="DataFrame"):
    """Imprime información básica de un DataFrame Spark."""
    print(f"\n📊 Info de {name}:")
    print(f"  Filas:	{df.count():,}")
    print(f"  Columnas: {len(df.columns)}")
    df.printSchema()


def get_spark_session(app_name="TFG_IDS", n_threads="*"):
    """
    Crea SparkSession optimizada para eespcachcpro01.

    Parámetros:
        app_name:   Nombre de la aplicación Spark.
        n_threads:  Hilos a usar. "*" = todos (64).
                    Usar 8, 16, 32, 64 para tests de escalabilidad (NB7).

    Recursos asignados (125 GB RAM total):
        · JVM heap:    	80 GB   (spark.driver.memory)
        · Ejecución/caché: 64 GB   (80 × 0.8 = 64 GB para procesar datos)
        · Storage (caché):  19 GB   (64 × 0.3 = 19 GB para DataFrames cacheados)
        · Python workers:  ~20 GB  (fuera de JVM — pandas, sklearn, etc.)
        · SO + buffer: 	~25 GB  (headroom para estabilidad)
    """
    from pyspark.sql import SparkSession

    # Paralelismo: 2× hilos para mejor distribución de carga
    if n_threads == "*":
        parallelism = LOGICAL_THREADS * 2  # 128
    else:
        parallelism = int(n_threads) * 2

    spark = (
        SparkSession.builder
        .appName(app_name)

        # --- Memoria (80 GB de 125 GB disponibles) ---
        .config("spark.driver.memory", "80g")
        .config("spark.driver.maxResultSize", "16g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.3")

        # --- Paralelismo ---
        .config("spark.default.parallelism", str(parallelism))
        .config("spark.sql.shuffle.partitions", str(parallelism))

        # --- Optimización de queries ---
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

        # --- Serialización y compresión ---
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.parquet.compression.codec", "snappy")

        # --- Python workers ---
        .config("spark.python.worker.memory", "4g")

        # --- Disco temporal (SSD/NVMe → shuffles rápidos) ---
        .config("spark.local.dir", os.path.join(BASE_PATH, "spark_tmp"))

        # --- Master: usar N hilos del nodo ---
        .master(f"local[{n_threads}]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

