# %% [markdown]
# # Notebook 6 — Evaluación y Comparativa de Modelos 📈
# **TFG: Detección de Intrusiones a Gran Escala utilizando ML Distribuido y Big Data**
#
# **Autor:** Eduardo Morillas Rodríguez
#
# **Entrada:** Modelos de `MODELS_PATH`, datos de `DATA_FINAL_PATH`
# **Salida:** Figuras en `FIGURES_PATH`, métricas detalladas en `LOGS_PATH`
#
# ---
#
# ## Relación con notebooks anteriores
#
# | NB | Qué aporta a NB6 |
# |----|-------------------|
# | NB1-NB3 | Pipeline de limpieza: dataset sin inf/dup/neg |
# | NB4 | Feature engineering + selección: 45 features + `Label` + `label_index` |
# | NB5 | Modelos entrenados: `best_random_forest` (RF TVS) y `best_gbt_ovr` (GBT OVR) |
#
# ## Qué hace este notebook
#
# ✅ Carga los modelos entrenados en NB5 (sin re-entrenar)
# ✅ Reconstruye el pipeline de preparación (VectorAssembler + split + StandardScaler)
# ✅ Genera predicciones sobre el test set (distribución real, sin balancear)
# ✅ Métricas globales: Accuracy, F1, Precision, Recall
# ✅ Métricas por clase: classification_report detallado
# ✅ Visualizaciones: Matrices de confusión, ROC, PR, comparativas
# ✅ Análisis de errores: principales confusiones entre clases
# ✅ Comparativa con el estado del arte
#
# ## Respaldo en la literatura
#
# ### Evaluación sobre test sin balancear
# - **Shanmugam et al. (2025, MDPI Electronics)**: "The test set must reflect the
#   real-world class distribution to obtain realistic performance estimates."
#   El test set no se balancea para que las métricas sean representativas del
#   rendimiento en producción.
#
# ### Métricas para datasets desbalanceados
# - **Chimphlee & Chimphlee (2023, IJCNC)**: Usan F1-score (weighted) como métrica
#   principal para IDS con clases desbalanceadas. Accuracy sola es engañosa
#   (87% prediciendo siempre Benign).
# - **Abdelaziz et al. (2025, Springer)**: Reportan weighted F1 además de accuracy;
#   precision y recall por clase para identificar debilidades en clases minoritarias.
#
# ### Matrices de confusión y análisis de errores
# - **Songma et al. (2023, MDPI Computers)**: Analizan confusiones entre clases
#   de ataque similares (DoS variants, Brute Force variants) para entender
#   limitaciones del modelo.
# - **Leevy & Khoshgoftaar (2020, J. Big Data)**: Survey de CSE-CIC-IDS 2018;
#   destacan que las clases minoritarias (SQL Injection, XSS) son las más
#   difíciles de clasificar correctamente.
#
# ### Curvas ROC y PR
# - **Fawcett (2006, Pattern Recognition Letters)**: Referencia clásica para
#   interpretación de curvas ROC. AUC resume el rendimiento discriminativo
#   del clasificador en un solo valor.
# - **Davis & Goadrich (2006, ICML)**: Precision-Recall es más informativa
#   que ROC cuando las clases están severamente desbalanceadas.
#
# ## Nota sobre curvas ROC/PR y OneVsRest
#
# `OneVsRestModel` (usado para GBT) no genera vectores de probabilidad
# en su output — solo devuelve `prediction`. Por ello, las curvas ROC y PR
# solo se generan para RF (que sí produce `probability`). Esta es una
# limitación conocida de la implementación de Spark MLlib.


# %% [markdown]
# ---
# ## Imports y Configuración


# %%
%autosave 60

import json
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
	confusion_matrix, classification_report,
	roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassificationModel, OneVsRestModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from config import *

spark = get_spark_session("TFG_IDS_NB6_Evaluacion")


# %% [markdown]
# ---
# ## 6.0 — Reconstrucción del Pipeline de Preparación
#
# Los modelos de NB5 esperan un vector `features` escalado. Necesitamos
# reconstruir el mismo pipeline (VectorAssembler → split estratificado →
# StandardScaler) con las mismas semillas para obtener el mismo test set.
#
# > **Nota:** El scaler se re-ajusta sobre df_train con los mismos datos
# > y parámetros → produce idénticos resultados que en NB5.


# %%
# Cargar dataset
df = spark.read.parquet(os.path.join(DATA_FINAL_PATH, "dataset_final"))
total_rows = df.count()

# Identificar feature columns
exclude_cols = {"Label", "label_index"}
feature_cols = [
	f.name for f in df.schema.fields
	if f.name not in exclude_cols
	and isinstance(f.dataType, (DoubleType, FloatType, IntegerType, LongType))
]

# VectorAssembler
assembler = VectorAssembler(
	inputCols=feature_cols,
	outputCol="features_raw",
	handleInvalid="skip"
)
df = assembler.transform(df)

# Split estratificado (mismo que NB5)
df = df.withColumn("_rand", F.rand(seed=SEED))
fractions = {
	row["label_index"]: TRAIN_RATIO
	for row in df.select("label_index").distinct().collect()
}
df_train = df.sampleBy("label_index", fractions=fractions, seed=SEED)
df_test = df.join(df_train.select("_rand"), on="_rand", how="left_anti")
df_train = df_train.drop("_rand")
df_test = df_test.drop("_rand")

# StandardScaler (fit en train, transform en test — mismo que NB5)
scaler = StandardScaler(
	inputCol="features_raw",
	outputCol="features",
	withMean=True,
	withStd=True
)
scaler_model = scaler.fit(df_train)
df_test = scaler_model.transform(df_test)

n_test = df_test.count()

# Mapa de labels
label_map = df.select("Label", "label_index").distinct().orderBy("label_index").toPandas()
idx_to_label = dict(zip(label_map["label_index"].astype(int), label_map["Label"]))
class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
n_classes = len(class_names)

print(f"📊 Test set reconstruido: {n_test:,} filas, {n_classes} clases")
print(f"   Features: {len(feature_cols)}")


# %% [markdown]
# ---
# ## 6.0.1 — Carga de Modelos y Predicciones
#
# Se cargan los mejores modelos de NB5:
# - **RF TVS Best**: RandomForest con numTrees=200, maxDepth=10, sqrt
# - **GBT OVR Best**: GBT OneVsRest con maxIter=100, maxDepth=8, stepSize=0.1


# %%
# Cargar modelos
best_rf = RandomForestClassificationModel.load(os.path.join(MODELS_PATH, "best_random_forest"))
best_gbt = OneVsRestModel.load(os.path.join(MODELS_PATH, "best_gbt_ovr"))

print(f"✅ Modelos cargados:")
print(f"   RF:  numTrees={best_rf.getNumTrees}, maxDepth={best_rf.getOrDefault('maxDepth')}")
print(f"   GBT: OneVsRest (15 clasificadores binarios)")

# Generar predicciones
predictions_rf = best_rf.transform(df_test)
predictions_gbt = best_gbt.transform(df_test)

# Convertir a pandas para sklearn
pred_rf_pd = predictions_rf.select("label_index", "prediction", "probability").toPandas()
pred_gbt_pd = predictions_gbt.select("label_index", "prediction").toPandas()

print(f"✅ Predicciones generadas sobre {n_test:,} filas de test")


# %% [markdown]
# ---
# ## 6.1 — Métricas Globales
#
# ### Métricas utilizadas
#
# - **Accuracy**: Proporción de predicciones correctas. Con test desbalanceado
#   (Benign ≈87%), puede ser engañosa.
# - **F1 (weighted)**: Media armónica de precision y recall, ponderada por
#   el soporte de cada clase. Más representativa para datasets desbalanceados.
# - **Precision (weighted)**: Proporción de predicciones positivas correctas.
# - **Recall (weighted)**: Proporción de positivos reales detectados.
#
# > **Ref:** Chimphlee & Chimphlee (2023): F1-score como métrica principal
# > para IDS desbalanceados. Abdelaziz et al. (2025): Weighted F1 + accuracy.


# %%
print("📊 Métricas globales (evaluadas en TEST — distribución real):\n")

metrics = ["accuracy", "f1", "weightedPrecision", "weightedRecall"]
metric_names = ["Accuracy", "F1 (weighted)", "Precision (weighted)", "Recall (weighted)"]

results_global = {"Métrica": metric_names, "RF TVS Best": [], "GBT OVR Best": []}

for m in metrics:
	ev = MulticlassClassificationEvaluator(
    	labelCol="label_index", predictionCol="prediction", metricName=m
	)
	results_global["RF TVS Best"].append(f"{ev.evaluate(predictions_rf):.4f}")
	results_global["GBT OVR Best"].append(f"{ev.evaluate(predictions_gbt):.4f}")

results_global_df = pd.DataFrame(results_global)
print(results_global_df.to_string(index=False))

# Cargar métricas de NB5 para comparar con base
metrics_path = os.path.join(LOGS_PATH, "nb5_metrics.json")
if os.path.exists(metrics_path):
	with open(metrics_path, "r") as f:
    	nb5_metrics = json.load(f)
	print(f"\n📋 Métricas de NB5 (para referencia):")
	for m in nb5_metrics:
    	print(f"   {m['modelo']}: F1={m['f1']:.4f}, Acc={m['accuracy']:.4f}, Tiempo={m['tiempo_s']:.0f}s")


# %% [markdown]
# ---
# ## 6.2 — Métricas por Clase
#
# El desglose por clase es esencial en IDS: un modelo con F1 global alto
# puede fallar completamente en clases minoritarias (SQL Injection: ~50 muestras
# en test). Reportar precision/recall/F1 por clase permite identificar
# exactamente dónde falla cada modelo.
#
# > **Ref:** Abdelaziz et al. (2025): Reportan métricas por clase para
# > identificar debilidades en ataques minoritarios.
# > Leevy & Khoshgoftaar (2020): Las clases minoritarias son las más
# > difíciles y las más críticas en detección de intrusiones.


# %%
y_true_rf = pred_rf_pd["label_index"].astype(int).values
y_pred_rf = pred_rf_pd["prediction"].astype(int).values
y_true_gbt = pred_gbt_pd["label_index"].astype(int).values
y_pred_gbt = pred_gbt_pd["prediction"].astype(int).values

report_rf = classification_report(
	y_true_rf, y_pred_rf, target_names=class_names,
	output_dict=True, zero_division=0
)
report_gbt = classification_report(
	y_true_gbt, y_pred_gbt, target_names=class_names,
	output_dict=True, zero_division=0
)

report_rf_df = pd.DataFrame(report_rf).T
report_gbt_df = pd.DataFrame(report_gbt).T

print("📊 RF TVS Best — Métricas por clase:")
print(report_rf_df.to_string())
print(f"\n{'='*70}")
print("\n📊 GBT OVR Best — Métricas por clase:")
print(report_gbt_df.to_string())


# %% [markdown]
# ---
# ## 6.3 — Visualizaciones
#
# ### Figuras generadas
#
# | Nº | Contenido |
# |----|-----------|
# | 34 | Matriz de confusión — RF |
# | 35 | Matriz de confusión — GBT |
# | 36 | Matriz de confusión normalizada — RF |
# | 37 | Matriz de confusión normalizada — GBT |
# | 38 | Curvas ROC — RF (solo RF: GBT OVR no genera probabilidades) |
# | 39 | Curvas Precision-Recall — RF |
# | 40 | Comparativa global RF vs GBT |
# | 41 | F1 por clase — RF vs GBT |


# %%
# Figura 34: Confusion Matrix RF (valores absolutos)
cm_rf = confusion_matrix(y_true_rf, y_pred_rf)

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
	cm_rf, annot=True, fmt="d", cmap="Blues",
	xticklabels=class_names, yticklabels=class_names,
	linewidths=0.5, ax=ax
)
ax.set_xlabel("Predicción", fontsize=12)
ax.set_ylabel("Real", fontsize=12)
ax.set_title("Matriz de Confusión — RF TVS Best (valores absolutos)", fontsize=14)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save_figure(fig, "34_confusion_matrix_rf")


# %%
# Figura 35: Confusion Matrix GBT (valores absolutos)
cm_gbt = confusion_matrix(y_true_gbt, y_pred_gbt)

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
	cm_gbt, annot=True, fmt="d", cmap="Oranges",
	xticklabels=class_names, yticklabels=class_names,
	linewidths=0.5, ax=ax
)
ax.set_xlabel("Predicción", fontsize=12)
ax.set_ylabel("Real", fontsize=12)
ax.set_title("Matriz de Confusión — GBT OVR Best (valores absolutos)", fontsize=14)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save_figure(fig, "35_confusion_matrix_gbt")


# %%
# Figura 36: Confusion Matrix RF (normalizada por filas)
cm_rf_norm = cm_rf.astype(float) / cm_rf.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
	cm_rf_norm, annot=True, fmt=".2f", cmap="Blues",
	xticklabels=class_names, yticklabels=class_names,
	linewidths=0.5, ax=ax, vmin=0, vmax=1
)
ax.set_xlabel("Predicción", fontsize=12)
ax.set_ylabel("Real", fontsize=12)
ax.set_title("Matriz de Confusión Normalizada — RF TVS Best (recall por clase)", fontsize=14)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save_figure(fig, "36_confusion_matrix_rf_norm")


# %%
# Figura 37: Confusion Matrix GBT (normalizada por filas)
cm_gbt_norm = cm_gbt.astype(float) / cm_gbt.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
	cm_gbt_norm, annot=True, fmt=".2f", cmap="Oranges",
	xticklabels=class_names, yticklabels=class_names,
	linewidths=0.5, ax=ax, vmin=0, vmax=1
)
ax.set_xlabel("Predicción", fontsize=12)
ax.set_ylabel("Real", fontsize=12)
ax.set_title("Matriz de Confusión Normalizada — GBT OVR Best (recall por clase)", fontsize=14)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save_figure(fig, "37_confusion_matrix_gbt_norm")


# %%
# Figura 38: ROC Curves RF
# Nota: Solo RF genera vectores de probabilidad. OneVsRestModel (GBT)
# solo devuelve prediction, sin probability → no se puede calcular ROC.
y_true_bin = label_binarize(y_true_rf, classes=list(range(n_classes)))

try:
	proba_rf = np.array(pred_rf_pd["probability"].tolist())

	fig, ax = plt.subplots(figsize=(14, 10))
	auc_scores = {}
	for i, cls in enumerate(class_names):
    	fpr, tpr, _ = roc_curve(y_true_bin[:, i], proba_rf[:, i])
    	auc_score = auc(fpr, tpr)
    	auc_scores[cls] = auc_score
    	ax.plot(fpr, tpr, linewidth=1.5,
            	label=f"{cls} (AUC={auc_score:.3f})", alpha=0.8)

	ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.500)")
	ax.set_xlabel("False Positive Rate", fontsize=12)
	ax.set_ylabel("True Positive Rate", fontsize=12)
	ax.set_title("Curvas ROC — RF TVS Best", fontsize=14)
	ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
	plt.tight_layout()
	save_figure(fig, "38_roc_curves_rf")

	# AUC medio
	macro_auc = np.mean(list(auc_scores.values()))
	print(f"  AUC macro-average: {macro_auc:.4f}")
except Exception as e:
	print(f"  ⚠️ ROC no generadas: {e}")


# %%
# Figura 39: Precision-Recall Curves RF
try:
	fig, ax = plt.subplots(figsize=(14, 10))
	ap_scores = {}
	for i, cls in enumerate(class_names):
    	prec, rec, _ = precision_recall_curve(y_true_bin[:, i], proba_rf[:, i])
    	ap = average_precision_score(y_true_bin[:, i], proba_rf[:, i])
    	ap_scores[cls] = ap
    	ax.plot(rec, prec, linewidth=1.5,
            	label=f"{cls} (AP={ap:.3f})", alpha=0.8)

	ax.set_xlabel("Recall", fontsize=12)
	ax.set_ylabel("Precision", fontsize=12)
	ax.set_title("Curvas Precision-Recall — RF TVS Best", fontsize=14)
	ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
	plt.tight_layout()
	save_figure(fig, "39_precision_recall_rf")

	# AP medio
	macro_ap = np.mean(list(ap_scores.values()))
	print(f"  AP macro-average: {macro_ap:.4f}")
except Exception as e:
	print(f"  ⚠️ PR no generadas: {e}")


# %%
# Figura 40: Comparativa global RF vs GBT
results_plot = results_global_df.copy()
results_plot["RF TVS Best"] = results_plot["RF TVS Best"].astype(float)
results_plot["GBT OVR Best"] = results_plot["GBT OVR Best"].astype(float)

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(results_plot))
w = 0.35
bars_rf = ax.bar(x - w/2, results_plot["RF TVS Best"], w,
              	label="RF TVS Best", color="#3498db", alpha=0.8)
bars_gbt = ax.bar(x + w/2, results_plot["GBT OVR Best"], w,
               	label="GBT OVR Best", color="#e67e22", alpha=0.8)

# Añadir valores sobre las barras
for bar in bars_rf:
	ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
        	f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
for bar in bars_gbt:
	ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
        	f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(results_plot["Métrica"])
ax.legend(fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_title("Comparativa Global — RF vs GBT (evaluado en test)", fontsize=14)
ax.set_ylabel("Score", fontsize=12)
plt.tight_layout()
save_figure(fig, "40_comparativa_rf_gbt")


# %%
# Figura 41: F1 por clase — RF vs GBT
f1_cls = pd.DataFrame({
	"Clase": class_names,
	"RF": [report_rf_df.loc[c, "f1-score"] if c in report_rf_df.index else 0 for c in class_names],
	"GBT": [report_gbt_df.loc[c, "f1-score"] if c in report_gbt_df.index else 0 for c in class_names]
}).sort_values("GBT")

fig, ax = plt.subplots(figsize=(14, 10))
y = np.arange(len(f1_cls))
h = 0.35
ax.barh(y - h/2, f1_cls["RF"], h, label="RF TVS Best", color="#3498db", alpha=0.8)
ax.barh(y + h/2, f1_cls["GBT"], h, label="GBT OVR Best", color="#e67e22", alpha=0.8)
ax.set_yticks(y)
ax.set_yticklabels(f1_cls["Clase"])
ax.legend(fontsize=12)
ax.set_xlim(0, 1.1)
ax.set_xlabel("F1-score", fontsize=12)
ax.set_title("F1 por Clase — RF vs GBT (evaluado en test)", fontsize=14)
plt.tight_layout()
save_figure(fig, "41_f1_por_clase")


# %% [markdown]
# ---
# ## 6.4 — Análisis de Errores
#
# ### Objetivo
#
# Identificar las principales confusiones entre clases para entender
# las limitaciones de cada modelo. Esto es especialmente relevante para:
# - Clases minoritarias (SQL Injection, XSS) que pueden ser mal clasificadas
# - Clases similares (DoS variants, Brute Force variants) que comparten
#   patrones de tráfico
#
# > **Ref:** Songma et al. (2023): Analizan confusiones entre clases
# > similares para entender limitaciones del modelo.


# %%
print("📋 Top 15 confusiones — RF TVS Best:\n")

cm_rf_norm = cm_rf.astype(float) / cm_rf.sum(axis=1, keepdims=True)
confusions_rf = []
for i in range(n_classes):
	for j in range(n_classes):
    	if i != j and cm_rf[i, j] > 0:
        	confusions_rf.append({
            	"Real": class_names[i],
            	"Predicho": class_names[j],
            	"N": cm_rf[i, j],
            	"%": f"{cm_rf_norm[i, j]*100:.2f}%"
        	})
conf_rf_df = pd.DataFrame(confusions_rf).sort_values("N", ascending=False)
print(conf_rf_df.head(15).to_string(index=False))

print(f"\n{'='*70}")
print("\n📋 Top 15 confusiones — GBT OVR Best:\n")

cm_gbt_norm = cm_gbt.astype(float) / cm_gbt.sum(axis=1, keepdims=True)
confusions_gbt = []
for i in range(n_classes):
	for j in range(n_classes):
    	if i != j and cm_gbt[i, j] > 0:
        	confusions_gbt.append({
            	"Real": class_names[i],
            	"Predicho": class_names[j],
            	"N": cm_gbt[i, j],
            	"%": f"{cm_gbt_norm[i, j]*100:.2f}%"
        	})
conf_gbt_df = pd.DataFrame(confusions_gbt).sort_values("N", ascending=False)
print(conf_gbt_df.head(15).to_string(index=False))


# %% [markdown]
# ---
# ## 6.5 — Comparativa con el Estado del Arte
#
# ### Contexto
#
# La comparación directa con otros trabajos requiere cautela:
# - Cada estudio usa un subconjunto diferente de features
# - Algunos no balancean y reportan accuracy inflada por la clase Benign (87%)
# - Algunos evalúan sobre datos balanceados → métricas no representativas
# - Pocos usan frameworks distribuidos (Spark) → diferentes escalas
#
# **Este TFG evalúa sobre el test set con distribución real (sin balancear)**,
# lo que produce métricas más conservadoras pero más honestas.
#
# > **Ref:** Leevy & Khoshgoftaar (2020): "The best performance scores for
# > each study were unexpectedly high overall" — muchos reportan >99% accuracy
# > sin balanceo, lo que infla las métricas artificialmente.


# %%
print("📊 Comparativa con el Estado del Arte:\n")

sota = pd.DataFrame({
	"Estudio": [
    	"Este TFG (RF)",
    	"Este TFG (GBT)",
    	"sara-7 (2026, GitHub)",
    	"Songma et al. (2023)",
    	"Chimphlee & Chimphlee (2023)",
    	"Abdelaziz et al. (2025)",
    	"Göcs & Johanyák (2023)",
	],
	"Dataset": [
    	"CSE-CIC-IDS 2018",
    	"CSE-CIC-IDS 2018",
    	"CSE-CIC-IDS 2018",
    	"CSE-CIC-IDS 2018",
    	"CICIDS-2018",
    	"CICIDS-2017",
    	"CSE-CIC-IDS 2018",
	],
	"Método": [
    	"RF (Spark MLlib)",
    	"GBT OvR (Spark MLlib)",
    	"RF + GBT (Spark)",
    	"RF + XGBoost",
    	"Ensemble tree-based",
    	"RF (26 features)",
    	"RF (5-15 features)",
	],
	"Accuracy": [
    	results_global["RF TVS Best"][0],
    	results_global["GBT OVR Best"][0],
    	">99%",
    	"~97-99%",
    	"98.36%",
    	"99.8%",
    	"~95-98%",
	],
	"F1": [
    	results_global["RF TVS Best"][1],
    	results_global["GBT OVR Best"][1],
    	">0.999",
    	"~0.97-0.99",
    	"97.98%",
    	"99.8%",
    	"N/A",
	],
	"Distribuido": [
    	"✅ Spark",
    	"✅ Spark",
    	"✅ Spark",
    	"❌ No",
    	"❌ No",
    	"❌ No",
    	"❌ No",
	],
	"Test balanceado": [
    	"❌ Real dist.",
    	"❌ Real dist.",
    	"N/A",
    	"N/A",
    	"N/A",
    	"N/A",
    	"N/A",
	],
})

print(sota.to_string(index=False))

print("""
📝 Notas:
   - Las métricas de otros estudios no son directamente comparables:
 	muchos evalúan sobre datos balanceados o no reportan F1 weighted.
   - Este TFG evalúa sobre test con distribución REAL (Benign ≈87%),
 	lo que produce métricas más conservadoras pero más representativas
 	del rendimiento en producción.
   - Los estudios con >99% accuracy suelen no balancear → el modelo
 	predice "Benign" y acierta 87% automáticamente.
""")


# %% [markdown]
# ---
# ## 6.6 — Resumen y Conclusiones


# %%
print("\n" + "=" * 70)
print("📋 RESUMEN DE EVALUACIÓN — NB6")
print("=" * 70)

print(f"""
  Dataset: {total_rows:,} filas, {len(feature_cols)} features, {n_classes} clases
  Test:	{n_test:,} filas (distribución real, sin balancear)

  Métricas globales:
""")
print(results_global_df.to_string(index=False))

print(f"""

  Hallazgos principales:
  1. GBT OVR supera a RF en todas las métricas (F1: 0.96 vs 0.89)
  2. El tuning mejoró GBT significativamente (base: 0.926 → best: 0.960)
  3. El tuning apenas mejoró RF (base: 0.895 → best: 0.896)
  4. GBT es ~80× más lento que RF (44.5h vs 35 min)
  5. Las clases minoritarias son las más difíciles para ambos modelos

  Trade-off RF vs GBT:
  ┌─────────┬────────────┬────────────┐
  │     	│ RF     	│ GBT    	│
  ├─────────┼────────────┼────────────┤
  │ F1  	│ 0.8955 	│ 0.9601 	│
  │ Tiempo  │ 35 min 	│ 44.5h  	│
  │ Uso 	│ Tiempo real│ Batch  	│
  └─────────┴────────────┴────────────┘
""")

print("=" * 70)
print("\n➡️  Siguiente: NB7 — Escalabilidad")
print("	· Análisis de rendimiento vs número de cores")
print("	· Strong scaling / Weak scaling")


# %%
# spark.stop()

