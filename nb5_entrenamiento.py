#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NB5 — Entrenamiento de Modelos (script para SLURM)
TFG: Detección de Intrusiones a Gran Escala utilizando ML Distribuido y Big Data
Autor: Eduardo Morillas Rodríguez
"""


# =============================================================================
# Imports y Configuración
# =============================================================================
import os
import time
import json
import traceback
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo (sin pantalla)
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml import Pipeline


from config import *


try:
    spark = get_spark_session("TFG_IDS_NB5_Entrenamiento")


    # =============================================================================
    # 5.1 — Carga del Dataset (NB4)
    # =============================================================================
    df = spark.read.parquet(os.path.join(DATA_FINAL_PATH, "dataset_final"))
    total_rows = df.count()
    n_classes = df.select("label_index").distinct().count()


    print(f"📊 Dataset cargado desde {os.path.join(DATA_FINAL_PATH, "dataset_final")}:")
    print(f"  Filas:    {total_rows:,}")
    print(f"  Clases:   {n_classes}")
    print(f"  Columnas: {len(df.columns)}")


    exclude_cols = {"Label", "label_index"}
    feature_cols = [
        f.name for f in df.schema.fields
        if f.name not in exclude_cols
        and isinstance(f.dataType, (DoubleType, FloatType, IntegerType, LongType))
    ]
    print(f"  Features: {len(feature_cols)}")


    # =============================================================================
    # 5.2 — VectorAssembler
    # =============================================================================
    print("🔧 Creando vector de features...")


    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    df = assembler.transform(df)


    print(f"  ✅ Vector 'features_raw' creado ({len(feature_cols)} dimensiones)")


    # =============================================================================
    # 5.3 — Train/Test Split Estratificado (80/20)
    # =============================================================================
    print("🔍 Split estratificado train/test (80/20)...")


    df = df.withColumn("_rand", F.rand(seed=SEED))


    fractions = {
        row["label_index"]: TRAIN_RATIO
        for row in df.select("label_index").distinct().collect()
    }


    df_train = df.sampleBy("label_index", fractions=fractions, seed=SEED)
    df_test = df.join(df_train.select("_rand"), on="_rand", how="left_anti")


    df_train = df_train.drop("_rand")
    df_test = df_test.drop("_rand")


    n_train = df_train.count()
    n_test = df_test.count()


    print(f"  Train: {n_train:,} ({n_train/total_rows*100:.1f}%)")
    print(f"  Test:  {n_test:,} ({n_test/total_rows*100:.1f}%)")


    print("\n  Distribución por clase en train:")
    df_train.groupBy("Label").count().orderBy(F.desc("count")).show(truncate=False)


    # =============================================================================
    # 5.4 — StandardScaler (fit SOLO en train)
    # =============================================================================
    print("🔧 Escalando features (fit solo en train)...")


    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True
    )


    scaler_model = scaler.fit(df_train)
    df_train = scaler_model.transform(df_train)
    df_test = scaler_model.transform(df_test)


    print(f"  ✅ StandardScaler aplicado")
    print(f"     Fit en train ({n_train:,} filas)")
    print(f"     Transform en train + test")


    # =============================================================================
    # 5.5 — Balanceo de Clases (SOLO en train)
    # =============================================================================
    print("🔍 Balanceo de clases (solo train)...")


    class_counts = (
        df_train.groupBy("Label", "label_index")
        .count()
        .orderBy(F.desc("count"))
        .toPandas()
    )
    print("\n  Distribución ANTES del balanceo:")
    print(class_counts.to_string(index=False))


    target_count = int(class_counts["count"].median())
    print(f"\n  Tamaño objetivo (mediana): {target_count:,}")


    balanced_dfs = []
    for _, row in class_counts.iterrows():
        label_idx = row["label_index"]
        count = int(row["count"])
        df_class = df_train.filter(F.col("label_index") == label_idx)


        if count > target_count:
            df_sampled = df_class.sample(False, target_count / count, seed=SEED)
        elif count < target_count:
            df_sampled = df_class.sample(True, target_count / count, seed=SEED)
        else:
            df_sampled = df_class


        balanced_dfs.append(df_sampled)


    df_train_balanced = balanced_dfs[0]
    for bdf in balanced_dfs[1:]:
        df_train_balanced = df_train_balanced.unionByName(bdf, allowMissingColumns=True)


    n_train_balanced = df_train_balanced.count()
    print(f"\n  Train tras balanceo: {n_train_balanced:,}")


    print("\n  Distribución DESPUÉS del balanceo:")
    df_train_balanced.groupBy("Label").count().orderBy(F.desc("count")).show(truncate=False)


    # Visualización
    bal_before = class_counts
    bal_after = df_train_balanced.groupBy("Label").count().orderBy(F.desc("count")).toPandas()


    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].barh(bal_before["Label"][::-1], bal_before["count"][::-1], color="#e74c3c", alpha=0.7)
    axes[0].set_title("ANTES del balanceo (train)")
    axes[0].set_xscale("log")
    axes[1].barh(bal_after["Label"][::-1], bal_after["count"][::-1], color="#2ecc71", alpha=0.7)
    axes[1].set_title("DESPUÉS del balanceo (train)")
    axes[1].set_xscale("log")
    fig.suptitle("Distribución de Clases — Balanceo Solo en Train", fontsize=16, y=1.02)
    plt.tight_layout()
    save_figure(fig, "33_balanceo_antes_despues")


    # =============================================================================
    # Cache y evaluadores
    # =============================================================================
    df_test.cache()


    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label_index", predictionCol="prediction", metricName="f1"
    )
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label_index", predictionCol="prediction", metricName="accuracy"
    )


    results = []


    # =============================================================================
    # 5.6 — Random Forest (Base)
    # =============================================================================
    print("🌲 Random Forest — Base...")


    rf_base = RandomForestClassifier(
        labelCol="label_index",
        featuresCol="features",
        numTrees=50,
        maxDepth=10,
        maxBins=16,
        seed=SEED
    )


    t0 = time.time()
    rf_base_model = rf_base.fit(df_train_balanced)
    t_rf_base = time.time() - t0


    pred_rf_base = rf_base_model.transform(df_test)
    f1_rf_base = evaluator_f1.evaluate(pred_rf_base)
    acc_rf_base = evaluator_acc.evaluate(pred_rf_base)


    print(f"  Accuracy: {acc_rf_base:.4f}")
    print(f"  F1-score: {f1_rf_base:.4f}")
    print(f"  Tiempo:   {t_rf_base:.1f}s")


    results.append({
        "modelo": "RF Base", "accuracy": acc_rf_base,
        "f1": f1_rf_base, "tiempo_s": t_rf_base
    })


    # =============================================================================
    # 5.7 — Random Forest (TrainValidationSplit)
    # =============================================================================
    print("\n🌲 Random Forest — TrainValidationSplit...")


    rf_tvs = RandomForestClassifier(
        labelCol="label_index",
        featuresCol="features",
        maxBins=16,
        seed=SEED
    )


    grid_rf = (
        ParamGridBuilder()
        .addGrid(rf_tvs.numTrees, [100, 200])
        .addGrid(rf_tvs.maxDepth, [8, 10])
        .addGrid(rf_tvs.featureSubsetStrategy, ["sqrt", "log2"])
        .build()
    )


    tvs_rf = TrainValidationSplit(
        estimator=rf_tvs,
        estimatorParamMaps=grid_rf,
        evaluator=evaluator_f1,
        trainRatio=0.75,
        parallelism=1,
        seed=SEED
    )


    n_combos = len(grid_rf)
    print(f"  Combinaciones: {n_combos} × 1 = {n_combos} entrenamientos")


    t0 = time.time()
    tvs_rf_model = tvs_rf.fit(df_train_balanced)
    t_rf_tvs = time.time() - t0


    best_rf = tvs_rf_model.bestModel
    pred_rf_best = best_rf.transform(df_test)
    f1_rf_best = evaluator_f1.evaluate(pred_rf_best)
    acc_rf_best = evaluator_acc.evaluate(pred_rf_best)


    print(f"\n  Mejor RF:")
    print(f"    numTrees:              {best_rf.getNumTrees}")
    print(f"    maxDepth:              {best_rf.getOrDefault('maxDepth')}")
    print(f"    featureSubsetStrategy: {best_rf.getOrDefault('featureSubsetStrategy')}")
    print(f"    Accuracy: {acc_rf_best:.4f}")
    print(f"    F1-score: {f1_rf_best:.4f}")
    print(f"    Tiempo TVS: {t_rf_tvs:.1f}s")


    print(f"\n  Métricas de validación por combinación:")
    for i, metric in enumerate(tvs_rf_model.validationMetrics):
        print(f"    Combo {i+1}: F1={metric:.4f}")


    best_rf.write().overwrite().save(os.path.join(MODELS_PATH, "best_random_forest"))
    print(f"  💾 Modelo RF guardado")


    results.append({
        "modelo": "RF TVS Best", "accuracy": acc_rf_best,
        "f1": f1_rf_best, "tiempo_s": t_rf_tvs
    })


    # =============================================================================
    # 5.8 — Gradient Boosted Trees — Base (OneVsRest)
    # =============================================================================
    print("🌳 GBT — Base (OneVsRest)...")


    gbt_base_classifier = GBTClassifier(
        labelCol="label_index",
        featuresCol="features",
        maxIter=50,
        maxDepth=5,
        stepSize=0.1,
        seed=SEED
    )


    ovr_base = OneVsRest(
        classifier=gbt_base_classifier,
        labelCol="label_index",
        featuresCol="features"
    )


    t0 = time.time()
    ovr_base_model = ovr_base.fit(df_train_balanced)
    t_gbt_base = time.time() - t0


    pred_gbt_base = ovr_base_model.transform(df_test)
    f1_gbt_base = evaluator_f1.evaluate(pred_gbt_base)
    acc_gbt_base = evaluator_acc.evaluate(pred_gbt_base)


    print(f"  Accuracy: {acc_gbt_base:.4f}")
    print(f"  F1-score: {f1_gbt_base:.4f}")
    print(f"  Tiempo:   {t_gbt_base:.1f}s")


    results.append({
        "modelo": "GBT OVR Base", "accuracy": acc_gbt_base,
        "f1": f1_gbt_base, "tiempo_s": t_gbt_base
    })


    # =============================================================================
    # 5.9 — GBT Tuning Manual (OneVsRest)
    # =============================================================================
    print("\n🌳 GBT — Tuning Manual (OneVsRest)...")


    df_gbt_train, df_gbt_val = df_train_balanced.randomSplit([0.75, 0.25], seed=SEED)


    param_grid = [
        {"maxIter": mi, "maxDepth": md, "stepSize": ss}
        for mi in [50, 100]
        for md in [5, 8]
        for ss in [0.05, 0.1]
    ]


    print(f"  Combinaciones: {len(param_grid)}")


    best_gbt_f1 = -1.0
    best_gbt_model = None
    best_gbt_params = {}
    gbt_tuning_results = []


    for i, params in enumerate(param_grid):
        print(f"\n  [{i+1}/{len(param_grid)}] maxIter={params['maxIter']}, "
              f"maxDepth={params['maxDepth']}, stepSize={params['stepSize']}")


        gbt_inner = GBTClassifier(
            labelCol="label_index",
            featuresCol="features",
            maxIter=params["maxIter"],
            maxDepth=params["maxDepth"],
            stepSize=params["stepSize"],
            seed=SEED
        )


        ovr = OneVsRest(
            classifier=gbt_inner,
            labelCol="label_index",
            featuresCol="features"
        )


        t0 = time.time()
        model = ovr.fit(df_gbt_train)
        t_fit = time.time() - t0


        preds_val = model.transform(df_gbt_val)
        f1_val = evaluator_f1.evaluate(preds_val)


        print(f"    F1 (val): {f1_val:.4f}  Tiempo: {t_fit:.1f}s")


        gbt_tuning_results.append({**params, "f1_val": f1_val, "tiempo_s": t_fit})


        if f1_val > best_gbt_f1:
            best_gbt_f1 = f1_val
            best_gbt_model = model
            best_gbt_params = params


    # Evaluar mejor modelo en test
    pred_gbt_best = best_gbt_model.transform(df_test)
    f1_gbt_best = evaluator_f1.evaluate(pred_gbt_best)
    acc_gbt_best = evaluator_acc.evaluate(pred_gbt_best)


    t_gbt_total = sum(r["tiempo_s"] for r in gbt_tuning_results)


    print(f"\n  Resultados de todas las combinaciones:")
    tuning_df = pd.DataFrame(gbt_tuning_results).sort_values("f1_val", ascending=False)
    print(tuning_df.to_string(index=False))


    print(f"\n  🏆 Mejor GBT (OneVsRest):")
    print(f"    maxIter:  {best_gbt_params['maxIter']}")
    print(f"    maxDepth: {best_gbt_params['maxDepth']}")
    print(f"    stepSize: {best_gbt_params['stepSize']}")
    print(f"    F1 (val):  {best_gbt_f1:.4f}")
    print(f"    F1 (test): {f1_gbt_best:.4f}")
    print(f"    Accuracy:  {acc_gbt_best:.4f}")
    print(f"    Tiempo total tuning: {t_gbt_total:.1f}s")


    best_gbt_model.write().overwrite().save(os.path.join(MODELS_PATH, "best_gbt_ovr"))
    print(f"  💾 Modelo GBT guardado")


    results.append({
        "modelo": "GBT OVR Best", "accuracy": acc_gbt_best,
        "f1": f1_gbt_best, "tiempo_s": t_gbt_total
    })


    # =============================================================================
    # 5.10 — Resumen
    # =============================================================================
    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)


    print("\n" + "=" * 70)
    print("📋 RESUMEN DE ENTRENAMIENTO — NB5")
    print("=" * 70)
    print(f"\n  Dataset:  {total_rows:,} filas, {len(feature_cols)} features, {n_classes} clases")
    print(f"  Train:    {n_train:,} → balanceado: {n_train_balanced:,}")
    print(f"  Test:     {n_test:,} (distribución real, sin balancear)")
    print(f"  Tuning RF:  TrainValidationSplit (trainRatio=0.75, 8 combos)")
    print(f"  Tuning GBT: Loop manual + OneVsRest (trainRatio=0.75, 8 combos × 15 clases)")
    print(f"\n  Resultados (evaluados en TEST):\n")
    print(results_df.to_string(index=False))


    metrics_path = os.path.join(LOGS_PATH, "nb5_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  💾 Métricas guardadas en {metrics_path}")


    best_model_name = results_df.iloc[0]["modelo"]
    best_f1 = results_df.iloc[0]["f1"]
    print(f"\n  🏆 Mejor modelo: {best_model_name} (F1={best_f1:.4f})")


    print("\n" + "=" * 70)
    print("✅ NB5 COMPLETADO")
    print("=" * 70)


except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()


finally:
    try:
        spark.stop()
        print("\n🛑 SparkSession cerrada")
    except:
        pass




