from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


warnings.filterwarnings("ignore")
plt.style.use("ggplot")


RAW_DATASET = Path("dataset.csv")
ARTIFACTS_DIR = Path("artifacts")
TREATED_DATASET_PATH = ARTIFACTS_DIR / "dataset_tratado.csv"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CLUSTER_PROFILE_PATH = ARTIFACTS_DIR / "cluster_profiles.csv"
CLUSTER_INTERPRETATION_PATH = ARTIFACTS_DIR / "cluster_interpretation.txt"
BUNDLE_PATH = ARTIFACTS_DIR / "insurance_cluster_bundle.joblib"
ELBOW_PATH = ARTIFACTS_DIR / "elbow.png"
SILHOUETTE_PATH = ARTIFACTS_DIR / "silhouette.png"
PCA_PATH = ARTIFACTS_DIR / "clusters_pca.png"
CLUSTER_AGE_CHARGES_PATH = ARTIFACTS_DIR / "cluster_age_charges.png"
CLUSTER_BMI_CHARGES_PATH = ARTIFACTS_DIR / "cluster_bmi_charges.png"
CLUSTER_BOXPLOT_PATH = ARTIFACTS_DIR / "cluster_charges_boxplot.png"

MODEL_NAMES = ("linear", "knn", "tree")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip().str.lower()

    for column in cleaned.select_dtypes(include="object").columns.tolist():
        cleaned[column] = cleaned[column].astype(str).str.strip().str.lower()

    for column in ["age", "bmi", "children", "charges"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna().drop_duplicates().reset_index(drop=True)
    return cleaned


def fit_encoders(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "sex": LabelEncoder().fit(df["sex"]),
        "smoker": LabelEncoder().fit(df["smoker"]),
        "region": make_one_hot_encoder().fit(df[["region"]]),
    }


def build_treated_dataframe(df: pd.DataFrame, encoders: Dict[str, Any]) -> pd.DataFrame:
    base = df[["age", "bmi", "children"]].reset_index(drop=True).copy()
    base["sex"] = encoders["sex"].transform(df["sex"])
    base["smoker"] = encoders["smoker"].transform(df["smoker"])

    region_encoded = encoders["region"].transform(df[["region"]])
    region_columns = encoders["region"].get_feature_names_out(["region"])
    region_df = pd.DataFrame(region_encoded, columns=region_columns, index=df.index)

    parts = [base.reset_index(drop=True), region_df.reset_index(drop=True)]
    if "charges" in df.columns:
        parts.append(df[["charges"]].reset_index(drop=True))

    return pd.concat(parts, axis=1)


def prepare_features(treated_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return treated_df.drop(columns=["charges"]).copy(), treated_df["charges"].copy()


def fit_scaler(features: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def scale_features(scaler: StandardScaler, features: pd.DataFrame) -> np.ndarray:
    return scaler.transform(features)


def compute_k_search(scaled_features: np.ndarray, max_k: int = 8) -> pd.DataFrame:
    rows = []
    upper = min(max_k, len(scaled_features) - 1)
    for k in range(2, max(3, upper + 1)):
        if k >= len(scaled_features):
            continue
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled_features)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled_features, labels)),
            }
        )
    return pd.DataFrame(rows)


def choose_best_k(search_df: pd.DataFrame) -> int:
    if search_df.empty:
        return 2
    best_row = search_df.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(best_row["k"])


def plot_cluster_search(search_df: pd.DataFrame) -> None:
    if search_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(search_df["k"], search_df["inertia"], marker="o", linewidth=2)
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    fig.tight_layout()
    fig.savefig(ELBOW_PATH, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(search_df["k"], search_df["silhouette"], marker="o", linewidth=2, color="#c44e52")
    ax.set_title("Silhouette Score")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette")
    fig.tight_layout()
    fig.savefig(SILHOUETTE_PATH, dpi=160)
    plt.close(fig)


def fit_kmeans(scaled_features: np.ndarray, n_clusters: int) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    model.fit(scaled_features)
    return model


def plot_pca_clusters(scaled_features: np.ndarray, labels: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(scaled_features)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=45, alpha=0.85)
    ax.set_title("Clusters in 2D via PCA")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(PCA_PATH, dpi=160)
    plt.close(fig)


def plot_interpretable_clusters(treated_df: pd.DataFrame, labels: np.ndarray) -> None:
    labeled = treated_df.copy()
    labeled["cluster"] = labels

    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster in sorted(labeled["cluster"].unique()):
        subset = labeled[labeled["cluster"] == cluster]
        ax.scatter(subset["age"], subset["charges"], s=30, alpha=0.75, label=f"Cluster {cluster}")
    ax.set_title("Age x Charges by cluster")
    ax.set_xlabel("Age")
    ax.set_ylabel("Charges")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(CLUSTER_AGE_CHARGES_PATH, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster in sorted(labeled["cluster"].unique()):
        subset = labeled[labeled["cluster"] == cluster]
        ax.scatter(subset["bmi"], subset["charges"], s=30, alpha=0.75, label=f"Cluster {cluster}")
    ax.set_title("BMI x Charges by cluster")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Charges")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(CLUSTER_BMI_CHARGES_PATH, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    order = sorted(labeled["cluster"].unique())
    ax.boxplot(
        [labeled.loc[labeled["cluster"] == cluster, "charges"].values for cluster in order],
        labels=[f"C{cluster}" for cluster in order],
    )
    ax.set_title("Charges distribution by cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Charges")
    fig.tight_layout()
    fig.savefig(CLUSTER_BOXPLOT_PATH, dpi=160)
    plt.close(fig)


def create_regressor(model_name: str, n_samples: int | None = None):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "knn":
        neighbors = 5 if n_samples is None else max(1, min(5, n_samples))
        return KNeighborsRegressor(n_neighbors=neighbors)
    if model_name == "tree":
        return DecisionTreeRegressor(max_depth=5, random_state=42)
    raise ValueError(f"Unknown model: {model_name}")


def fit_regressor(model_name: str, X: np.ndarray, y: pd.Series):
    model = create_regressor(model_name, len(y))
    model.fit(X, y)
    return model


def predict_cluster_model(global_model, cluster_models: Dict[int, Any], clusters: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
    predictions = np.zeros(len(X_scaled), dtype=float)
    for index, cluster in enumerate(clusters):
        model = cluster_models.get(int(cluster), global_model)
        predictions[index] = float(model.predict(X_scaled[index].reshape(1, -1))[0])
    return predictions


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def regression_improvement(global_rmse: float, cluster_rmse: float) -> float:
    if global_rmse == 0:
        return 0.0
    return float(((global_rmse - cluster_rmse) / global_rmse) * 100)


def build_cluster_models(
    model_name: str,
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    train_clusters: np.ndarray,
    fallback_model,
) -> Dict[int, Any]:
    cluster_models: Dict[int, Any] = {}
    for cluster in sorted(np.unique(train_clusters)):
        mask = train_clusters == cluster
        if mask.sum() < 2 and model_name == "linear":
            cluster_models[int(cluster)] = fallback_model
            continue
        cluster_models[int(cluster)] = fit_regressor(model_name, X_train_scaled[mask], y_train.iloc[mask])
    return cluster_models


def evaluate_regressor_strategy(
    model_name: str,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    train_clusters: np.ndarray,
    test_clusters: np.ndarray,
) -> Dict[str, Any]:
    global_model = fit_regressor(model_name, X_train_scaled, y_train)
    global_predictions = global_model.predict(X_test_scaled)
    global_metrics = regression_metrics(y_test, global_predictions)

    cluster_models = build_cluster_models(model_name, X_train_scaled, y_train, train_clusters, global_model)
    cluster_predictions = predict_cluster_model(global_model, cluster_models, test_clusters, X_test_scaled)
    cluster_metrics = regression_metrics(y_test, cluster_predictions)

    return {
        "model_name": model_name,
        "global_model": global_model,
        "cluster_models": cluster_models,
        "global_metrics": global_metrics,
        "cluster_metrics": cluster_metrics,
        "improvement_pct": regression_improvement(global_metrics["RMSE"], cluster_metrics["RMSE"]),
    }


def build_model_tables(strategy_results: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global_table = pd.DataFrame(
        [
            {"model": result["model_name"], **result["global_metrics"], "improvement_pct": result["improvement_pct"]}
            for result in strategy_results.values()
        ]
    ).sort_values("RMSE")

    cluster_table = pd.DataFrame(
        [
            {"model": result["model_name"], **result["cluster_metrics"], "improvement_pct": result["improvement_pct"]}
            for result in strategy_results.values()
        ]
    ).sort_values("RMSE")

    return global_table, cluster_table


def choose_best_results(strategy_results: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], str, Dict[str, Any]]:
    best_global_result = min(strategy_results.values(), key=lambda item: item["global_metrics"]["RMSE"])
    best_cluster_result = min(strategy_results.values(), key=lambda item: item["cluster_metrics"]["RMSE"])
    winning_scope = "clustered" if best_cluster_result["cluster_metrics"]["RMSE"] < best_global_result["global_metrics"]["RMSE"] else "global"
    winning_result = best_cluster_result if winning_scope == "clustered" else best_global_result
    return best_global_result, best_cluster_result, winning_scope, winning_result


def compute_cluster_profiles(treated_df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    profile_df = treated_df.copy()
    profile_df["cluster"] = cluster_labels
    summary = profile_df.groupby("cluster").agg(
        count=("charges", "size"),
        mean_age=("age", "mean"),
        mean_bmi=("bmi", "mean"),
        mean_children=("children", "mean"),
        smoker_rate=("smoker", "mean"),
        mean_charges=("charges", "mean"),
    ).reset_index()
    summary.to_csv(CLUSTER_PROFILE_PATH, index=False)
    return summary


def interpret_clusters(profile_df: pd.DataFrame, overall: Dict[str, float]) -> str:
    lines = ["Interpretacao automatica dos clusters", ""]
    for _, row in profile_df.iterrows():
        notes = []
        notes.append("mais fumantes que a media" if row["smoker_rate"] > overall["smoker_rate"] else "menos fumantes que a media")
        notes.append("perfil mais jovem" if row["mean_age"] < overall["mean_age"] else "perfil mais maduro")

        high_cost_threshold = overall["mean_charges"] * 1.2
        low_cost_threshold = overall["mean_charges"] * 0.8
        if row["mean_charges"] >= high_cost_threshold:
            notes.append("custos acima da media")
        elif row["mean_charges"] <= low_cost_threshold:
            notes.append("custos abaixo da media")
        else:
            notes.append("custos proximos da media")

        if row["mean_bmi"] > overall["mean_bmi"]:
            notes.append("IMC acima da media")

        lines.append(
            f"Cluster {int(row['cluster'])}: {', '.join(notes)}. "
            f"Media de charges = {row['mean_charges']:.2f}, fumantes = {row['smoker_rate']:.2f}."
        )

    text = "\n".join(lines)
    CLUSTER_INTERPRETATION_PATH.write_text(text, encoding="utf-8")
    return text


def transform_new_sample(sample: pd.DataFrame, encoders: Dict[str, Any], treated_columns: list[str]) -> pd.DataFrame:
    sample_clean = clean_dataset(sample)
    treated = build_treated_dataframe(sample_clean, encoders)
    feature_frame = treated.drop(columns=["charges"], errors="ignore")
    return feature_frame.reindex(columns=treated_columns, fill_value=0)


def prompt_user_sample() -> pd.DataFrame:
    print("\nPreencha os dados para previsao final:")
    sample = {
        "age": int(input("age: ")),
        "sex": input("sex (male/female): ").strip().lower(),
        "bmi": float(input("bmi: ")),
        "children": int(input("children: ")),
        "smoker": input("smoker (yes/no): ").strip().lower(),
        "region": input("region (northeast/northwest/southeast/southwest): ").strip().lower(),
    }
    return pd.DataFrame([sample])


def train_pipeline(dataset_path: Path = RAW_DATASET) -> Dict[str, Any]:
    ensure_artifacts_dir()

    raw_df = pd.read_csv(dataset_path)
    cleaned_df = clean_dataset(raw_df)
    encoders = fit_encoders(cleaned_df)
    treated_df = build_treated_dataframe(cleaned_df, encoders)
    treated_df.to_csv(TREATED_DATASET_PATH, index=False)

    features, target = prepare_features(treated_df)
    scaler = fit_scaler(features)
    full_scaled = scale_features(scaler, features)

    search_df = compute_k_search(full_scaled, max_k=8)
    plot_cluster_search(search_df)
    best_k = choose_best_k(search_df)

    kmeans = fit_kmeans(full_scaled, best_k)
    cluster_labels = kmeans.labels_

    X_train_scaled, X_test_scaled, y_train, y_test, train_clusters, test_clusters = train_test_split(
        full_scaled,
        target,
        cluster_labels,
        test_size=0.2,
        random_state=42,
        stratify=cluster_labels,
    )

    strategy_results = {
        model_name: evaluate_regressor_strategy(
            model_name,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            train_clusters,
            test_clusters,
        )
        for model_name in MODEL_NAMES
    }

    global_table, cluster_table = build_model_tables(strategy_results)
    best_global_result, best_cluster_result, winning_scope, winning_result = choose_best_results(strategy_results)

    plot_pca_clusters(full_scaled, cluster_labels)
    plot_interpretable_clusters(treated_df, cluster_labels)

    profile_df = compute_cluster_profiles(treated_df, cluster_labels)
    interpretation_text = interpret_clusters(
        profile_df,
        {
            "mean_age": float(treated_df["age"].mean()),
            "mean_bmi": float(treated_df["bmi"].mean()),
            "mean_charges": float(treated_df["charges"].mean()),
            "smoker_rate": float(treated_df["smoker"].mean()),
        },
    )

    final_bundle = {
        "kmeans": kmeans,
        "global_model": winning_result["global_model"],
        "cluster_models": winning_result["cluster_models"],
        "best_k": best_k,
        "best_model_name": winning_result["model_name"],
        "strategy": winning_scope,
    }

    joblib.dump(
        {
            "encoders": encoders,
            "scaler": scaler,
            "feature_columns": features.columns.tolist(),
            "bundle": final_bundle,
        },
        BUNDLE_PATH,
    )

    metrics = {
        "best_k": best_k,
        "cluster_search": search_df.to_dict(orient="records"),
        "global_models": global_table.to_dict(orient="records"),
        "clustered_models": cluster_table.to_dict(orient="records"),
        "best_global_model": best_global_result["model_name"],
        "best_clustered_model": best_cluster_result["model_name"],
        "comparison": {
            "winner": winning_scope,
            "winner_model": winning_result["model_name"],
            "global_rmse": float(best_global_result["global_metrics"]["RMSE"]),
            "cluster_rmse": float(best_cluster_result["cluster_metrics"]["RMSE"]),
            "rmse_difference": float(best_global_result["global_metrics"]["RMSE"] - best_cluster_result["cluster_metrics"]["RMSE"]),
            "improvement_pct": regression_improvement(
                best_global_result["global_metrics"]["RMSE"],
                best_cluster_result["cluster_metrics"]["RMSE"],
            ),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    search_df.to_csv(ARTIFACTS_DIR / "cluster_search.csv", index=False)

    return {
        "raw_df": raw_df,
        "cleaned_df": cleaned_df,
        "treated_df": treated_df,
        "search_df": search_df,
        "best_k": best_k,
        "global_table": global_table,
        "cluster_table": cluster_table,
        "winning_scope": winning_scope,
        "winning_model_name": winning_result["model_name"],
        "profile_df": profile_df,
        "interpretation_text": interpretation_text,
        "bundle": final_bundle,
        "encoders": encoders,
        "scaler": scaler,
    }


def predict_sample(sample_df: pd.DataFrame, artifacts: Dict[str, Any]) -> Tuple[int, float]:
    encoders = artifacts["encoders"]
    bundle = artifacts["bundle"]
    scaler = artifacts["scaler"]
    feature_columns = artifacts["treated_df"].drop(columns=["charges"]).columns.tolist()

    sample_features = transform_new_sample(sample_df, encoders, feature_columns)
    sample_scaled = scale_features(scaler, sample_features)

    cluster_id = int(bundle["kmeans"].predict(sample_scaled)[0])
    if bundle.get("strategy") == "clustered":
        model = bundle["cluster_models"].get(cluster_id, bundle["global_model"])
    else:
        model = bundle["global_model"]
    predicted_cost = float(model.predict(sample_scaled)[0])
    return cluster_id, predicted_cost


def print_report(results: Dict[str, Any]) -> None:
    print("\n=== Resumo do projeto ===")
    print(f"Melhor k encontrado: {results['best_k']}")

    print("\nMelhores modelos globais:")
    print(results["global_table"].to_string(index=False))

    print("\nMelhores modelos por cluster:")
    print(results["cluster_table"].to_string(index=False))

    print(f"\nMelhor abordagem: regressao {results['winning_scope']} ({results['winning_model_name']})")
    print("\nInterpretacao dos clusters:")
    print(results["interpretation_text"])


def main() -> None:
    results = train_pipeline()
    print_report(results)

    try:
        if input("\nDeseja fazer uma previsao com novos dados? [s/n]: ").strip().lower().startswith("s"):
            sample_df = prompt_user_sample()
            cluster_id, predicted_cost = predict_sample(sample_df, results)
            print(f"\nCluster identificado: {cluster_id}")
            print(f"Custo previsto: US$ {predicted_cost:,.2f}")
    except EOFError:
        pass


if __name__ == "__main__":
    main()