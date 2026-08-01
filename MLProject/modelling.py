"""
modelling.py — Entry point MLflow Project untuk pelatihan ulang otomatis.

Berkas ini adalah penyesuaian dari `modelling_tuning.py` pada Kriteria 2 agar
dapat dijalankan sebagai **MLflow Project** di dalam workflow CI. Perbedaannya:

  - Hyperparameter tidak lagi dicari, melainkan diterima sebagai parameter
    entry point. Nilai bawaannya adalah kombinasi terbaik hasil pencarian pada
    Kriteria 2, sehingga pelatihan ulang berjalan cepat dan dapat direproduksi.
  - Pencatatan diarahkan ke penyimpanan berkas lokal (`mlruns/`), bukan ke
    server, karena runner CI tidak menjalankan MLflow Tracking Server.
  - Setelah selesai, `run_id` ditulis ke berkas agar langkah CI berikutnya dapat
    membangun image Docker dari model yang baru saja dihasilkan.

Dijalankan lewat MLflow Project:

    mlflow run MLProject --env-manager=local
    mlflow run MLProject -P n_estimators=500 -P max_depth=16

Atau langsung:

    python modelling.py --n-estimators 746
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

DIR_SKRIP = Path(__file__).resolve().parent
DIR_DATA = DIR_SKRIP / "idx_liquidity_preprocessing"

NAMA_EKSPERIMEN = "IDX Likuiditas - CI Retraining"
BERKAS_RUN_ID = DIR_SKRIP / "run_id.txt"

KOLOM_TARGET = "kelas_likuiditas"
KOLOM_BUKAN_FITUR = ["kode", KOLOM_TARGET, "kelas_kode", "bagian"]

RANDOM_STATE = 42
N_LIPATAN_CV = 5


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def muat_dataset(dir_data: Path) -> dict:
    """Baca data latih, data uji, dan metadata preprocessing."""
    latih = pd.read_csv(dir_data / "idx_liquidity_train.csv")
    uji = pd.read_csv(dir_data / "idx_liquidity_test.csv")
    metadata = json.loads((dir_data / "metadata_preprocessing.json").read_text(encoding="utf-8"))

    fitur = [k for k in latih.columns if k not in KOLOM_BUKAN_FITUR]
    return {
        "X_train": latih[fitur], "y_train": latih[KOLOM_TARGET],
        "X_test": uji[fitur], "y_test": uji[KOLOM_TARGET],
        "uji_mentah": uji, "fitur": fitur, "metadata": metadata,
    }


def prediksi_naif(uji: pd.DataFrame, metadata: dict) -> pd.Series:
    """Tolok ukur naif: salin kelas likuiditas tahun sebelumnya.

    Nilai `advt_median` dipulihkan dengan membalik standarisasi lalu log1p,
    memakai parameter yang tersimpan pada metadata preprocessing.
    """
    par = metadata["parameter_dipelajari_dari_data_latih"]
    rupiah = np.expm1(
        uji["advt_median"].to_numpy() * par["scaler_scale"]["advt_median"]
        + par["scaler_mean"]["advt_median"]
    )
    batas_likuid = metadata["ambang_kelas_rupiah"]["likuid"]
    batas_menengah = metadata["ambang_kelas_rupiah"]["menengah"]

    def ke_kelas(v: float) -> str:
        if v >= batas_likuid:
            return "Likuid"
        if v >= batas_menengah:
            return "Menengah"
        return "Tidak Likuid"

    return pd.Series([ke_kelas(v) for v in rupiah], index=uji.index)


# --------------------------------------------------------------------------- #
# Artefak
# --------------------------------------------------------------------------- #

def tulis_artefak(dir_art: Path, y_uji, pred, peluang, kelas, y_naif, y_latih,
                  model, fitur) -> dict[str, float]:
    """Tulis artefak evaluasi dan kembalikan metrik ringkasnya."""
    # -- Perbandingan terhadap tolok ukur naif ----------------------------- #
    mayoritas = y_latih.value_counts().idxmax()
    baris = []
    for nama, yp in [
        ("Tebak kelas mayoritas", pd.Series([mayoritas] * len(y_uji), index=y_uji.index)),
        ("Salin kelas tahun lalu", y_naif),
        ("Model hasil CI", pd.Series(pred, index=y_uji.index)),
    ]:
        baris.append({
            "strategi": nama,
            "akurasi": float(accuracy_score(y_uji, yp)),
            "f1_makro": float(f1_score(y_uji, yp, average="macro", zero_division=0)),
        })
    pd.DataFrame(baris).to_csv(dir_art / "baseline_perbandingan.csv", index=False)
    (dir_art / "baseline_perbandingan.json").write_text(
        json.dumps(baris, indent=2, ensure_ascii=False), encoding="utf-8")

    # -- Confusion matrix --------------------------------------------------- #
    cm = confusion_matrix(y_uji, pred, labels=kelas)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(kelas)), kelas, fontsize=9)
    ax.set_yticks(range(len(kelas)), kelas, fontsize=9)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya")
    ax.set_title("Confusion matrix — hasil CI", fontweight="bold")
    for i in range(len(kelas)):
        for j in range(len(kelas)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout()
    fig.savefig(dir_art / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # -- Laporan per kelas -------------------------------------------------- #
    laporan = classification_report(y_uji, pred, labels=kelas,
                                    output_dict=True, zero_division=0)
    (dir_art / "laporan_per_kelas.json").write_text(
        json.dumps(laporan, indent=2, ensure_ascii=False), encoding="utf-8")

    # -- Feature importance bawaan model ------------------------------------ #
    penting = pd.DataFrame({
        "fitur": fitur,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    penting.to_csv(dir_art / "feature_importance.csv", index=False)

    return {
        "baseline_naif_f1_macro": baris[1]["f1_makro"],
        "baseline_naif_accuracy": baris[1]["akurasi"],
        "baseline_mayoritas_accuracy": baris[0]["akurasi"],
        "selisih_f1_makro_vs_naif": baris[2]["f1_makro"] - baris[1]["f1_makro"],
    }


# --------------------------------------------------------------------------- #
# Program utama
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Nilai bawaan = kombinasi terbaik hasil RandomizedSearchCV pada Kriteria 2
    p.add_argument("--n-estimators", type=int, default=746)
    p.add_argument("--max-depth", type=int, default=32)
    p.add_argument("--min-samples-split", type=int, default=13)
    p.add_argument("--min-samples-leaf", type=int, default=3)
    p.add_argument("--max-features", type=str, default="sqrt")
    p.add_argument("--bootstrap", type=str, default="False")
    p.add_argument("--dir-data", type=Path, default=DIR_DATA)
    p.add_argument("--experiment", type=str, default=NAMA_EKSPERIMEN)
    args = p.parse_args()

    bootstrap = str(args.bootstrap).strip().lower() in {"true", "1", "yes"}
    max_depth = None if args.max_depth in (0, -1) else args.max_depth
    max_features = None if args.max_features.lower() == "none" else args.max_features

    print("=" * 70)
    print("PELATIHAN ULANG OTOMATIS — KLASIFIKASI LIKUIDITAS SAHAM IDX")
    print("=" * 70)

    data = muat_dataset(args.dir_data)
    kelas = data["metadata"]["urutan_kelas"]
    print(f"\nData latih : {data['X_train'].shape[0]} emiten x {data['X_train'].shape[1]} fitur")
    print(f"Data uji   : {data['X_test'].shape[0]} emiten")

    # Bila MLFLOW_TRACKING_URI tidak disetel, MLflow memakai ./mlruns relatif
    # terhadap direktori kerja — perilaku yang memang diinginkan di CI.
    uri = mlflow.get_tracking_uri()
    print(f"\nTracking URI : {uri}")

    # `mlflow run` sudah membuat run sendiri dan mengeksposnya lewat
    # MLFLOW_RUN_ID. Dalam keadaan itu skrip harus MELANJUTKAN run tersebut,
    # bukan membuat yang baru — memanggil set_experiment() atau start_run()
    # dengan run baru akan ditolak MLflow karena bentrok dengan run aktif.
    # Nama eksperimen saat dijalankan lewat MLflow Project ditentukan oleh
    # opsi --experiment-name pada perintah `mlflow run`.
    run_id_project = os.environ.get("MLFLOW_RUN_ID")

    if run_id_project:
        print(f"Mode         : MLflow Project (melanjutkan run {run_id_project[:8]}…)")
        konteks_run = mlflow.start_run(run_id=run_id_project)
    else:
        mlflow.set_experiment(args.experiment)
        print(f"Mode         : langsung")
        print(f"Eksperimen   : {args.experiment}")
        konteks_run = mlflow.start_run(run_name="ci-retraining")

    with konteks_run as run:
        if run_id_project:
            mlflow.set_tag("mlflow.runName", "ci-retraining")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        print("\nMelatih model ...")
        model.fit(data["X_train"], data["y_train"])

        pred = model.predict(data["X_test"])
        peluang = model.predict_proba(data["X_test"])
        kelas_model = list(model.classes_)

        skor_cv = cross_val_score(
            model, data["X_train"], data["y_train"],
            cv=StratifiedKFold(n_splits=N_LIPATAN_CV, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1_macro", n_jobs=-1,
        )

        metrik = {
            "test_accuracy_score": float(accuracy_score(data["y_test"], pred)),
            "test_f1_macro": float(f1_score(data["y_test"], pred, average="macro", zero_division=0)),
            "test_f1_score": float(f1_score(data["y_test"], pred, average="weighted", zero_division=0)),
            "test_precision_macro": float(precision_score(data["y_test"], pred, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(data["y_test"], pred, average="macro", zero_division=0)),
            "test_log_loss": float(log_loss(data["y_test"], peluang, labels=kelas_model)),
            "test_roc_auc": float(roc_auc_score(data["y_test"], peluang, multi_class="ovr",
                                                average="macro", labels=kelas_model)),
            "cv_f1_macro_mean": float(skor_cv.mean()),
            "cv_f1_macro_std": float(skor_cv.std()),
        }
        for nama_kelas, nilai in zip(
            kelas_model, f1_score(data["y_test"], pred, average=None, labels=kelas_model, zero_division=0)
        ):
            metrik[f"test_f1_{nama_kelas.replace(' ', '_').lower()}"] = float(nilai)

        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "jumlah_fitur": data["X_train"].shape[1],
            "jumlah_data_latih": data["X_train"].shape[0],
            "jumlah_data_uji": data["X_test"].shape[0],
        })

        with tempfile.TemporaryDirectory() as tmp:
            dir_art = Path(tmp)
            metrik.update(tulis_artefak(
                dir_art, data["y_test"], pred, peluang, kelas,
                prediksi_naif(data["uji_mentah"], data["metadata"]),
                data["y_train"], model, data["fitur"],
            ))
            mlflow.log_artifacts(str(dir_art), artifact_path="evaluasi")

        mlflow.log_metrics(metrik)

        # Signature dan input example diperlukan agar image Docker hasil
        # `mlflow models build-docker` dapat memvalidasi permintaan inferensi.
        signature = infer_signature(data["X_train"], model.predict(data["X_train"]))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=data["X_train"].head(3),
        )

        mlflow.set_tags({
            "kriteria": "3 - Workflow CI",
            "dipicu_oleh": os.environ.get("GITHUB_EVENT_NAME", "manual"),
            "commit": os.environ.get("GITHUB_SHA", "-")[:7],
            "nomor_run_ci": os.environ.get("GITHUB_RUN_NUMBER", "-"),
        })

        # run_id ditulis ke berkas agar langkah CI berikutnya dapat merujuk
        # model ini saat membangun image Docker.
        BERKAS_RUN_ID.write_text(run.info.run_id, encoding="utf-8")

        print("\n" + "-" * 70)
        print("HASIL")
        print("-" * 70)
        print(f"F1 makro (uji)      : {metrik['test_f1_macro']:.4f}")
        print(f"Akurasi (uji)       : {metrik['test_accuracy_score']:.4f}")
        print(f"F1 makro (CV)       : {metrik['cv_f1_macro_mean']:.4f} ± {metrik['cv_f1_macro_std']:.4f}")
        print(f"Baseline naif (F1)  : {metrik['baseline_naif_f1_macro']:.4f}")
        print(f"Selisih vs naif     : {metrik['selisih_f1_makro_vs_naif']:+.4f}")
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Ditulis ke: {BERKAS_RUN_ID}")

        if metrik["selisih_f1_makro_vs_naif"] <= 0:
            print("\nPERINGATAN: model tidak melampaui tolok ukur naif.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
