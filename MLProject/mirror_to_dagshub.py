"""
mirror_to_dagshub.py — Salin run MLflow lokal ke DagsHub.

Dipakai pada workflow CI agar riwayat pelatihan ulang otomatis ikut tercatat di
dashboard DagsHub bersama eksperimen manual dari Kriteria 2.

Pendekatannya **menyalin**, bukan melatih ulang. Model dilatih sekali saja lewat
MLflow Project, lalu run yang sudah jadi itu direplikasi ke DagsHub beserta
parameter, metrik, tag, dan seluruh artefaknya. Dengan begitu catatan di DagsHub
dijamin identik dengan artefak yang di-commit ke repositori — bukan hasil
pelatihan kedua yang bisa saja berbeda.

Skrip ini sengaja dirancang **tidak pernah menggagalkan CI**:

  - kredensial tidak lengkap  -> keluar dengan kode 0 dan pesan lewati
  - DagsHub tidak dapat dihubungi -> keluar dengan kode 0 dan peringatan

Kegagalan menyalin ke DagsHub tidak boleh membatalkan pemenuhan kriteria yang
sesungguhnya, yaitu tersimpannya artefak di repositori GitHub.

Pemakaian:

    python mirror_to_dagshub.py --run-id <id> --local-uri file:///path/mlruns
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

NAMA_EKSPERIMEN = "IDX Likuiditas - CI Retraining"

# Tag internal MLflow yang tidak boleh disalin mentah-mentah karena akan
# ditimpa atau tidak relevan pada run tujuan.
TAG_DILEWATI = {
    "mlflow.user",
    "mlflow.source.name",
    "mlflow.source.type",
    "mlflow.source.git.commit",
    "mlflow.project.entryPoint",
    "mlflow.project.backend",
    "mlflow.log-model.history",
}


def lewati(pesan: str) -> int:
    """Cetak alasan dan keluar dengan status berhasil agar CI tetap lanjut."""
    print(f"[LEWATI] {pesan}")
    print("Penyalinan ke DagsHub dilewati. Artefak tetap tersimpan di repositori.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True, help="run_id pada penyimpanan lokal")
    p.add_argument("--local-uri", required=True, help="tracking URI lokal (file://...)")
    p.add_argument("--experiment", default=NAMA_EKSPERIMEN)
    args = p.parse_args()

    repo = os.environ.get("DAGSHUB_REPO", "").strip()
    pengguna = os.environ.get("MLFLOW_TRACKING_USERNAME", "").strip()
    token = os.environ.get("MLFLOW_TRACKING_PASSWORD", "").strip()

    if not (repo and pengguna and token):
        kurang = [n for n, v in [("DAGSHUB_REPO", repo),
                                 ("MLFLOW_TRACKING_USERNAME", pengguna),
                                 ("MLFLOW_TRACKING_PASSWORD", token)] if not v]
        return lewati("kredensial DagsHub belum lengkap: " + ", ".join(kurang))

    uri_dagshub = f"https://dagshub.com/{repo}.mlflow"
    print(f"Sumber  : {args.local_uri}")
    print(f"Tujuan  : {uri_dagshub}")
    print(f"Run ID  : {args.run_id}")

    # -- Baca run dari penyimpanan lokal ----------------------------------- #
    try:
        sumber = MlflowClient(tracking_uri=args.local_uri)
        run = sumber.get_run(args.run_id)
    except Exception as exc:  # noqa: BLE001
        return lewati(f"run lokal tidak dapat dibaca: {exc}")

    print(f"\nParameter : {len(run.data.params)}")
    print(f"Metrik    : {len(run.data.metrics)}")

    # -- Unduh artefak ke direktori sementara ------------------------------ #
    with tempfile.TemporaryDirectory() as tmp:
        try:
            dir_artefak = Path(sumber.download_artifacts(args.run_id, "", tmp))
        except Exception as exc:  # noqa: BLE001
            return lewati(f"artefak lokal tidak dapat diunduh: {exc}")

        berkas = [f for f in dir_artefak.rglob("*") if f.is_file()]
        print(f"Artefak   : {len(berkas)} berkas "
              f"({sum(f.stat().st_size for f in berkas) / 1e6:.1f} MB)")

        # -- Tulis ke DagsHub ---------------------------------------------- #
        try:
            mlflow.set_tracking_uri(uri_dagshub)
            mlflow.set_experiment(args.experiment)

            tag = {k: v for k, v in run.data.tags.items() if k not in TAG_DILEWATI}
            tag["disalin_dari_run_lokal"] = args.run_id
            tag["sumber_penyalinan"] = "GitHub Actions CI"

            with mlflow.start_run(run_name="ci-retraining") as baru:
                mlflow.log_params(run.data.params)
                mlflow.log_metrics(run.data.metrics)
                mlflow.set_tags(tag)
                mlflow.log_artifacts(str(dir_artefak))

                print(f"\nBerhasil disalin.")
                print(f"  Run ID DagsHub : {baru.info.run_id}")
                print(f"  Tautan         : {uri_dagshub}/#/experiments/"
                      f"{baru.info.experiment_id}/runs/{baru.info.run_id}")
        except Exception as exc:  # noqa: BLE001
            return lewati(f"gagal menulis ke DagsHub: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
