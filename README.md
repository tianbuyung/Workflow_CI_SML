# Workflow_CI_SML — Pelatihan Ulang Otomatis Model Likuiditas Saham IDX

Repositori **Kriteria 3** submission kelas *Membangun Sistem Machine Learning* (Dicoding).

Melatih ulang model klasifikasi likuiditas emiten BEI secara otomatis lewat
**MLflow Project** yang dijalankan oleh **GitHub Actions**, lalu membungkus
hasilnya menjadi image Docker dan mengirimkannya ke Docker Hub.

| Sumber | Tautan |
|---|---|
| Dataset & preprocessing (Kriteria 1) | [Eksperimen_SML_Septian-Maulana](https://github.com/tianbuyung/Eksperimen_SML_Septian-Maulana) |
| Eksperimen & tuning (Kriteria 2) | [DagsHub — SMSML_Septian-Maulana](https://dagshub.com/tiandev/SMSML_Septian-Maulana) |
| Image hasil CI | [Docker Hub — tiandev/idx-liquidity-model](https://hub.docker.com/r/tiandev/idx-liquidity-model) |

---

## Struktur Repositori

```text
Workflow_CI_SML
├── .github/workflows/
│   └── ci.yml                              # workflow CI
└── MLProject/
    ├── MLProject                           # definisi MLflow Project
    ├── conda.yaml                          # environment
    ├── modelling.py                        # entry point pelatihan
    ├── mirror_to_dagshub.py                # penyalin run ke DagsHub
    ├── DockerHub.txt                       # tautan image Docker Hub
    ├── run_id.txt                          # run_id hasil pelatihan terakhir
    ├── idx_liquidity_preprocessing/        # dataset siap latih
    └── mlruns/                             # artefak hasil CI (di-commit)
```

---

## Cara Menjalankan

### Lokal

```bash
export MLFLOW_TRACKING_URI="file://$(pwd)/MLProject/mlruns"

mlflow run MLProject \
  --env-manager=local \
  --experiment-name "IDX Likuiditas - CI Retraining"
```

Mengubah hyperparameter:

```bash
mlflow run MLProject --env-manager=local \
  -P n_estimators=500 -P max_depth=16
```

Dengan environment conda terisolasi:

```bash
mlflow run MLProject --env-manager=conda
```

### Membangun image Docker

```bash
mlflow models build-docker \
  --model-uri "runs:/$(cat MLProject/run_id.txt)/model" \
  --name tiandev/idx-liquidity-model \
  --enable-mlserver

docker run -d --rm -p 5005:8080 tiandev/idx-liquidity-model:latest
```

---

## Alur Workflow CI

Dipicu oleh `push` ke `main`, `workflow_dispatch` (dengan hyperparameter yang bisa
diubah), dan jadwal mingguan setiap Senin 01:00 UTC.

```text
Checkout
  → Setup Python 3.12.7 + dependencies
  → Tetapkan tracking URI berbasis berkas (mlruns/)
  → mlflow run MLProject
  → Ambil run_id
  → Validasi mutu model            ← gerbang mutu
  → Unggah artifact GitHub
  → Salin run ke DagsHub           ← tidak memblokir
  → Commit artefak ke repository
  → Login Docker Hub
  → mlflow models build-docker
  → Push image (:latest dan :<nomor-run>)
  → Ringkasan
```

### Gerbang mutu

Sebelum artefak disimpan dan image dibangun, workflow memeriksa hasil pelatihan:

| Pemeriksaan | Bila gagal |
|---|---|
| Artefak `model/model.pkl` ada | **CI gagal** |
| F1 makro ≥ 0,50 | **CI gagal** |
| Model melampaui tolok ukur naif | Peringatan, CI tetap lanjut |

Ambang keras dipisahkan dari peringatan lunak dengan sengaja. Model yang jauh di
bawah 0,50 pasti bermasalah dan tidak boleh sampai ter-publish. Sedangkan model
yang sedikit di bawah tolok ukur naif belum tentu rusak — bisa jadi kondisi pasar
memang sedang berubah, dan itu justru informasi yang perlu dilihat manusia,
bukan alasan menghentikan pipeline.

### Penyalinan ke DagsHub tidak memblokir

Langkah penyalinan memakai `continue-on-error: true`, dan `mirror_to_dagshub.py`
sendiri selalu keluar dengan kode 0 meski gagal. Alasannya: syarat kriteria ini
adalah **tersimpannya artefak di repositori**, bukan di DagsHub. Gangguan pada
layanan pihak ketiga tidak boleh membatalkan pemenuhan syarat tersebut.

Penyalinan dipilih alih-alih melatih ulang ke DagsHub, agar catatan di DagsHub
dijamin identik dengan artefak yang di-commit — bukan hasil pelatihan kedua yang
bisa berbeda.

---

## Secret yang Diperlukan

Disimpan di **Settings → Secrets and variables → Actions**.

| Secret | Wajib | Kegunaan |
|---|---|---|
| `DOCKERHUB_USERNAME` | ya | login Docker Hub |
| `DOCKERHUB_TOKEN` | ya | access token Docker Hub |
| `DAGSHUB_TOKEN` | tidak | menyalin run ke DagsHub; bila kosong, langkah dilewati tanpa menggagalkan CI |

Pastikan juga **Settings → Actions → General → Workflow permissions** disetel ke
**Read and write permissions**, agar langkah commit artefak dapat menulis ke repositori.

---

## Pemenuhan Kriteria

| Tingkat | Ketentuan | Pemenuhan |
|---|---|---|
| **Basic** | Folder `MLProject` + workflow CI yang menghasilkan model saat trigger terpantik | `MLProject/` dengan `MLProject`, `conda.yaml`, `modelling.py`; workflow `ci.yml` |
| **Skilled** | Menyimpan artefak ke suatu repositori | Artefak di-commit ke repo ini **dan** diunggah sebagai artifact GitHub |
| **Advance** | Membuat Docker Image ke Docker Hub memakai `mlflow build-docker` | Langkah *Bangun image* dan *Push* pada `ci.yml` |

---

## Model yang Dihasilkan

| Aspek | Nilai |
|---|---|
| Algoritma | `RandomForestClassifier` (scikit-learn 1.5.2) |
| Hyperparameter | hasil `RandomizedSearchCV` pada Kriteria 2, dijadikan nilai bawaan entry point |
| Fitur | 28 (16 numerik + 12 one-hot sektor) |
| Data | 710 latih / 178 uji |
| F1 makro (uji) | 0,7118 |
| Akurasi (uji) | 0,7247 |
| Tolok ukur naif | 0,6590 F1 makro — dilampaui **+0,0528** |

Metrik utama adalah **F1 makro**, bukan akurasi, karena kelas sangat timpang
(51,5% / 30,0% / 18,5%). Tolok ukur pembanding bukan tebakan acak melainkan
strategi *"salin kelas likuiditas tahun lalu"* — lawan yang jauh lebih sulit dan
jauh lebih jujur.
