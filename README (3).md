# 🍃 Leaf Disease Identification using SVM + Handcrafted Features

Klasifikasi penyakit daun (3 kelas) menggunakan pipeline **Computer Vision klasik**: segmentasi area sakit pada daun, ekstraksi fitur tangan (handcrafted features), dan klasifikasi dengan **Support Vector Machine (SVM)** — tanpa deep learning.

> Proyek ini dikerjakan sebagai tugas mata kuliah **Pengolahan Citra Digital**, dengan fokus pada pemahaman pipeline image processing klasik dari nol: preprocessing → segmentasi → ekstraksi fitur → klasifikasi.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-CV-green?logo=opencv)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Colab](https://img.shields.io/badge/Made%20in-Google%20Colab-yellow?logo=googlecolab)](https://colab.research.google.com/)

---

## Hasil Akhir

| Metrik | Skor |
|---|---|
| **Test Accuracy** | **96.67%** |
| Test Precision (weighted) | 96.75% |
| Test Recall (weighted) | 96.67% |
| Test F1-Score (weighted) | 96.66% |
| Mean CV Accuracy (5-Fold) | 96.91% ± 0.64% |
| Kernel SVM terbaik | RBF (C=10) |

**Classification Report (Test Set, n=751):**

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| mildew | 0.95 | 0.99 | 0.97 | 360 |
| rose_P01 | 0.98 | 0.96 | 0.97 | 186 |
| rose_R02 | 0.99 | 0.93 | 0.96 | 205 |

---

##  Pipeline

```
Citra Daun (RGB)
      │
      ▼
1. Preprocessing       → Resize 640×640, CLAHE (kontras), Gaussian Blur
      │
      ▼
2. Segmentasi          → Otsu Thresholding + HSV Color Masking → Morphological Closing
      │
      ▼
3. Ekstraksi Fitur     → 93 fitur total (lihat tabel di bawah)
      │
      ▼
4. Klasifikasi SVM     → Kernel Linear vs RBF, dipilih berdasarkan akurasi validasi
      │
      ▼
   Prediksi Kelas
```

### Ekstraksi Fitur (93 fitur)

| Grup Fitur | Jumlah | Deskripsi | Akurasi (fitur ini saja) |
|---|---|---|---|
| **Color Moments** | 9 | Mean, Variance, Skewness — 3 channel HSV | 91.61% |
| **HSV Histogram** | 48 | Histogram 16-bin per channel H, S, V | 95.34% |
| **GLCM** | 4 | Contrast, Correlation, Energy, Homogeneity (Gray-Level Co-occurrence Matrix) | 80.03% |
| **LBP** | 32 | Local Binary Pattern histogram (uniform, radius=2, points=16) | 82.82% |
| **Total (gabungan)** | **93** | — | **96.67%** |

Kombinasi seluruh fitur terbukti jauh lebih baik dibanding fitur tunggal mana pun — menunjukkan bahwa informasi warna (Color Moments + HSV Histogram) dan tekstur (GLCM + LBP) saling melengkapi.

---

## Dataset

Dataset: **Leaf Disease (NSDSR)** dari [Roboflow Universe — Roboflow 100](https://universe.roboflow.com/roboflow-100/leaf-disease-nsdsr), terdiri dari 3 kelas:

- `mildew` — penyakit embun tepung
- `rose_P01` — penyakit pada daun mawar (tipe P01)
- `rose_R02` — penyakit pada daun mawar (tipe R02)

Dataset asli (format YOLOv8: train/valid/test) digabung kembali lalu **di-split ulang secara stratified** (70% train+val, 30% test; dari 70% diambil 20% untuk validasi) agar distribusi kelas merata di setiap subset — mengatasi ketidakseimbangan split bawaan dataset.

### Contoh Sampel per Kelas

| `mildew` | `rose_P01` | `rose_R02` |
|---|---|---|
| ![mildew](samples/mildew.jpg) | ![rose_P01](samples/rose_P01.jpg) | ![rose_R02](samples/rose_R02.jpg) |
| Embun tepung — bercak putih keabuan menyebar di permukaan daun | Bercak kecoklatan/keunguan dengan tekstur berbintik di permukaan daun | Daun tampak relatif sehat secara visual |

---

## Instalasi & Menjalankan

Notebook ini awalnya dikerjakan di **Google Colab**. Untuk menjalankan secara lokal:

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Lalu buka `PCD_LeafDisease_SVM.ipynb` dengan Jupyter:

```bash
jupyter notebook PCD_LeafDisease_SVM.ipynb
```

> **Catatan:** Notebook menggunakan API key Roboflow untuk mengunduh dataset secara otomatis. Jika ingin menjalankan ulang, ganti dengan API key Roboflow milikmu sendiri di bagian *"1. Mount Google Drive & Download Dataset"*.

Atau langsung buka di Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/<repo-name>/blob/main/PCD_LeafDisease_SVM.ipynb)

---


```

---

## Isi Notebook

1. Install & Import Library
2. Mount Google Drive & Download Dataset (Roboflow)
3. Eksplorasi Dataset (distribusi kelas, visualisasi anotasi)
4. Preprocessing (resize, CLAHE, Gaussian Blur)
5. Segmentasi Area Penyakit (Otsu + HSV Masking)
6. Ekstraksi Fitur (Color Moments, HSV Histogram, GLCM, LBP)
7. Build Dataset — ekstraksi fitur dari seluruh citra
8. Re-Split Stratified (train/val/test)
9. Klasifikasi SVM (Linear vs RBF) + 5-Fold Cross Validation
10. Evaluasi Final pada Test Set (confusion matrix, classification report)
11. Analisis Kontribusi tiap Grup Fitur
12. Prediksi Sampel Gambar Baru
13. Simpan Model (`.pkl`)

---

## Tech Stack

- **OpenCV** & **scikit-image** — image processing & ekstraksi fitur (GLCM, LBP)
- **scikit-learn** — SVM, preprocessing, evaluasi model
- **Roboflow** — sumber & download dataset
- **NumPy / Matplotlib** — komputasi numerik & visualisasi

---

##  Catatan

Proyek ini berfokus pada pendekatan **classical machine learning** (bukan deep learning/CNN) untuk menunjukkan bahwa kombinasi ekstraksi fitur warna dan tekstur yang tepat, dipasangkan dengan SVM, masih bisa mencapai akurasi tinggi (96.67%) pada tugas klasifikasi citra sederhana dengan dataset berskala kecil-menengah.

---

##  Lisensi

Proyek ini dibuat untuk keperluan akademik (tugas mata kuliah Pengolahan Citra Digital).
