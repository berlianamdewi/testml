# 🌊 Flood Evacuation Dashboard
## Adaptive Geospatial Routing · Random Forest · Cilacap Regency
**IJIGSP 2025 | Telkom University Purwokerto**

---

## 🚀 Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan dashboard
```bash
streamlit run app.py
```

Dashboard akan terbuka otomatis di browser: `http://localhost:8501`

---

## 📦 Deploy ke Streamlit Cloud (gratis)

1. Push folder ini ke GitHub:
```bash
git init
git add .
git commit -m "flood evacuation dashboard"
git remote add origin https://github.com/USERNAME/flood-evacuation-dashboard.git
git push -u origin main
```

2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo → pilih `app.py` → Deploy!

---

## 🗂️ Struktur File

```
flood_dashboard/
├── app.py              ← Main Streamlit app (semua tab)
├── requirements.txt    ← Python dependencies
└── README.md          ← Dokumentasi ini
```

---

## 🗺️ Fitur Dashboard

| Tab | Konten |
|-----|--------|
| 📊 Data & Preprocessing | Raw data, preprocessing result, time series cuaca |
| 🗺️ Peta Spasial | Folium map: flood points, shelter, heatmap, cluster, rute overlay |
| 🤖 Evaluasi Model | Accuracy, Cohen's κ, Confusion Matrix, Feature Importance, Learning Curve |
| 🌳 Decision Tree | Visualisasi pohon keputusan + aturan text |
| 🎯 Simulasi Rute | Interactive what-if: slider RH/RR/jarak/flood → prediksi + peta |
| 💡 Novelty Analysis | Perbandingan paper vs novelty, kontradiksi sehat, tabel akurasi |

---

## ⚡ Novelty Utama

```python
# Adaptive threshold dari data riil Cilacap (bukan literatur umum)
MEAN_RH = 83.53  # threshold Risk L3/L4
MEAN_RR = 15.16  # threshold Risk L3/L4
MAX_RR  = 135.0  # threshold Risk L5

# Labeling dengan ground truth banjir nyata (BNPB)
FLOOD_DATES = ["2024-10-09","2024-10-22","2024-11-06",
               "2024-11-19","2024-12-03","2024-12-08"]

# Override: hari banjir nyata → SAFEST (meskipun cuaca tampak normal)
if row.is_flood_day == 1:
    return 1  # SAFEST
```

---

## 📊 Data Sources

- **Route data**: Google Maps API (855 route simulations)
- **Weather data**: BMKG Tunggul Wulung Station (77 days, Oct–Dec 2024)
- **Flood points**: BNPB Cilacap (120 flood points)
- **Shelters**: BNPB Cilacap (314 evacuation shelters)
- **Satellite**: Sentinel-2 (NDWI, NDMI indices)
