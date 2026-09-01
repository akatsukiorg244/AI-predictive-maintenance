# 🔧 Centrifugal Pump Digital Twin — Judge Presentation

&gt; **Trained on real sensor data** | **3-Layer Architecture** | **13 Features** | **Live WebSocket Streaming**

---

## 🏗️ Architecture Overview (For Judges)

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 🧠 **ML Layer** | Python, scikit-learn, joblib | Trains 3 RandomForest models on 13 real sensor features |
| ⚙️ **Backend Layer** | FastAPI, WebSocket, uvicorn | Serves predictions via REST API + streams live data |
| 🎨 **Frontend Layer** | Three.js, Chart.js, vanilla JS | 3D pump visualization + real-time dashboard |

**Data Flow:** CSV → ML Training → `.pkl` artifact → FastAPI loads → WebSocket streams → Three.js renders

---

## 🚀 Quick Start

### 1. Train the Model (ML Layer)
```bash
cd ml/
python train_model.py        # Uses your converted_file (1).csv
# Output: pump_model.pkl
