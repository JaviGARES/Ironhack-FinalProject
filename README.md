# 🏠 Hybrid House Price Prediction (Tabular + Images)

A multi-modal ML system combining engineered tabular features with CNN image embeddings

This project builds a hybrid machine learning model that predicts house prices by combining:
- **Structured tabular data** (bedrooms, bathrooms, square footage, city…)
- **Visual information** extracted from exterior house photos using EfficientNet
- **Gradient-boosted decision trees** (XGBoost) for final price prediction

The approach outperforms tabular-only models by capturing implicit visual attributes such as condition, curb appeal, architectural style, landscaping, and exterior quality—features that are typically unavailable or too subjective for users to input manually.

---

## 📂 Dataset

The project uses the public Kaggle dataset:

🔗 [House Prices and Images - SoCal](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal)

It contains:
- **15,474** Southern California listings
- Tabular metadata (bed/bath/sqft, city, price, etc.)
- One exterior house image per listing
- Clean structure with no missing values

---

## ✨ Project Highlights

### ✔️ Advanced Feature Engineering
- `log₁₊price` target transformation
- Spaciousness metrics: `sqft_per_bed`, `sqft_per_bath`
- Total rooms
- Log-transformed sqft
- Target-encoded city (mean log-price per city in training split)
- Standardized numeric features

### ✔️ CNN Transfer Learning (EfficientNet)
- EfficientNetB0 and EfficientNetB3 pretrained on ImageNet
- Fine-tuned EfficientNetB0 (last 10 layers unfrozen)
- Strong augmentation pipeline (flip, zoom, rotation, translation, contrast)
- Extracted 128-dimensional visual embeddings representing house condition & style

### ✔️ Hybrid Architecture
The hybrid model concatenates:
```
[scaled tabular features] + [128-dim image embedding]
```
and feeds the combined vector into a tuned XGBoost regressor.

### ✔️ Performance

| Model | R² (test) |
|-------|-----------|
| Linear Regression (baseline) | ~0.40 |
| Random Forest | ~0.42 |
| XGBoost (baseline) | ~0.69 |
| XGBoost (feat-engineered + tuned) | ~0.78 |
| **Hybrid Model (Tabular + Images)** | **~0.80** |

---

## 📁 Repository Structure
```
📦 FinalProject/
│
├── .gitignore                    # Git ignore rules
├── README.md                     # Project documentation
├── main.ipynb                    # Full training pipeline: tabular, CNN, hybrid
├── Presentation.pptx             # Project presentation slides
│
└── streamlit_app/
    ├── app.py                    # Streamlit demo application
    ├── requirements.txt          # Python dependencies
    └── models/
        ├── city_target_enc.json           # Target-encoding mapping for city feature
        ├── config.json                    # Model configuration (image size, features, etc.)
        ├── effnetb0_simple_lastlp.keras   # EfficientNetB0 variant
        ├── effnetb0_t1_best.keras         # Fine-tuned EfficientNetB0 (best checkpoint)
        ├── effnetb3_t1_best.keras         # Fine-tuned EfficientNetB3 (experimental)
        ├── hybrid_tuned_model.pkl         # Final hybrid regressor (tabular + images)
        ├── hybrid_xgb_model.pkl           # Hybrid XGBoost model
        └── xgb_fe_tuned_pipeline.pkl      # Tabular-only preprocessing + XGBoost
```

---

## 🧠 Modeling Pipeline

### 1️⃣ Exploratory Data Analysis
- Verified dataset integrity (15,474 rows, all images present)
- Visualized price distribution
- Modeled log-price due to heavy skew
- Analyzed correlations and feature relationships
- Identified location as a major driver → target encoding

---

### 2️⃣ Tabular Modeling

**Baseline models:**
- Linear Regression
- Random Forest
- XGBoost

XGBoost clearly outperformed with R² ≈ 0.69, but not enough → needed better features.

**Feature engineering dramatically improved results:**
- Spaciousness ratios
- `log_sqft`
- `total_rooms`
- Target encoding

**XGBoost FE + tuning ⇒ R² ≈ 0.78**

---

### 3️⃣ Image Modeling (CNN)

**Models tested:**
- EfficientNetB0 (baseline)
- EfficientNetB3
- EfficientNetB0 fine-tuned (last 10 layers unfrozen)

CNNs alone performed poorly for price prediction, but:

**EfficientNetB0 tuned** produced the best embeddings, used in the hybrid model.

**Images capture:**
- Renovation quality
- Architectural style
- Curb appeal
- Landscaping
- General exterior condition

---

### 4️⃣ Hybrid Model

**Architecture:**
1. Encode tabular data → StandardScaler
2. Compute image embedding → EfficientNetB0 tuned
3. Concatenate → `[tabular_scaled | embedding_128]`
4. Predict log-price → XGBoost regressor
5. Convert back to USD using `expm1`

**Performance:**
- **R² ≈ 0.80** on test set
- Visual features successfully improved the model

---

## 🚀 Streamlit Demo

A Streamlit app is included so users can:
- Input property details
- Optionally upload an exterior photo
- Choose:
  - Tabular-only price estimate
  - Hybrid model price estimate
- Get a final predicted price + explanation text

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction/streamlit_app
```

### 2️⃣ Create a virtual environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

### 5️⃣ Use the interface
- Fill in tabular features
- Upload an image (optional)
- Choose model type
- View the predicted price

---

## 📌 Model Artifacts

Inside `streamlit_app/models`:

| File | Purpose |
|------|---------|
| `xgb_fe_tuned_pipeline.pkl` | Full preprocessing + tuned XGBoost (tabular-only model) |
| `effnetb0_t1_best.keras` | Fine-tuned EfficientNetB0 - main CNN for image embeddings |
| `effnetb0_simple_lastlp.keras` | EfficientNetB0 variant (alternative architecture) |
| `effnetb3_t1_best.keras` | Fine-tuned EfficientNetB3 (experimental, higher capacity) |
| `hybrid_xgb_model.pkl` | Hybrid regressor (tabular + image features) |
| `hybrid_tuned_model.pkl` | Final tuned hybrid model |
| `city_target_enc.json` | Target-encoding mapping for city feature |
| `config.json` | Model configuration (image dimensions, feature list) |

**Note:** The app automatically loads the best-performing model configuration.

---

## 📊 Presentation

A detailed project presentation is available in `Presentation.pptx`, covering:
- Problem statement and motivation
- Data exploration and insights
- Modeling approach and architecture
- Results and performance metrics
- Demo and future improvements

---

## 🧭 Future Work

- Incorporate multiple images per listing (interior + exterior)
- Use satellite imagery to capture neighborhood quality
- Explore multi-task learning (predict price + condition score)
- Implement SHAP for full model explainability
- Deploy the system as a full API + web app

---

## 👤 Author

**Javier García Esteve** — Final project for the IRONHACK Data Science Bootcamp.

Includes end-to-end ML engineering, deep learning, image modeling, and application deployment.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).