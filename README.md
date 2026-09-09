# 🏠 Real Estate Property Tier Prediction

An end-to-end Machine Learning pipeline and interactive web application designed to classify real estate properties into economic tiers (**Low, Medium, or High**) based on physical attributes, location metadata, estimated evaluations, and tax indicators.

## 🔗 Live Application Demo
*(If deployed on Streamlit Community Cloud, insert your custom link here)*  
👉 [Live Streamlit App Link](https://streamlit.io)

---

## 📌 Project Overview
This project processes a housing dataset containing historical property transaction details. The engineering lifecycle spans exploratory data analysis (EDA), multi-column text encoding, outlier scrubbing, feature scaling, model benchmarking, and full application compilation.

A **Decision Tree Classifier** was finalized due to its excellent stratification properties, achieving near-flawless macro/weighted metrics across balanced quantile splits.

### 📊 Dataset Attributes Included
* **Date Columns:** Transaction timelines extracted into engineered `Day`, `Month`, and `Year` variables.
* **Numerical Metrics:** `Estimated Value`, `Sale Price`, `num_rooms`, `num_bathrooms`, `carpet_area`, and `property_tax_rate`.
* **Categorical Layouts:** `Locality`, `Property`, `Residential`, and structural orientation (`Face`).
* **Target Classification:** `Property_Tier` (Categorized seamlessly via 3-quantile distribution mapping).

---

## 📁 Repository Structure
```text
├── .github/
├── data/
│   └── V3.csv                 # Raw data file containing transaction rows
├── notebooks/
│   └── real_estate_eda.ipynb  # Phase-by-phase model construction workbook
├── app.py                     # Python source script running the Streamlit app
├── real_estate_model.pkl      # Serialized Decision Tree Classifier artifact
├── label_encoders.pkl         # Dictionary of fitted structural text label maps
├── scaler.pkl                 # Fitted global StandardScaler coefficients
├── requirements.txt           # Environment execution prerequisites
└── README.md                  # Project documentation manual
```

---

## ⚡ Performance Summary
Two models were evaluated during benchmarking under identical 80-20 partition states:

1. **Decision Tree Classifier (Chosen Model)**
   * **Accuracy:** `100.0%`
   * **Precision / Recall / F1-Score:** `1.00` across all tiers (`High`, `Low`, `Medium`)
   * Fully captures exact transaction price borders set via engineered quantile bins.

2. **Support Vector Machine (SVM) Classifier**
   * **Accuracy:** `38.07%`
   * Struggled due to un-optimized default hyper-parameters on multidimensional continuous feature landscapes.

---

## 🛠️ Local Installation and Setup

Follow these quick commands to install dependencies and run the server environment locally:

### 1. Clone the Workspace
```bash
git clone https://github.com
cd YOUR_REPOSITORY_NAME
```

### 2. Configure a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
Make sure your `requirements.txt` includes `streamlit`, `pandas`, `numpy`, and `scikit-learn`. Then execute:
```bash
pip install -r requirements.txt
```

### 4. Fire up the Streamlit Server
```bash
streamlit run app.py
```
Your default browser will launch a portal instantly pointing to `http://localhost:8501`.

---

## 💻 How to Use the Interface
1. **Configure Timeline Details:** Feed the target transaction year, month, and day matrix using inputs on the form sidebars.
2. **Assign Categorical Specifications:** Choose options from the picklists generated dynamically via the backend encoding files (e.g., `Waterbury`, `Single Family`, `Detached House`).
3. **Impute Core Numeric Measures:** Enter structural area volumes and financial value indexes.
4. **Trigger Generation:** Click **Predict Property Tier**. The application standardizes inputs on the fly using `scaler.pkl` and provides the predicted output class safely mapped inside informational block banners.

---

## 🤖 Model Engineering Lifecycle
The code pipeline follows rigorous, production-grade workflows:
* **Missing Value Imputation:** Categorical values are handled via mode/unknown tracking; continuous numeric columns are bound using global distribution means.
* **Statistical Interquartile Outlier Removal:** Errant transaction errors are clipped safely out of target fields via adaptive IQR filters ($Q1 - 1.5 \times IQR$ to $Q3 + 1.5 \times IQR$).
* **State Preservation:** Scikit-Learn pipelines are saved carefully inside separate pickle collections to avoid feature leak behaviors on deployment entry interfaces.

---

