<div align="center">
  
#  Apple Product Sales Analytics & Machine Learning Pipeline
  
**A production-grade, end-to-end data science dashboard for analyzing global hardware ecosystems and forecasting services revenue using non-linear pipelines and dynamic clustering.**
  
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Data_Visualization-purple?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Engineering-black?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

---

</div>

## 📖 Project Overview

The **Apple Insights Platform** transforms static historical sales data (`apple_sales_2024.csv`) into actionable business intelligence. Originally prototyped in a monolithic Jupyter Notebook, the analytics pipeline evaluates hardware unit sales spanning across multiple global demographics (North America, Europe, Asia, Greater China) and models their fundamental impact on **Services Revenue** optimization.

This repository features a completely decoupled architecture, deploying scalable backend modularity alongside a natively integrated, dark-themed SaaS frontend application crafted for both Data Scientists and Executive Stakeholders.

<br>

## 🚀 Core Features

- **Executive Summary Engine**: Deploys dynamic, jargon-free business logic to instantly evaluate product dominance, regional penetration risks, and automated Service Expansion opportunities securely utilizing Streamlit `st.success` / `st.warning` routing.
- **Automated Feature Selection**: Pre-computes Multi-Collinearity and Non-Linear Random Forest signal importances to mathematically aggressively prune inherently weak variables natively before injecting them into the training matrices.
- **Predictive Modelling & What-If Simulator**: Access robust pipeline parameterizations integrating 5 custom statistical engines:
  - `Linear & Multiple Linear Regression`
  - `Polynomial Regression (Auto-Scaled Interaction Multipliers)`
  - `Random Forest & Gradient Boosting (Robust Ensembles)`
  - Includes a real-time **Sales Simulator** adjusting predictions instantaneously across user-slidable limits.
- **Unsupervised K-Means Segmentation**: Actively explores dataset variance mathematically using interactive dynamic clustering natively cast onto 3-Dimensional coordinate spaces, including native Silhouette validation.
- **1-Click Export Engineering**: Compiles actively filtered, multi-variable dataframe modifications directly into a locally sanitized `.csv` package for one-button downloads globally!

<br>

## 📂 System Architecture

The project enforces robust decoupling principles isolating backend Machine Learning engines and graphical parameters from the frontend layout wrappers.

```tree
├── app/
│   ├── main.py                    # Streamlit Application Entry Node
│   └── components/
│       ├── kpi.py                 # Executive Dashboard Numeric Summaries
│       ├── sidebar.py             # Global Parametric Configuration Layouts
│       ├── styles.py              # Custom CSS Topologies (SaaS Dark Mode)
│       └── tabs/                  # Isolated Feature Interface Pages
│           ├── analytics.py
│           ├── data_quality.py
│           ├── executive.py
│           ├── geospatial.py 
│           ├── overview.py
│           ├── predictions.py
│           └── segmentation.py
├── data/
│   └── raw/
│       └── apple_sales_2024.csv   # Historical Ground Truth Data
├── src/
│   ├── analysis/
│   │   ├── insights.py            # Business Logic Interpreter
│   │   └── statistics.py          # ANOVA & Statistical Validation
│   ├── data/
│   │   └── loader.py              # Exception-Handled DataFrame Ingestion
│   ├── features/
│   │   └── builder.py             # Feature Engineering & Scaling
│   ├── models/
│   │   ├── clustering.py          # K-Means Parameterization
│   │   └── regression.py          # Pipeline Modularity & Supervised AI
│   └── visualization/
│       └── charts.py              # Plotly Abstract Factory Constraints
├── .streamlit/
│   └── config.toml                # Native UI Injection Environment Params
└── requirements.txt               # Strict Environment Dependency Lock
```

<br>

## ⚙️ Local Installation Guide

1. **Clone the repository:**
```bash
git clone https://github.com/Khanz9664/Comprehensive-Data-Analysis-Visualization-of-Apple-Product-Sales.git
cd Comprehensive-Data-Analysis-Visualization-of-Apple-Product-Sales
```

2. **Initialize a secure Virtual Environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3. **Install exact pip dependencies:**
```bash
pip install -r requirements.txt
```

4. **Boot the Streamlit Engine:**
```bash
streamlit run app/main.py
```

<br>

## 📊 Analytics Breakdown

- **Data Exploration**: Heatmaps, radar geometries, and explicit mathematical schemas (Missing Values, IQR Object Detection) guarantee dataset integrity safely before AI processing.
- **Geospatial Insights**: Evaluates exact distribution pipelines isolating regional success stories utilizing native Plotly Choropleth integrations.
- **Data Engineering**: Dynamic polynomial transformations (`degree=X`), feature standardizations (`StandardScaler()`), and automated precision matrices (`KFold CV`) guarantee production-grade statistical inference natively.
