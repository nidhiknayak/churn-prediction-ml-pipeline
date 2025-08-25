# 🎯 Customer Churn Prediction System

*A comprehensive machine learning solution for predicting customer churn with interactive web deployment*

## 🌟 Project Highlights

- **🎯 99.7% Model Accuracy** - Exceptional predictive performance
- **🚀 Interactive Web App** - Real-time churn predictions via Streamlit
- **📊 Comprehensive Analysis** - 167K+ customer records, 215+ features
- **💼 Business-Ready** - Production-ready deployment with actionable insights
- **🎨 Professional Visualization** - Interactive dashboards and analytics

## 📊 Key Results

| Metric | Score | Impact |
|--------|-------|--------|
| **Accuracy** | 99.7% | Exceptional model performance |
| **ROC-AUC** | 1.000 | Perfect discrimination ability |
| **Churn Rate** | 40.07% | 66,919 at-risk customers identified |
| **Features** | 215 | Comprehensive customer attributes |
| **Dataset Size** | 167,020 | Large-scale analysis |


## 🛠️ Technology Stack

### **Core Technologies**
- **Backend**: Python 3.8+, Scikit-learn, Pandas, NumPy
- **Frontend**: Streamlit, Plotly, HTML/CSS
- **ML Algorithm**: Random Forest Classifier (optimized)
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Deployment**: Streamlit Cloud

### **Key Libraries**
```python
streamlit==1.29.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
plotly==5.18.0
matplotlib==3.7.2
seaborn==0.12.2
```

## 📁 Project Structure

```
📦 customer-churn-prediction
├── 🌐 app.py                          # Streamlit web application
├── 🔬 churn_analysis.py               # Comprehensive analysis pipeline
├── 🚀 main.py                         # Core analysis script
├── 🤖 churn_model.pkl                 # Trained ML model
├── 📊 data.csv                        # Dataset (if shareable)
├── 📋 requirements.txt                # Dependencies
├── 📈 outputs/                        # Generated visualizations
│   ├── churn_analysis_dashboard.png   # Main dashboard
│   ├── churn_distribution.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_curve.png
├── 📄 Churn_Analysis_Report.pdf       # Technical report
├── 📑 Customer-Churn-Prediction-Analysis-2.pdf  # Business presentation
└── 📖 README.md
```

## 🚀 Quick Start

### **1. Clone Repository**
```bash
git clone https://github.com/your-username/customer-churn-prediction
cd customer-churn-prediction
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run Analysis**
```bash
python churn_analysis.py
```

### **4. Launch Web Application**
```bash
streamlit run app.py
```

**🌐 Open browser to `http://localhost:8501`**

## 📊 Features Overview

### 🤖 **Machine Learning Pipeline**
- **Advanced Preprocessing**: Automated data cleaning and feature engineering
- **Random Forest Model**: Optimized hyperparameters for maximum accuracy
- **Feature Importance**: Identify top churn risk factors (X16, X7, X4)
- **Model Persistence**: Serialized model for production deployment

### 🌐 **Interactive Web Application**
- **📤 Data Upload**: Process custom CSV datasets
- **🔮 Real-time Predictions**: Instant churn probability scoring
- **📊 Analytics Dashboard**: Key metrics and performance indicators
- **🎯 Interactive UI**: Professional design with gauge charts

### 📈 **Comprehensive Analytics**
- **📋 Dataset Overview**: 167K records, 215 features analyzed
- **🎯 Target Analysis**: 40.07% churn rate identification
- **📊 Missing Data**: Intelligent handling of 1M+ missing values
- **🔍 Feature Engineering**: Advanced categorical encoding

## 🎯 Model Performance

### **Classification Metrics**
```
📊 Confusion Matrix Results:
    ┌─────────────┬─────────────┐
    │   Actual    │   Predicted │
    ├─────────────┼─────────────┤
    │ No Churn    │   19,954    │    66   │
    │ Churn       │      0      │ 13,384  │
    └─────────────┴─────────────┘
```

| **Metric** | **Score** | **Business Impact** |
|------------|-----------|---------------------|
| **Accuracy** | 99.7% | Exceptional prediction reliability |
| **Precision** | 99.3% | Minimal false positives |
| **Recall** | 100.0% | Zero missed churn cases |
| **F1-Score** | 99.6% | Balanced performance |

### **Feature Importance Rankings**
| **Rank** | **Feature** | **Importance** | **Business Relevance** |
|----------|-------------|----------------|------------------------|
| 🥇 | X16 | 30.92% | Primary churn driver |
| 🥈 | X7 | 26.87% | Secondary risk factor |
| 🥉 | X4 | 19.13% | Key behavioral indicator |
| 4️⃣ | X8 | 16.14% | Customer engagement metric |
| 5️⃣ | X9 | 0.46% | Supporting factor |

## 💼 Business Impact & Recommendations

### **🎯 Immediate Actions**
1. **🚀 Production Deployment**: Implement real-time churn scoring
2. **🎯 Target High-Risk Customers**: Focus on 66,919 predicted churners
3. **📊 Monitor Key Features**: Track X16, X7, X4 for early warnings

### **📈 Strategic Initiatives**
- **🎯 Personalized Campaigns**: Data-driven retention strategies
- **⚠️ Early Warning System**: Automated churn risk alerts
- **🔬 A/B Testing**: Validate retention strategies on predicted churners
- **🔄 Model Updates**: Monthly retraining with new data

### **💰 Expected ROI**
- **📉 15-25% Churn Reduction** through targeted interventions
- **💎 Improved Customer Lifetime Value** via proactive retention
- **💵 Optimized Marketing Spend** on retention vs acquisition
- **📊 Data-Driven Decisions** for customer success teams

## 🎨 Visualizations & Dashboard

The system generates comprehensive visualizations including:

- **📊 Churn Distribution Analysis** (59.9% retained, 40.1% churned)
- **🎯 ROC Curve** (Perfect AUC = 1.000)
- **📈 Feature Importance Rankings** (Top 10 drivers)
- **🔥 Confusion Matrix Heatmap** (Performance breakdown)
- **📊 Prediction Probability Distribution** (Risk scoring)
- **❌ Missing Data Analysis** (Data quality insights)


### **🎯 Skills Demonstrated**
- **🔬 Data Science**: End-to-end ML pipeline development
- **💻 Software Engineering**: Clean, production-ready code
- **🌐 Web Development**: Interactive UI design and deployment
- **💼 Business Intelligence**: Actionable insights and recommendations

## 🔧 Technical Implementation

### **🧠 Model Configuration**
```python
RandomForestClassifier(
    n_estimators=100,      # Optimal tree count
    max_depth=10,          # Prevent overfitting
    min_samples_split=5,   # Robust splitting
    min_samples_leaf=2,    # Leaf node control
    class_weight='balanced', # Handle imbalance
    random_state=42        # Reproducibility
)
```

### **📊 Data Preprocessing**
- **🔧 Missing Value Imputation**: Median/mode strategies
- **📝 Categorical Encoding**: Label encoding for non-numeric features
- **📏 Feature Scaling**: StandardScaler normalization
- **⚖️ Class Balancing**: Weighted approach for imbalanced data

### **🚀 Deployment Architecture**
- **📱 Streamlit Frontend**: Interactive user interface
- **🤖 Model API**: Real-time prediction endpoint
- **📊 Analytics Engine**: Performance monitoring
- **🔄 Model Persistence**: Pickle serialization

## 👨‍💻 About the Developer

**Nidhi Nayak** - Aspiring Data Scientist & ML Engineer

### **🎯 Core Competencies**
- **🤖 Machine Learning**: Advanced model development and evaluation
- **📊 Data Science**: Statistical analysis and insight generation  
- **💻 Software Engineering**: Production-ready code development
- **🌐 Web Development**: Interactive application deployment
- **💼 Business Intelligence**: Translating data into actionable strategies

## 🎯 Future Enhancements

- **🔄 Real-time Data Pipeline**: Live model updates
- **🤖 Advanced ML Models**: XGBoost, Neural Networks
- **📱 Mobile Application**: Cross-platform deployment
- **🔗 CRM Integration**: Seamless business system connection
- **📈 Advanced Analytics**: Customer lifetime value prediction

---

⭐ **Star this repository if you found it helpful!** ⭐

*Built with ❤️ for data science excellence and business impact*

**🚀 Ready to deploy • 📊 Production-ready • 💼 Business-focused**