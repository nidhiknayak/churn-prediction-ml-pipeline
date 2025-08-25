import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class ChurnAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_processed = None
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def load_and_explore_data(self):
        """Load data and perform initial exploration"""
        print("=== LOADING AND EXPLORING DATA ===")
        self.df = pd.read_csv(self.file_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Drop UID
        if 'UID' in self.df.columns:
            self.df = self.df.drop(columns=['UID'])
        
        # Target analysis
        print(f"\nTarget distribution:")
        target_counts = self.df['Target_ChurnFlag'].value_counts()
        print(target_counts)
        print(f"Churn rate: {target_counts[1] / len(self.df) * 100:.2f}%")
        
        # Missing values analysis
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(f"\nColumns with missing values: {len(missing_df[missing_df['Missing_Count'] > 0])}")
        print(f"Total missing values: {missing_data.sum()}")
        
        return missing_df
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n=== PREPROCESSING DATA ===")
        self.df_processed = self.df.copy()
        
        # Handle target variable
        self.df_processed['Target_ChurnFlag'] = pd.to_numeric(
            self.df_processed['Target_ChurnFlag'], errors='coerce'
        )
        
        # Remove rows with NaN target
        initial_rows = len(self.df_processed)
        self.df_processed = self.df_processed.dropna(subset=['Target_ChurnFlag'])
        print(f"Rows removed due to missing target: {initial_rows - len(self.df_processed)}")
        
        # Separate numeric and categorical columns
        numeric_cols = self.df_processed.select_dtypes(include=['number']).columns.tolist()
        numeric_cols.remove('Target_ChurnFlag')
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric features: {len(numeric_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        
        # Handle categorical variables
        for col in categorical_cols:
            self.df_processed[col] = self.df_processed[col].fillna('Missing')
            le = LabelEncoder()
            self.df_processed[col] = le.fit_transform(self.df_processed[col])
            self.label_encoders[col] = le
        
        # Handle missing values in numeric columns
        for col in numeric_cols:
            if self.df_processed[col].isnull().sum() > 0:
                self.df_processed[col] = self.df_processed[col].fillna(
                    self.df_processed[col].median()
                )
        
        print(f"Final dataset shape: {self.df_processed.shape}")
        print(f"Remaining missing values: {self.df_processed.isnull().sum().sum()}")
        
    def train_model(self):
        """Train the churn prediction model"""
        print("\n=== TRAINING MODEL ===")
        
        # Prepare features and target
        X = self.df_processed.drop('Target_ChurnFlag', axis=1)
        y = self.df_processed['Target_ChurnFlag']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluation metrics
        print("\n=== MODEL PERFORMANCE ===")
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # ROC-AUC
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {auc_score:.4f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10))
        
        # Store test data for visualization
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        return auc_score, cm
    
    def create_visualizations(self):
        """Create visualizations for insights"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Churn Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Target Distribution
        target_counts = self.df['Target_ChurnFlag'].value_counts()
        axes[0, 0].pie(target_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                       colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Churn Distribution')
        
        # 2. Feature Importance (Top 10)
        top_features = self.feature_importance.head(10)
        axes[0, 1].barh(range(len(top_features)), top_features['importance'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Top 10 Feature Importance')
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)
        axes[0, 2].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].set_title('ROC Curve')
        axes[0, 2].legend()
        
        # 4. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 5. Prediction Probability Distribution
        axes[1, 1].hist(self.y_pred_proba[self.y_test == 0], alpha=0.7, label='No Churn', bins=30)
        axes[1, 1].hist(self.y_pred_proba[self.y_test == 1], alpha=0.7, label='Churn', bins=30)
        axes[1, 1].set_xlabel('Prediction Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
        
        # 6. Missing Data Analysis
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        top_missing = missing_percent[missing_percent > 0].head(10)
        if len(top_missing) > 0:
            axes[1, 2].bar(range(len(top_missing)), top_missing.values)
            axes[1, 2].set_xticks(range(len(top_missing)))
            axes[1, 2].set_xticklabels(top_missing.index, rotation=45, ha='right')
            axes[1, 2].set_ylabel('Missing Percentage')
            axes[1, 2].set_title('Top 10 Columns with Missing Data')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                           transform=axes[1, 2].transAxes, fontsize=14)
            axes[1, 2].set_title('Missing Data Analysis')
        
        plt.tight_layout()
        plt.savefig('churn_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Dashboard saved as 'churn_analysis_dashboard.png'")
    
    def generate_insights(self):
        """Generate business insights"""
        print("\n=== KEY BUSINESS INSIGHTS ===")
        
        # Calculate key metrics
        churn_rate = self.df['Target_ChurnFlag'].sum() / len(self.df) * 100
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)
        
        insights = [
            f"🎯 Overall churn rate: {churn_rate:.2f}%",
            f"📊 Model performance (AUC): {auc_score:.3f}",
            f"🔍 Dataset size: {len(self.df):,} customers",
            f"📈 Features analyzed: {len(self.df.columns)-1}",
        ]
        
        # Top risk factors
        top_features = self.feature_importance.head(5)['feature'].tolist()
        insights.append(f"⚠️ Top risk factors: {', '.join(top_features[:3])}")
        
        # Model accuracy insights
        cm = confusion_matrix(self.y_test, self.y_pred)
        accuracy = np.diag(cm).sum() / cm.sum()
        precision = cm[1,1] / (cm[1,1] + cm[0,1])
        recall = cm[1,1] / (cm[1,1] + cm[1,0])
        
        insights.extend([
            f"✅ Model accuracy: {accuracy:.3f}",
            f"🎯 Precision: {precision:.3f}",
            f"📡 Recall: {recall:.3f}"
        ])
        
        for insight in insights:
            print(insight)
        
        return insights

# Usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ChurnAnalyzer(r"C:\Users\nidhi\OneDrive\Documents\churn project\deepq_ai_assignment1_data.csv")
    
    try:
        # Run complete analysis
        missing_df = analyzer.load_and_explore_data()
        analyzer.preprocess_data()
        auc_score, cm = analyzer.train_model()
        analyzer.create_visualizations()
        insights = analyzer.generate_insights()
        
        print("\n✅ Analysis completed successfully!")
        print("📊 Dashboard saved as 'churn_analysis_dashboard.png'")
        print("🚀 Ready for presentation and deployment!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("🔧 Please check your data file path and format.")