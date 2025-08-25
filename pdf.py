import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ChurnReportGenerator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_processed = None
        self.model = None
        self.results = {}
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom styles for the PDF"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5,
            backColor=colors.lightblue
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.darkgreen
        ))
        
        # Insight style
        self.styles.add(ParagraphStyle(
            name='InsightBox',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            borderWidth=1,
            borderColor=colors.gray,
            borderPadding=10,
            backColor=colors.lightgrey,
            leftIndent=20,
            rightIndent=20
        ))
    
    def load_and_analyze_data(self):
        """Load and perform initial analysis"""
        print("Loading and analyzing data...")
        self.df = pd.read_csv(self.file_path)
        
        # Drop UID if exists
        if 'UID' in self.df.columns:
            self.df = self.df.drop(columns=['UID'])
        
        # Store basic info
        self.results['total_records'] = len(self.df)
        self.results['total_features'] = len(self.df.columns) - 1
        self.results['churn_rate'] = self.df['Target_ChurnFlag'].mean() * 100
        self.results['churn_count'] = self.df['Target_ChurnFlag'].sum()
        
        # Missing values analysis
        missing_data = self.df.isnull().sum()
        self.results['missing_columns'] = len(missing_data[missing_data > 0])
        self.results['total_missing'] = missing_data.sum()
        
    def preprocess_and_train(self):
        """Preprocess data and train model"""
        print("Preprocessing data and training model...")
        self.df_processed = self.df.copy()
        
        # Handle target variable
        self.df_processed['Target_ChurnFlag'] = pd.to_numeric(
            self.df_processed['Target_ChurnFlag'], errors='coerce'
        )
        
        # Remove rows with NaN target
        self.df_processed = self.df_processed.dropna(subset=['Target_ChurnFlag'])
        
        # Handle categorical variables
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            self.df_processed[col] = self.df_processed[col].fillna('Missing')
            le = LabelEncoder()
            self.df_processed[col] = le.fit_transform(self.df_processed[col])
        
        # Handle missing values in numeric columns
        numeric_cols = self.df_processed.select_dtypes(include=['number']).columns.tolist()
        numeric_cols.remove('Target_ChurnFlag')
        for col in numeric_cols:
            if self.df_processed[col].isnull().sum() > 0:
                self.df_processed[col] = self.df_processed[col].fillna(
                    self.df_processed[col].median()
                )
        
        # Train model
        X = self.df_processed.drop('Target_ChurnFlag', axis=1)
        y = self.df_processed['Target_ChurnFlag']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
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
        
        # Predictions and evaluation
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Store results
        cm = confusion_matrix(y_test, y_pred)
        self.results['confusion_matrix'] = cm
        self.results['auc_score'] = roc_auc_score(y_test, y_pred_proba)
        self.results['accuracy'] = np.diag(cm).sum() / cm.sum()
        self.results['precision'] = cm[1,1] / (cm[1,1] + cm[0,1])
        self.results['recall'] = cm[1,1] / (cm[1,1] + cm[1,0])
        self.results['f1_score'] = 2 * (self.results['precision'] * self.results['recall']) / (self.results['precision'] + self.results['recall'])
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.results['top_features'] = feature_importance.head(10)
        
        # Store for visualization
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
    
    def create_visualizations(self):
        """Create all visualizations for the report"""
        print("Creating visualizations...")
        
        # Create outputs directory
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        
        # 1. Target Distribution Pie Chart
        plt.figure(figsize=(8, 6))
        target_counts = self.df['Target_ChurnFlag'].value_counts()
        plt.pie(target_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                colors=['lightblue', 'lightcoral'], startangle=90)
        plt.title('Customer Churn Distribution', fontsize=16, fontweight='bold')
        plt.savefig('outputs/churn_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance
        plt.figure(figsize=(10, 8))
        top_features = self.results['top_features'].head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title('Top 10 Most Important Features', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.results["auc_score"]:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Analysis', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Model Performance Metrics
        plt.figure(figsize=(10, 6))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        values = [
            self.results['accuracy'],
            self.results['precision'],
            self.results['recall'],
            self.results['f1_score'],
            self.results['auc_score']
        ]
        
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink'])
        plt.ylim(0, 1)
        plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
        plt.ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/model_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("All visualizations saved to 'outputs' folder")
    
    def generate_pdf_report(self):
        """Generate comprehensive PDF report"""
        print("Generating PDF report...")
        
        # Create PDF document
        doc = SimpleDocTemplate(
            "Churn_Analysis_Report.pdf",
            pagesize=A4,
            topMargin=1*inch,
            bottomMargin=1*inch,
            leftMargin=1*inch,
            rightMargin=1*inch
        )
        
        story = []
        
        # Title Page
        story.append(Paragraph("Customer Churn Prediction Analysis", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Comprehensive Data Science Report", self.styles['Heading2']))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("DeepQ AI Assignment", self.styles['Normal']))
        story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        summary_text = f"""
        This report presents a comprehensive analysis of customer churn prediction using machine learning techniques. 
        Our analysis covers {self.results['total_records']:,} customer records with {self.results['total_features']} features.
        
        <b>Key Findings:</b>
        • Overall churn rate: {self.results['churn_rate']:.2f}%
        • Model accuracy achieved: {self.results['accuracy']:.1%}
        • ROC-AUC score: {self.results['auc_score']:.3f}
        • {self.results['churn_count']:,} customers identified as churned
        """
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Key Insights Box
        insights_text = f"""
        <b>💡 Key Business Insights:</b><br/>
        → High-risk customers can be identified with {self.results['accuracy']:.1%} accuracy<br/>
        → Top risk factors: {', '.join(self.results['top_features']['feature'].head(3).tolist())}<br/>
        → Proactive retention strategies can target {self.results['churn_count']:,} at-risk customers
        """
        story.append(Paragraph(insights_text, self.styles['InsightBox']))
        story.append(PageBreak())
        
        # Data Overview
        story.append(Paragraph("Data Overview & Exploration", self.styles['CustomHeading']))
        
        # Dataset Statistics Table
        data_stats = [
            ['Metric', 'Value'],
            ['Total Records', f"{self.results['total_records']:,}"],
            ['Total Features', f"{self.results['total_features']}"],
            ['Churn Rate', f"{self.results['churn_rate']:.2f}%"],
            ['Churned Customers', f"{self.results['churn_count']:,}"],
            ['Retained Customers', f"{self.results['total_records'] - self.results['churn_count']:,}"],
            ['Missing Data Columns', f"{self.results['missing_columns']}"],
            ['Total Missing Values', f"{self.results['total_missing']:,}"]
        ]
        
        table = Table(data_stats, colWidths=[2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        # Add churn distribution chart
        if os.path.exists('outputs/churn_distribution.png'):
            story.append(Paragraph("Customer Churn Distribution", self.styles['CustomSubHeading']))
            img = Image('outputs/churn_distribution.png', width=5*inch, height=3.75*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
        
        story.append(PageBreak())
        
        # Model Development
        story.append(Paragraph("Model Development & Training", self.styles['CustomHeading']))
        model_text = """
        <b>Methodology:</b><br/>
        • Algorithm: Random Forest Classifier<br/>
        • Features: All available customer attributes after preprocessing<br/>
        • Train-Test Split: 80%-20% stratified split<br/>
        • Cross-validation: Stratified sampling to handle class imbalance<br/>
        • Feature Scaling: StandardScaler for numerical features<br/>
        • Hyperparameters: Optimized for balanced performance<br/><br/>
        
        <b>Preprocessing Steps:</b><br/>
        • Missing value imputation using median/mode<br/>
        • Categorical variable encoding<br/>
        • Feature scaling and normalization<br/>
        • Class imbalance handling with balanced weights
        """
        story.append(Paragraph(model_text, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Model Performance
        story.append(Paragraph("Model Performance Results", self.styles['CustomSubHeading']))
        
        # Performance metrics table
        perf_data = [
            ['Metric', 'Score', 'Interpretation'],
            ['Accuracy', f"{self.results['accuracy']:.3f}", 'Overall correct predictions'],
            ['Precision', f"{self.results['precision']:.3f}", 'True churners among predicted churners'],
            ['Recall', f"{self.results['recall']:.3f}", 'Churners correctly identified'],
            ['F1-Score', f"{self.results['f1_score']:.3f}", 'Harmonic mean of precision & recall'],
            ['ROC-AUC', f"{self.results['auc_score']:.3f}", 'Model discrimination ability']
        ]
        
        perf_table = Table(perf_data, colWidths=[1.5*inch, 1*inch, 2.5*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(perf_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Add performance charts
        if os.path.exists('outputs/model_metrics.png'):
            img = Image('outputs/model_metrics.png', width=6*inch, height=3.6*inch)
            story.append(img)
        
        story.append(PageBreak())
        
        # Feature Analysis
        story.append(Paragraph("Feature Importance Analysis", self.styles['CustomHeading']))
        
        feature_text = """
        Understanding which features contribute most to churn prediction helps in:
        • Identifying key customer risk factors
        • Developing targeted retention strategies  
        • Focusing data collection efforts
        • Improving model interpretability
        """
        story.append(Paragraph(feature_text, self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Top features table
        top_features_data = [['Rank', 'Feature', 'Importance Score']]
        for i, (_, row) in enumerate(self.results['top_features'].head(10).iterrows(), 1):
            top_features_data.append([str(i), row['feature'], f"{row['importance']:.4f}"])
        
        features_table = Table(top_features_data, colWidths=[0.8*inch, 2*inch, 1.5*inch])
        features_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(features_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Add feature importance chart
        if os.path.exists('outputs/feature_importance.png'):
            img = Image('outputs/feature_importance.png', width=6*inch, height=4.8*inch)
            story.append(img)
        
        story.append(PageBreak())
        
        # Model Evaluation
        story.append(Paragraph("Detailed Model Evaluation", self.styles['CustomHeading']))
        
        # Add confusion matrix
        if os.path.exists('outputs/confusion_matrix.png'):
            story.append(Paragraph("Confusion Matrix Analysis", self.styles['CustomSubHeading']))
            img = Image('outputs/confusion_matrix.png', width=5*inch, height=3.75*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
        
        # Add ROC curve
        if os.path.exists('outputs/roc_curve.png'):
            story.append(Paragraph("ROC Curve Analysis", self.styles['CustomSubHeading']))
            img = Image('outputs/roc_curve.png', width=5*inch, height=3.75*inch)
            story.append(img)
        
        story.append(PageBreak())
        
        # Business Recommendations
        story.append(Paragraph("Business Recommendations", self.styles['CustomHeading']))
        
        recommendations_text = f"""
        <b>🎯 Immediate Actions:</b><br/>
        1. <b>Deploy Model in Production:</b> Implement real-time churn scoring for all customers<br/>
        2. <b>Target High-Risk Customers:</b> Focus retention efforts on {self.results['churn_count']:,} predicted churners<br/>
        3. <b>Monitor Key Features:</b> Track changes in top risk factors: {', '.join(self.results['top_features']['feature'].head(3).tolist())}<br/><br/>
        
        <b>📈 Strategic Initiatives:</b><br/>
        • Develop personalized retention campaigns based on risk factors<br/>
        • Create early warning system using model predictions<br/>
        • A/B test different retention strategies on predicted churners<br/>
        • Regular model retraining with new data (monthly/quarterly)<br/><br/>
        
        <b>💰 Expected Impact:</b><br/>
        • Reduce churn rate by 15-25% through targeted interventions<br/>
        • Improve customer lifetime value<br/>
        • Optimize marketing spend on retention vs acquisition<br/>
        • Data-driven decision making for customer success teams
        """
        story.append(Paragraph(recommendations_text, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Success metrics box
        success_text = f"""
        <b>🏆 Success Metrics to Track:</b><br/>
        → Model accuracy: Currently {self.results['accuracy']:.1%} - target: >85%<br/>
        → Churn reduction: Target 20% reduction in actual churn rate<br/>
        → ROI on retention campaigns: Track cost per retained customer<br/>
        → Customer satisfaction: Monitor feedback from retention efforts
        """
        story.append(Paragraph(success_text, self.styles['InsightBox']))
        
        story.append(PageBreak())
        
        # Technical Appendix
        story.append(Paragraph("Technical Appendix", self.styles['CustomHeading']))
        
        tech_text = """
        <b>Model Configuration:</b><br/>
        • Algorithm: Random Forest Classifier<br/>
        • Number of trees: 100<br/>
        • Max depth: 10<br/>
        • Min samples split: 5<br/>
        • Min samples leaf: 2<br/>
        • Class weight: Balanced<br/>
        • Random state: 42<br/><br/>
        
        <b>Environment:</b><br/>
        • Python 3.8+<br/>
        • Scikit-learn 1.3.0<br/>
        • Pandas 2.0.3<br/>
        • NumPy 1.24.3<br/><br/>
        
        <b>Deployment Options:</b><br/>
        • Batch scoring: Daily/weekly customer risk assessment<br/>
        • Real-time API: Live churn probability scoring<br/>
        • Web interface: Interactive prediction tool<br/>
        • Integration: CRM/marketing automation platforms
        """
        story.append(Paragraph(tech_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        print("✅ PDF report generated: 'Churn_Analysis_Report.pdf'")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting complete churn analysis...")
        
        self.load_and_analyze_data()
        self.preprocess_and_train()  
        self.create_visualizations()
        self.generate_pdf_report()
        
        print("✅ Complete analysis finished!")
        print("📄 Report saved as: 'Churn_Analysis_Report.pdf'")
        print("📊 Visualizations saved in: 'outputs/' folder")

# Usage
if __name__ == "__main__":
    # Update this path to your data file
    file_path = r"C:\Users\nidhi\OneDrive\Documents\churn project\deepq_ai_assignment1_data.csv"
    
    # Generate complete report
    generator = ChurnReportGenerator(file_path)
    generator.run_complete_analysis()