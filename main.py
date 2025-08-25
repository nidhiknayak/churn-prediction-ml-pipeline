import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r"C:\Users\nidhi\OneDrive\Documents\churn project\deepq_ai_assignment1_data.csv")

print("=== INITIAL DATA INSPECTION ===")
print(f"Initial shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}")
print(f"First few rows:\n{df.head()}")

# Drop UID
if 'UID' in df.columns:
    df = df.drop(columns=['UID'])
    print(f"Shape after dropping UID: {df.shape}")

print("\n=== TARGET VARIABLE INSPECTION ===")
print(f"Target column unique values: {df['Target_ChurnFlag'].unique()}")
print(f"Target value counts:\n{df['Target_ChurnFlag'].value_counts()}")
print(f"Target data type: {df['Target_ChurnFlag'].dtype}")

# Check if target needs conversion
if df['Target_ChurnFlag'].dtype == 'object':
    print("Target is object type, checking for conversion needs...")
    # Try to see what values we have
    print(f"Sample target values: {df['Target_ChurnFlag'].head(10).tolist()}")
    
    # Convert target to numeric more carefully
    df['Target_ChurnFlag'] = pd.to_numeric(df['Target_ChurnFlag'], errors='coerce')
    print(f"After numeric conversion - NaN count: {df['Target_ChurnFlag'].isna().sum()}")

# Drop rows where target is NaN
initial_rows = len(df)
df = df.dropna(subset=['Target_ChurnFlag'])
print(f"Rows dropped due to target NaN: {initial_rows - len(df)}")
print(f"Shape after dropping target NaN: {df.shape}")

print("\n=== COLUMN TYPE ANALYSIS ===")
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

print("\n=== MISSING VALUES ANALYSIS ===")
missing_counts = df.isnull().sum()
missing_percentages = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_counts.index,
    'Missing_Count': missing_counts.values,
    'Missing_Percentage': missing_percentages.values
}).sort_values('Missing_Percentage', ascending=False)
print("Missing values by column:")
print(missing_df[missing_df['Missing_Count'] > 0])

# Instead of dropping all non-numeric columns, let's encode categorical ones
print("\n=== PREPROCESSING CATEGORICAL VARIABLES ===")
df_processed = df.copy()

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    if col != 'Target_ChurnFlag':  # Don't encode target
        print(f"Encoding column: {col}")
        le = LabelEncoder()
        # Handle missing values by creating a separate category
        df_processed[col] = df_processed[col].fillna('Missing')
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

print(f"Shape after encoding categorical variables: {df_processed.shape}")

# Handle missing values in numeric columns more intelligently
print("\n=== HANDLING MISSING VALUES ===")
numeric_cols_to_fill = df_processed.select_dtypes(include=['number']).columns.tolist()
numeric_cols_to_fill.remove('Target_ChurnFlag')  # Don't fill target

for col in numeric_cols_to_fill:
    missing_count = df_processed[col].isnull().sum()
    if missing_count > 0:
        print(f"Filling {missing_count} missing values in {col} with median")
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())

print(f"Final shape: {df_processed.shape}")
print(f"Any remaining missing values: {df_processed.isnull().sum().sum()}")

# Confirm target distribution
print(f"\nFinal target value counts:\n{df_processed['Target_ChurnFlag'].value_counts()}")

# Prepare data for modeling
X = df_processed.drop('Target_ChurnFlag', axis=1)
y = df_processed['Target_ChurnFlag']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Only proceed with modeling if we have data
if len(X) > 0:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\n=== MODEL EVALUATION ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
else:
    print("ERROR: No data remaining after preprocessing!")
    print("Please check your data file and preprocessing steps.")