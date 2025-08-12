import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load data
file_path = 'biometric_data.csv'
use_sample_data = False
try:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' not found in {os.getcwd()}. Available files: {os.listdir('.')}")
    df = pd.read_csv(file_path)
    print("Data loaded successfully!")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First few rows:\n", df.head())
    print("Heart rate range:", df['heart_rate'].min(), df['heart_rate'].max())
    print("Stress level range:", df['stress_level'].min(), df['stress_level'].max())
    print("Unique heart_rate values:", df['heart_rate'].unique())
    print("Unique stress_level values:", df['stress_level'].unique()[:10], "...")
except FileNotFoundError as e:
    print(e)
    print("Creating sample data as fallback...")
    use_sample_data = True

# Fallback to sample data if heart_rate is invalid
if not use_sample_data:
    non_zero_hr = df[df['heart_rate'] != 0]['heart_rate']
    if non_zero_hr.empty or non_zero_hr.nunique() <= 1:
        print("Warning: No valid or varied heart_rate values. Using sample data.")
        use_sample_data = True

if use_sample_data:
    df = pd.DataFrame({
        'content_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        'heart_rate': [75, 76, 80, 82, 70, 71, 78, 79, 73, 74, 81, 80],
        'stress_level': [0.3, 0.4, 0.5, 0.6, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.6, 0.5],
        'is_human': [True, True, False, False, True, True, False, False, True, True, False, False],
        'timestamp': ['2023-10-01 10:00:00', '2023-10-01 10:00:01', 
                      '2023-10-01 10:00:02', '2023-10-01 10:00:03', 
                      '2023-10-01 10:00:04', '2023-10-01 10:00:05',
                      '2023-10-01 10:00:06', '2023-10-01 10:00:07',
                      '2023-10-01 10:00:08', '2023-10-01 10:00:09',
                      '2023-10-01 10:00:10', '2023-10-01 10:00:11']
    })
    df.to_csv('biometric_data_sample.csv', index=False)
    print("Sample data created and saved as 'biometric_data_sample.csv'")
    print("Sample data shape:", df.shape)
    print("Sample data first few rows:\n", df.head())

# Check if df is defined and not empty
if 'df' not in locals() or df.empty:
    print("Error: DataFrame 'df' is not defined or empty. Stopping execution.")
    exit()

# Convert is_human to boolean and timestamp to datetime
df['is_human'] = df['is_human'].astype(bool)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms' if not use_sample_data else None, errors='coerce')

# Feature engineering
df['heart_rate_change'] = df['heart_rate'].diff()
df['stress_change'] = df['stress_level'].diff()
df['heart_rate_stability'] = df['heart_rate'].rolling(window=5).std()
print("Shape after feature engineering:", df.shape)

# Handle invalid heart_rate values (only for original data)
if not use_sample_data:
    non_zero_hr = df[df['heart_rate'] != 0]['heart_rate']
    if non_zero_hr.empty:
        mean_hr = 70  # Default to typical resting heart rate
    else:
        mean_hr = non_zero_hr.mean()
    print(f"Mean non-zero heart_rate: {mean_hr:.2f}")
    df['heart_rate'] = df['heart_rate'].replace(0, mean_hr)
print("Rows with valid heart_rate:", df['heart_rate'].notna().sum())

# Apply filters
df = df[(df['heart_rate'] >= 50) & (df['heart_rate'] < 150)]
print("Shape after heart rate filter:", df.shape)
df = df[(df['stress_level'] >= 0) & (df['stress_level'] <= 1)]
print("Shape after stress level filter:", df.shape)

if df.empty:
    print("Error: No data remains after filtering. Using sample data.")
    df = pd.read_csv('biometric_data_sample.csv')
    df['is_human'] = df['is_human'].astype(bool)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['heart_rate_change'] = df['heart_rate'].diff()
    df['stress_change'] = df['stress_level'].diff()
    df['heart_rate_stability'] = df['heart_rate'].rolling(window=5).std()
    print("Sample data shape after reload:", df.shape)

# Aggregate features
print("Unique content_ids:", df['content_id'].nunique())
features = df.groupby('content_id').agg({
    'heart_rate': ['mean', 'std', 'max', 'min'],
    'stress_level': ['mean', 'std', 'max'],
    'heart_rate_change': ['mean', 'std'],
    'stress_change': ['mean', 'std'],
    'heart_rate_stability': ['mean'],
    'is_human': 'first'
}).reset_index()

# Flatten column names
features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in features.columns]
print("Shape after aggregation:", features.shape)

# Prepare ML data
if features.empty or len(features) < 4:
    print("Error: Too few samples for training. Need at least 4 samples.")
    exit()

X = features.drop(['content_id', 'is_human_first'], axis=1)
y = features['is_human_first']

# Handle NaN values
X = X.fillna(X.mean())
print("X shape:", X.shape, "y shape:", y.shape)

print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
print(f"Human content: {sum(y)} samples")
print(f"AI content: {len(y) - sum(y)} samples")

# Train model with cross-validation
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=min(3, len(y)), scoring='accuracy')  # Adjust cv based on sample size
y_pred = cross_val_predict(model, X, y, cv=min(3, len(y)))

print(f"\n🎯 Cross-Validation Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"Baseline (random guessing): 50%")
print(f"Improvement over baseline: +{(cv_scores.mean()-0.5)*100:.1f} percentage points")

# Fit model for feature importance
model.fit(X, y)
print("\n📊 Feature Importance (what gives away AI content in YOUR biology):")
feature_importance = list(zip(X.columns, model.feature_importances_))
feature_importance.sort(key=lambda x: x[1], reverse=True)
for feature, importance in feature_importance[:5]:
    print(f"  {feature}: {importance:.3f}")

# Detailed classification report
print("\n📋 Detailed Results:")
print(classification_report(y, y_pred, target_names=['AI Content', 'Human Content'], zero_division=0))

# Save model
import joblib
joblib.dump(model, 'personal_ai_detector.pkl')
print("\n💾 Model saved as 'personal_ai_detector.pkl'")

# Visualization
if not df.empty:
    plt.figure(figsize=(12, 8))

    # Plot 1: Feature importance
    plt.subplot(2, 2, 1)
    features_plot = [f[0] for f in feature_importance[:8]]
    importance_plot = [f[1] for f in feature_importance[:8]]
    plt.barh(features_plot, importance_plot)
    plt.title('Top Features for AI Detection')
    plt.xlabel('Importance')

    # Plot 2: Heart rate distribution
    plt.subplot(2, 2, 2)
    human_hr = df[df['is_human'] == True]['heart_rate']
    ai_hr = df[df['is_human'] == False]['heart_rate']
    if not human_hr.empty or not ai_hr.empty:
        bins = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] if human_hr.nunique() > 1 else [49.5, 50.5]
        plt.hist(human_hr, bins=bins, alpha=0.7, label='Human Content')
        plt.hist(ai_hr, bins=bins, alpha=0.7, label='AI Content')
        plt.title('Heart Rate Response Distribution')
        plt.xlabel('Heart Rate (BPM)')
        plt.legend()

    # Plot 3: Stress level distribution
    plt.subplot(2, 2, 3)
    human_stress = df[df['is_human'] == True]['stress_level']
    ai_stress = df[df['is_human'] == False]['stress_level']
    if not human_stress.empty or not ai_stress.empty:
        plt.hist(human_stress, bins=20, alpha=0.7, label='Human Content')
        plt.hist(ai_stress, bins=20, alpha=0.7, label='AI Content')
        plt.title('Stress Level Response Distribution')
        plt.xlabel('Stress Level (0-1)')
        plt.legend()

    # Plot 4: Timeline of responses
    plt.subplot(2, 2, 4)
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        plt.scatter(df[df['is_human'] == True]['timestamp'], 
                   df[df['is_human'] == True]['heart_rate'], 
                   alpha=0.6, label='Human Content', s=10)
        plt.scatter(df[df['is_human'] == False]['timestamp'], 
                   df[df['is_human'] == False]['heart_rate'], 
                   alpha=0.6, label='AI Content', s=10)
        plt.title('Heart Rate Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Heart Rate (BPM)')
        plt.legend()
        plt.gcf().autofmt_xdate()  # Rotate date labels

    plt.tight_layout()
    plt.savefig('biometric_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n📈 Analysis visualization saved as 'biometric_analysis.png'")
else:
    print("Skipping visualization: No data available.")