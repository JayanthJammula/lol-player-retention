import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
from sklearn.utils import resample

# 1. Load Data
file_path = './player_features_manual_x.csv'
df = pd.read_csv(file_path)

# Check for missing columns
required_columns = ['total_game_duration', 'win_loss_ratio', 'unique_champions', 'kill_death_ratio', 
                    'avg_champion_level', 'avg_gold_usage', 'avg_daily_streak', 'daily_engagement', 
                    'session_frequency', 'total_active_days']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Missing columns from DataFrame: {missing_columns}")

# 2. Descriptive Statistics
print("Descriptive Statistics:\n")
available_columns = [col for col in required_columns if col in df.columns]
descriptive_stats = df[available_columns].describe()
print(descriptive_stats)

# 3. Visualizations
# Histograms
plt.figure(figsize=(20, 15))
for i, col in enumerate(available_columns):
    plt.subplot(5, 2, i + 1)
    sns.histplot(df[col], kde=True, color='blue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Box Plots
plt.figure(figsize=(20, 15))
for i, col in enumerate(available_columns):
    plt.subplot(5, 2, i + 1)
    sns.boxplot(x=df[col], color='green')
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
if len(available_columns) > 1:
    correlation_matrix = df[available_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

# Pairplot
if 'churn' in df.columns:
    sns.pairplot(df[available_columns + ['churn']], hue='churn')
    plt.show()

# 4. Inferential Statistics
# Split Churners and Non-Churners
if 'churn' in df.columns:
    df_churn = df[df['churn'] == 1]
    df_non_churn = df[df['churn'] == 0]

    # t-test to compare churners and non-churners
    print("T-Tests between Churners and Non-Churners:\n")
    ttest_results = {}
    for col in available_columns:
        t_stat, p_val = ttest_ind(df_churn[col], df_non_churn[col], nan_policy='omit')
        ttest_results[col] = {'t_stat': t_stat, 'p_value': p_val}
        print(f"{col}: t-stat = {t_stat:.4f}, p-value = {p_val:.4f}")

# Chi-Square Test for categorical variables (if any)
if 'playstyle' in df.columns:
    print("\nChi-Square Test for Playstyle vs Churn:\n")
    contingency_table = pd.crosstab(df['playstyle'], df['churn'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square Statistic = {chi2:.4f}, p-value = {p:.4f}")

# ANOVA to assess the impact of 'time_of_day' on 'session_frequency'
if 'time_of_day' in df.columns and 'session_frequency' in df.columns:
    print("\nANOVA Test for Time of Day Impact on Session Frequency:\n")
    unique_times = df['time_of_day'].unique()
    groups = [df[df['time_of_day'] == time]['session_frequency'] for time in unique_times]
    f_stat, p_val = f_oneway(*groups)
    print(f"F-statistic = {f_stat:.4f}, p-value = {p_val:.4f}")

# 5. Outlier Detection
# Identify outliers using the IQR method
print("\nOutlier Detection:\n")
outlier_detection_results = {}
for col in available_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_detection_results[col] = len(outliers)
    print(f"{col}: {len(outliers)} outliers detected.")

# 6. Deep Learning Model (LSTM RNN) with Batch Normalization, Reduced L2, and Bidirectional LSTM
X = df.drop(columns=['puuid', 'churn'])
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential([
    Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.005))),
    BatchNormalization(),
    Dropout(0.5),
    LSTM(32, return_sequences=True, kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.5),
    LSTM(16, return_sequences=False, kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Predictions
y_pred = (model.predict(X_test_reshaped) > 0.5).astype("int32").flatten()
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Churn', 'Churn'], yticklabels=['Non-Churn', 'Churn'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Bootstrap Confidence Intervals
def compute_bootstrap_ci(y_true, y_pred, metric, n_resamples=1000, alpha=0.05):
    np.random.seed(42)
    scores = []
    for _ in range(n_resamples):
        indices = resample(range(len(y_true)), replace=True)
        y_true_sample = y_true.iloc[indices]
        y_pred_sample = y_pred[indices]
        score = metric(y_true_sample, y_pred_sample)
        scores.append(score)
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return lower, upper


accuracy_ci = compute_bootstrap_ci(y_test, y_pred, accuracy_score)
precision_ci = compute_bootstrap_ci(y_test, y_pred, precision_score)
recall_ci = compute_bootstrap_ci(y_test, y_pred, recall_score)
f1_ci = compute_bootstrap_ci(y_test, y_pred, f1_score)

performance_metrics = {
    'accuracy': {'value': accuracy, 'ci': accuracy_ci},
    'precision': {'value': precision, 'ci': precision_ci},
    'recall': {'value': recall, 'ci': recall_ci},
    'f1_score': {'value': f1, 'ci': f1_ci}
}

print("Performance Metrics with Confidence Intervals:")
print(performance_metrics)

# 6. Visualize Training History
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Training and Validation Accuracy with BatchNorm, Dropout=0.3, L2=0.005')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training and Validation Loss with BatchNorm, Dropout=0.3, L2=0.005')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()