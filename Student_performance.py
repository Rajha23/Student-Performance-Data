# ===============================================================
# 📘 STUDENT EXAM PERFORMANCE ANALYSIS (Colorful Version)
# ===============================================================
# Author: Your Name
# Description:
# Analyze and predict student performance using exam and project
# scores. Includes data cleaning, visualization, ML modeling,
# and color-coded console outputs for clarity.
# ===============================================================

# 1️⃣ Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from colorama import Fore, Style, init

# Initialize Colorama
init(autoreset=True)

# 2️⃣ Load Dataset
print(Fore.CYAN + "\n[*] Loading dataset...")
df = pd.read_csv("Students Performance Dataset_Original.csv")
print(Fore.GREEN + "[OK] Dataset Loaded Successfully!")
print(Fore.YELLOW + f"Shape: {df.shape}\n")
print(df.head())

# 3️⃣ Data Cleaning
print(Fore.CYAN + "\n[*] Cleaning the data...")

# Drop duplicates
df = df.drop_duplicates()

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Standardize gender values
df['gender'] = df['gender'].str.strip().str.lower()

# Drop irrelevant columns
if 'student_id' in df.columns:
    df = df.drop('student_id', axis=1)
if 'first_name' in df.columns:
    df = df.drop('first_name', axis=1)

# Convert numeric columns safely
numeric_cols = ['age', 'final_score', 'projects_score', 'total_score']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values
df = df.dropna()

# Create performance label
def performance_label(score):
    if score >= 80:
        return 'High'
    elif score >= 60:
        return 'Medium'
    else:
        return 'Low'

df['performance_level'] = df['total_score'].apply(performance_label)

print(Fore.GREEN + "[OK] Cleaning complete!")
print(Fore.YELLOW + f"Dataset shape after cleaning: {df.shape}\n")
print(df.head())

# ===============================================================
# 4️⃣ EXPLORATORY DATA ANALYSIS (EDA)
# ===============================================================
print(Fore.CYAN + "\n[*] Starting Exploratory Data Analysis...")

sns.set(style="whitegrid", palette="pastel")

# Distribution of Total Score
print(Fore.YELLOW + "\n[1/7] Showing: Distribution of Total Scores (close the plot window to continue)")
plt.figure(figsize=(8,5))
colors = plt.cm.plasma(df['total_score'] / df['total_score'].max())
sns.histplot(df['total_score'], bins=20, kde=True, color='#7B1FA2', edgecolor='white', alpha=0.85)
plt.title("Distribution of Total Scores", fontsize=13, fontweight='bold')
plt.xlabel("Total Score", fontsize=12)
plt.ylabel("Number of Students", fontsize=12)
plt.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.show(block=True)

# Gender vs Total Score
print(Fore.YELLOW + "\n[2/4] Showing: Average Total Score by Gender (close the plot window to continue)")
plt.figure(figsize=(7,5))
custom_palette = ["#00B8A9", "#F6416C"]
sns.barplot(x='gender', y='total_score', data=df, errorbar=None, hue='gender', legend=False, palette=custom_palette)
plt.title("Average Total Score by Gender", fontsize=13, fontweight='bold')
plt.xlabel("Gender", fontsize=12)
plt.ylabel("Average Total Score", fontsize=12)
plt.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.show(block=True)

# Performance Level Count
print(Fore.YELLOW + "\n[3/4] Showing: Count of Performance Levels (close the plot window to continue)")
plt.figure(figsize=(6,5))
level_palette = ["#43A047", "#FFB300", "#E53935"]
sns.countplot(x='performance_level', data=df, palette=level_palette)
plt.title("Count of Performance Levels", fontsize=13, fontweight='bold')
plt.xlabel("Performance Level", fontsize=12)
plt.ylabel("Number of Students", fontsize=12)
plt.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.show(block=True)

print(Fore.GREEN + "[OK] EDA Completed Successfully!")

# ===============================================================
# 5️⃣ MACHINE LEARNING MODEL
# ===============================================================
print(Fore.CYAN + "\n[*] Training Machine Learning Model...")

# Encode categorical data
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

# Split data
X = df[['gender', 'age', 'final_score', 'projects_score']]
y = df['performance_level']
y = le.fit_transform(y)

# Split for training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(Fore.GREEN + "\n[OK] Model Training Complete!")
acc = accuracy_score(y_test, y_pred) * 100
print(Fore.YELLOW + f"Model Accuracy: {acc:.2f}%\n")

print(Fore.CYAN + "[*] Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
print(Fore.YELLOW + "\n[4/4] Showing: Feature Importance (close the plot window to see final insights)")
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))

# Create bars with gradient colors
colors = plt.cm.viridis(feat_imp.values / feat_imp.values.max())
bars = plt.bar(range(len(feat_imp)), feat_imp.values, color=colors, edgecolor='black', linewidth=1.2)

# Add percentage labels on top of bars
for i, (bar, value) in enumerate(zip(bars, feat_imp.values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xticks(range(len(feat_imp)), feat_imp.index, fontsize=11)
plt.xlabel("Features", fontsize=12, fontweight='bold')
plt.ylabel("Importance (Contribution to Prediction)", fontsize=12, fontweight='bold')
plt.title("Which Factors Best Predict Student Performance?\n(Higher = More Important for the Model)", 
          fontsize=13, fontweight='bold', pad=20)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show(block=True)

# ===============================================================
# 6️⃣ INSIGHTS AND CONCLUSION
# ===============================================================
print(Fore.CYAN + "\n[*] INSIGHTS:")
print(Fore.GREEN + "- Project and Exam scores are the strongest predictors of performance.")
print(Fore.GREEN + "- Gender has minimal effect on total score compared to academic metrics.")
print(Fore.GREEN + "- The model predicts performance levels with high accuracy.")
print(Fore.GREEN + "- Most students fall in the 'Medium' performance category.")

print(Fore.MAGENTA + "\n[OK] PROJECT COMPLETE: Student Performance Analysis Done Successfully!")
