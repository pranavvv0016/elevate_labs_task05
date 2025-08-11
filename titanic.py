# ------------------------------------
# TASK 5 - Exploratory Data Analysis (EDA)
# Titanic Dataset
# ------------------------------------

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: for better plot visuals
sns.set(style="whitegrid", palette="pastel")
plt.rcParams["figure.figsize"] = (10, 6)

# 2. Load Dataset
df = pd.read_csv("titanic.csv")  # Make sure train.csv is in the same folder

# 3. Basic Inspection
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 4. Handle Missing Values
# Fill Age with median, Embarked with mode
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
df.drop(columns=["Cabin"])

# 5. Univariate Analysis
# Numerical Features
numeric_cols = ["Age", "Fare"]
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Categorical Features
categorical_cols = ["Sex", "Pclass", "Embarked", "Survived"]
for col in categorical_cols:
    plt.figure()
    sns.countplot(x=col, data=df)
    plt.title(f"Count of {col}")
    plt.show()

# 6. Bivariate Analysis
# Age vs Survived
plt.figure()
sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age vs Survival")
plt.show()

# Sex vs Survived
plt.figure()
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival Count by Sex")
plt.show()

# Pclass vs Survived
plt.figure()
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival Count by Passenger Class")
plt.show()

# Fare vs Survived
plt.figure()
sns.histplot(df[df["Survived"]==1]["Fare"], color="green", label="Survived", kde=True)
sns.histplot(df[df["Survived"]==0]["Fare"], color="red", label="Not Survived", kde=True)
plt.legend()
plt.title("Fare Distribution by Survival")
plt.show()

# 7. Correlation Heatmap
corr = df.corr(numeric_only=True)
plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 8. Pairplot (selected features)
sns.pairplot(df[["Survived", "Pclass", "Age", "Fare"]], hue="Survived")
plt.show()

# 9. Summary of Insights
print("\n--- Summary of Insights ---")
print("1. Most passengers were in 3rd class.")
print("2. Majority of passengers were male, but survival rate was higher for females.")
print("3. Younger passengers had slightly better survival chances.")
print("4. Higher fares were associated with higher survival rates.")
print("5. Pclass is strongly correlated with survival — 1st class had better survival rates.")
print("6. Age distribution is slightly right-skewed.")
print("7. Fare distribution has a long tail with outliers (very high fares).")
