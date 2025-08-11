import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import os

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("titanic.csv")

# ==============================
# PDF SETUP
# ==============================
pdf_file = "titanic_eda_report.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
styles = getSampleStyleSheet()
story = []

def save_plot(fig, filename):
    """Save Matplotlib figure and return file path."""
    path = os.path.join(os.getcwd(), filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

# ==============================
# BASIC INFO
# ==============================
story.append(Paragraph("Titanic Dataset EDA Report", styles["Title"]))
story.append(Spacer(1, 12))

story.append(Paragraph(f"Total Rows: {df.shape[0]}", styles["Normal"]))
story.append(Paragraph(f"Total Columns: {df.shape[1]}", styles["Normal"]))
story.append(Spacer(1, 12))

story.append(Paragraph("Columns:", styles["Heading2"]))
story.append(Paragraph(", ".join(df.columns), styles["Normal"]))

story.append(Spacer(1, 12))
story.append(Paragraph("Missing Values:", styles["Heading2"]))
missing = df.isnull().sum().to_frame("Missing Count")
story.append(Paragraph(missing.to_html(), styles["Normal"]))

story.append(Spacer(1, 12))
story.append(Paragraph("Summary Statistics:", styles["Heading2"]))
story.append(Paragraph(df.describe().round(2).to_html(), styles["Normal"]))

# ==============================
# DATA CLEANING
# ==============================
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df.drop(columns=["Cabin"], inplace=True, errors="ignore")

# ==============================
# UNIVARIATE ANALYSIS - NUMERICAL
# ==============================
numeric_cols = ["Age", "Fare"]
for col in numeric_cols:
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    story.append(Paragraph(f"Distribution of {col}", styles["Heading3"]))
    story.append(RLImage(save_plot(fig, f"{col}_dist.png"), width=400, height=250))

    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f"Boxplot of {col}")
    story.append(RLImage(save_plot(fig, f"{col}_box.png"), width=400, height=250))

# ==============================
# UNIVARIATE ANALYSIS - CATEGORICAL
# ==============================
categorical_cols = ["Sex", "Pclass", "Embarked", "Survived"]
for col in categorical_cols:
    fig, ax = plt.subplots()
    sns.countplot(x=col, data=df, ax=ax)
    ax.set_title(f"Count of {col}")
    story.append(Paragraph(f"Count of {col}", styles["Heading3"]))
    story.append(RLImage(save_plot(fig, f"{col}_count.png"), width=400, height=250))

# ==============================
# BIVARIATE ANALYSIS
# ==============================
# Age vs Survived
fig, ax = plt.subplots()
sns.boxplot(x="Survived", y="Age", data=df, ax=ax)
ax.set_title("Age vs Survival")
story.append(Paragraph("Age vs Survival", styles["Heading3"]))
story.append(RLImage(save_plot(fig, "age_vs_survival.png"), width=400, height=250))

# Sex vs Survived
fig, ax = plt.subplots()
sns.countplot(x="Sex", hue="Survived", data=df, ax=ax)
ax.set_title("Survival Count by Sex")
story.append(RLImage(save_plot(fig, "sex_vs_survival.png"), width=400, height=250))

# Pclass vs Survived
fig, ax = plt.subplots()
sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax)
ax.set_title("Survival Count by Passenger Class")
story.append(RLImage(save_plot(fig, "pclass_vs_survival.png"), width=400, height=250))

# Fare vs Survived (MISSING BEFORE - NOW ADDED)
fig, ax = plt.subplots()
sns.histplot(df[df["Survived"]==1]["Fare"], color="green", label="Survived", kde=True, ax=ax)
sns.histplot(df[df["Survived"]==0]["Fare"], color="red", label="Not Survived", kde=True, ax=ax)
ax.set_title("Fare Distribution by Survival")
ax.legend()
story.append(Paragraph("Fare vs Survival", styles["Heading3"]))
story.append(RLImage(save_plot(fig, "fare_vs_survival.png"), width=400, height=250))

# ==============================
# CORRELATION HEATMAP
# ==============================
fig, ax = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
story.append(Paragraph("Correlation Heatmap", styles["Heading3"]))
story.append(RLImage(save_plot(fig, "heatmap.png"), width=400, height=250))

# ==============================
# PAIRPLOT (Cleaned)
# ==============================
pairplot_df = df[["Survived", "Pclass", "Age", "Fare"]].apply(pd.to_numeric, errors="coerce").dropna()
sns.pairplot(pairplot_df, hue="Survived").savefig("pairplot.png")
story.append(Paragraph("Pairplot of Selected Features", styles["Heading3"]))
story.append(RLImage("pairplot.png", width=400, height=400))

# ==============================
# SUMMARY OF INSIGHTS
# ==============================
summary_text = """
1. Most passengers were in 3rd class.<br/>
2. Majority of passengers were male, but survival rate was higher for females.<br/>
3. Younger passengers had slightly better survival chances.<br/>
4. Higher fares were associated with higher survival rates.<br/>
5. Pclass is strongly correlated with survival — 1st class had better survival rates.<br/>
6. Age distribution is slightly right-skewed.<br/>
7. Fare distribution has a long tail with outliers (very high fares).
"""
story.append(Spacer(1, 12))
story.append(Paragraph("Summary of Insights", styles["Heading2"]))
story.append(Paragraph(summary_text, styles["Normal"]))

# ==============================
# BUILD PDF
# ==============================
doc.build(story)
print(f"✅ PDF report saved as {pdf_file}")
