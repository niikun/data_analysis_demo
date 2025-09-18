import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import warnings
warnings.filterwarnings('ignore')

# CSVファイルを読み込み
df = pd.read_csv('company_cluster_demo.csv')

# 英語カラム名にマッピング
column_mapping = {
    '企業名': 'Company',
    '売上(百万円)': 'Revenue_M_JPY',
    '業種': 'Industry',
    '従業員数': 'Employees',
    '過去取引回数': 'Past_Transactions',
    '取引単価(万円)': 'Unit_Price_10K_JPY',
    '直近取引金額(万円)': 'Recent_Amount_10K_JPY',
    '設立年': 'Founded_Year',
    '顧客満足度': 'Customer_Satisfaction',
    '地域': 'Region'
}

# 業種の英語マッピング
industry_mapping = {
    '製造': 'Manufacturing',
    'IT': 'IT',
    '小売': 'Retail',
    '飲食': 'Food_Service',
    '建設': 'Construction',
    'サービス': 'Services',
    '物流': 'Logistics',
    '金融': 'Finance',
    '教育': 'Education',
    '医療': 'Healthcare'
}

# 地域の英語マッピング
region_mapping = {
    '関東': 'Kanto',
    '関西': 'Kansai',
    '中部': 'Chubu',
    '九州': 'Kyushu',
    '中国': 'Chugoku'
}

# カラム名を英語に変更
df = df.rename(columns=column_mapping)
# 業種を英語に変更
df['Industry'] = df['Industry'].map(industry_mapping)
# 地域を英語に変更
df['Region'] = df['Region'].map(region_mapping)

# 企業年数を計算（2024年基準）
df['Company_Age'] = 2024 - df['Founded_Year']

# データの基本情報を確認
print("=== Basic Data Information ===")
print(f"Data shape: {df.shape}")
print("\n=== Column names ===")
print(df.columns.tolist())
print("\n=== Data types ===")
print(df.dtypes)
print("\n=== First 5 rows ===")
print(df.head())

# 基本統計量
print("\n=== Basic Statistics ===")
print(df.describe())

# 欠損値の確認
print("\n=== Missing Values ===")
print(df.isnull().sum())

# 業種別の分布
print("\n=== Company Count by Industry ===")
print(df['Industry'].value_counts())

# 相関関係の分析
numeric_cols = ['Revenue_M_JPY', 'Employees', 'Past_Transactions', 'Unit_Price_10K_JPY', 'Recent_Amount_10K_JPY', 'Founded_Year', 'Customer_Satisfaction', 'Company_Age']
correlation_matrix = df[numeric_cols].corr()

print("\n=== Correlation Matrix ===")
print(correlation_matrix)

# 可視化の設定
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# 1. 数値変数の分布
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=10, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# 空のサブプロットを非表示
axes[8].set_visible(False)

plt.tight_layout()
plt.savefig('distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. 相関関係のヒートマップ
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True)
plt.title('Correlation Matrix of Variables')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. 業種別の売上分布
plt.figure(figsize=(12, 6))
df.boxplot(column='Revenue_M_JPY', by='Industry', ax=plt.gca())
plt.title('Revenue Distribution by Industry')
plt.xticks(rotation=45)
plt.suptitle('')
plt.tight_layout()
plt.savefig('revenue_by_industry.png', dpi=150, bbox_inches='tight')
plt.show()

# 4. 散布図マトリックス
from pandas.plotting import scatter_matrix
fig = plt.figure(figsize=(15, 15))
scatter_matrix(df[numeric_cols], alpha=0.6, figsize=(15, 15), diagonal='hist')
plt.tight_layout()
plt.savefig('scatter_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# 5. 業種別の平均値比較
main_numeric_cols = ['Revenue_M_JPY', 'Employees', 'Past_Transactions', 'Unit_Price_10K_JPY', 'Recent_Amount_10K_JPY']
industry_stats = df.groupby('Industry')[main_numeric_cols].mean()
print("\n=== Average Values by Industry ===")
print(industry_stats)

# 6. 地域別の統計
print("\n=== Company Count by Region ===")
print(df['Region'].value_counts())

region_stats = df.groupby('Region')[main_numeric_cols].mean()
print("\n=== Average Values by Region ===")
print(region_stats)

# 7. 業種別の統計をプロット
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, col in enumerate(main_numeric_cols):
    industry_stats[col].plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Average {col} by Industry')
    axes[i].set_xlabel('Industry')
    axes[i].set_ylabel(col)
    axes[i].tick_params(axis='x', rotation=45)

axes[5].set_visible(False)
plt.tight_layout()
plt.savefig('industry_averages.png', dpi=150, bbox_inches='tight')
plt.show()

# 8. 顧客満足度と売上の関係
plt.figure(figsize=(10, 6))
plt.scatter(df['Customer_Satisfaction'], df['Revenue_M_JPY'], alpha=0.7, s=100)
plt.xlabel('Customer Satisfaction')
plt.ylabel('Revenue (M JPY)')
plt.title('Customer Satisfaction vs Revenue')
for i, company in enumerate(df['Company']):
    plt.annotate(company, (df['Customer_Satisfaction'].iloc[i], df['Revenue_M_JPY'].iloc[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
plt.tight_layout()
plt.savefig('satisfaction_vs_revenue.png', dpi=150, bbox_inches='tight')
plt.show()

# 9. 企業年数と各指標の関係
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

metrics = ['Revenue_M_JPY', 'Employees', 'Customer_Satisfaction', 'Past_Transactions']
for i, metric in enumerate(metrics):
    axes[i].scatter(df['Company_Age'], df[metric], alpha=0.7, s=100)
    axes[i].set_xlabel('Company Age (years)')
    axes[i].set_ylabel(metric)
    axes[i].set_title(f'Company Age vs {metric}')

plt.tight_layout()
plt.savefig('company_age_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== EDA Complete ===")
print("Generated files:")
print("- distributions.png: Distribution of each variable")
print("- correlation_heatmap.png: Correlation heatmap")
print("- revenue_by_industry.png: Revenue distribution by industry")
print("- scatter_matrix.png: Scatter plot matrix")
print("- industry_averages.png: Average values by industry")
print("- satisfaction_vs_revenue.png: Customer satisfaction vs revenue")
print("- company_age_analysis.png: Company age analysis")