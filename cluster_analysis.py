import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import japanize_matplotlib
import warnings
warnings.filterwarnings('ignore')

# データ読み込みと前処理
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

# 業種と地域の英語マッピング
industry_mapping = {
    '製造': 'Manufacturing', 'IT': 'IT', '小売': 'Retail', '飲食': 'Food_Service',
    '建設': 'Construction', 'サービス': 'Services', '物流': 'Logistics',
    '金融': 'Finance', '教育': 'Education', '医療': 'Healthcare'
}

region_mapping = {
    '関東': 'Kanto', '関西': 'Kansai', '中部': 'Chubu',
    '九州': 'Kyushu', '中国': 'Chugoku'
}

# データ変換
df = df.rename(columns=column_mapping)
df['Industry'] = df['Industry'].map(industry_mapping)
df['Region'] = df['Region'].map(region_mapping)
df['Company_Age'] = 2024 - df['Founded_Year']

print("=== Cluster Analysis for Revenue Prediction ===")
print(f"Target variable: Recent_Amount_10K_JPY")
print(f"Data shape: {df.shape}")

# クラスタリング用の特徴量選択
# 直近売上を予測するための説明変数を選択
clustering_features = [
    'Revenue_M_JPY',          # 売上規模
    'Employees',              # 組織規模
    'Past_Transactions',      # 取引実績
    'Unit_Price_10K_JPY',     # 取引単価
    'Customer_Satisfaction',  # 顧客満足度
    'Company_Age'            # 企業年数
]

# 業種をダミー変数化
industry_dummies = pd.get_dummies(df['Industry'], prefix='Industry')
region_dummies = pd.get_dummies(df['Region'], prefix='Region')

# 特徴量データフレーム作成
X = df[clustering_features].copy()
X = pd.concat([X, industry_dummies, region_dummies], axis=1)

print(f"Clustering features: {X.columns.tolist()}")
print(f"Feature matrix shape: {X.shape}")

# データ標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 最適クラスター数の決定
inertias = []
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

# エルボー法とシルエット分析のプロット
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# エルボー法
ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method for Optimal K')
ax1.grid(True)

# シルエット分析
ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True)

plt.tight_layout()
plt.savefig('optimal_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# クラスター数を4に固定
optimal_k = 4
print(f"\nForced number of clusters: {optimal_k}")
if optimal_k-2 < len(silhouette_scores):
    print(f"Silhouette score for K=4: {silhouette_scores[optimal_k-2]:.3f}")
else:
    print("Calculating silhouette score for K=4...")
    kmeans_temp = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    temp_labels = kmeans_temp.fit_predict(X_scaled)
    temp_silhouette = silhouette_score(X_scaled, temp_labels)
    print(f"Silhouette score for K=4: {temp_silhouette:.3f}")

# 最適なクラスター数でクラスタリング実行
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"\nCluster distribution:")
print(df['Cluster'].value_counts().sort_index())

# クラスター別統計分析
print("\n=== Cluster Characteristics ===")
cluster_stats = df.groupby('Cluster').agg({
    'Recent_Amount_10K_JPY': ['mean', 'std', 'min', 'max'],
    'Revenue_M_JPY': 'mean',
    'Employees': 'mean',
    'Past_Transactions': 'mean',
    'Unit_Price_10K_JPY': 'mean',
    'Customer_Satisfaction': 'mean',
    'Company_Age': 'mean'
}).round(2)

print(cluster_stats)

# 各クラスターの企業一覧
print("\n=== Companies by Cluster ===")
for cluster in sorted(df['Cluster'].unique()):
    companies = df[df['Cluster'] == cluster]['Company'].tolist()
    avg_recent_amount = df[df['Cluster'] == cluster]['Recent_Amount_10K_JPY'].mean()
    print(f"Cluster {cluster} (Avg Recent Amount: {avg_recent_amount:.1f}): {companies}")

# クラスター別業種分布
print("\n=== Industry Distribution by Cluster ===")
cluster_industry = pd.crosstab(df['Cluster'], df['Industry'])
print(cluster_industry)

# クラスター別地域分布
print("\n=== Region Distribution by Cluster ===")
cluster_region = pd.crosstab(df['Cluster'], df['Region'])
print(cluster_region)

# PCAによる2次元可視化
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# PCA軸の解釈
print(f"\n=== PCA Axis Interpretation ===")
print(f"PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
print(f"PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")

# 各軸への寄与が大きい特徴量
feature_names = X.columns
pc1_loadings = pca.components_[0]
pc2_loadings = pca.components_[1]

# PC1の主要構成要素
pc1_top = [(feature_names[i], abs(pc1_loadings[i])) for i in range(len(feature_names))]
pc1_top.sort(key=lambda x: x[1], reverse=True)
print(f"\nPC1 (First Axis) - Top contributors:")
for feature, loading in pc1_top[:5]:
    sign = "+" if pc1_loadings[list(feature_names).index(feature)] > 0 else "-"
    print(f"  {sign} {feature}: {loading:.3f}")

# PC2の主要構成要素
pc2_top = [(feature_names[i], abs(pc2_loadings[i])) for i in range(len(feature_names))]
pc2_top.sort(key=lambda x: x[1], reverse=True)
print(f"\nPC2 (Second Axis) - Top contributors:")
for feature, loading in pc2_top[:5]:
    sign = "+" if pc2_loadings[list(feature_names).index(feature)] > 0 else "-"
    print(f"  {sign} {feature}: {loading:.3f}")

# 軸の解釈
print(f"\n=== Axis Summary ===")
if any("Revenue" in f or "Employees" in f or "Past_Transactions" in f for f, _ in pc1_top[:3]):
    print("PC1: Enterprise Scale (売上・従業員・取引規模)")
else:
    print("PC1: Mixed Business Characteristics")

if any("Customer_Satisfaction" in f or "Company_Age" in f or "Industry" in f for f, _ in pc2_top[:3]):
    print("PC2: Business Maturity & Quality (企業年数・満足度・業種特性)")
else:
    print("PC2: Secondary Business Features")

# クラスター可視化
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i in range(optimal_k):
    cluster_mask = df['Cluster'] == i
    plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1],
                c=colors[i], label=f'Cluster {i}', alpha=0.7, s=100)

    # 企業名を表示
    for idx in df[cluster_mask].index:
        plt.annotate(df.loc[idx, 'Company'],
                    (X_pca[idx, 0], X_pca[idx, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Cluster Visualization (PCA)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_pca_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# 直近売上予測のためのクラスター分析結果
plt.figure(figsize=(15, 10))

# 1. クラスター別直近売上分布
plt.subplot(2, 3, 1)
df.boxplot(column='Recent_Amount_10K_JPY', by='Cluster', ax=plt.gca())
plt.title('Recent Amount by Cluster')
plt.suptitle('')

# 2. クラスター別売上規模
plt.subplot(2, 3, 2)
df.boxplot(column='Revenue_M_JPY', by='Cluster', ax=plt.gca())
plt.title('Revenue by Cluster')
plt.suptitle('')

# 3. クラスター別取引単価
plt.subplot(2, 3, 3)
df.boxplot(column='Unit_Price_10K_JPY', by='Cluster', ax=plt.gca())
plt.title('Unit Price by Cluster')
plt.suptitle('')

# 4. クラスター別顧客満足度
plt.subplot(2, 3, 4)
df.boxplot(column='Customer_Satisfaction', by='Cluster', ax=plt.gca())
plt.title('Customer Satisfaction by Cluster')
plt.suptitle('')

# 5. クラスター別企業年数
plt.subplot(2, 3, 5)
df.boxplot(column='Company_Age', by='Cluster', ax=plt.gca())
plt.title('Company Age by Cluster')
plt.suptitle('')

# 6. クラスター別取引回数
plt.subplot(2, 3, 6)
df.boxplot(column='Past_Transactions', by='Cluster', ax=plt.gca())
plt.title('Past Transactions by Cluster')
plt.suptitle('')

plt.tight_layout()
plt.savefig('cluster_characteristics.png', dpi=150, bbox_inches='tight')
plt.show()

# 売上予測モデルの評価
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# クラスターを特徴量として追加した売上予測
X_with_cluster = X.copy()
X_with_cluster['Cluster'] = df['Cluster']

# クラスターなしの予測
rf_without_cluster = RandomForestRegressor(random_state=42)
scores_without = cross_val_score(rf_without_cluster, X_scaled, df['Recent_Amount_10K_JPY'], cv=5)

# クラスターありの予測
X_with_cluster_scaled = scaler.fit_transform(X_with_cluster)
rf_with_cluster = RandomForestRegressor(random_state=42)
scores_with = cross_val_score(rf_with_cluster, X_with_cluster_scaled, df['Recent_Amount_10K_JPY'], cv=5)

print(f"\n=== Revenue Prediction Model Evaluation ===")
print(f"Without Cluster Features - CV Score: {scores_without.mean():.3f} (+/- {scores_without.std() * 2:.3f})")
print(f"With Cluster Features - CV Score: {scores_with.mean():.3f} (+/- {scores_with.std() * 2:.3f})")
print(f"Improvement: {scores_with.mean() - scores_without.mean():.3f}")

# 特徴量重要度（クラスターありモデル）
rf_with_cluster.fit(X_with_cluster_scaled, df['Recent_Amount_10K_JPY'])
feature_importance = pd.DataFrame({
    'feature': X_with_cluster.columns,
    'importance': rf_with_cluster.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n=== Feature Importance for Revenue Prediction ===")
print(feature_importance.head(10))

# 特徴量重要度の可視化
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Features for Recent Amount Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== Cluster Analysis Complete ===")
print("Generated files:")
print("- optimal_clusters.png: Elbow method and silhouette analysis")
print("- cluster_pca_visualization.png: PCA visualization of clusters")
print("- cluster_characteristics.png: Cluster characteristics comparison")
print("- feature_importance.png: Feature importance for revenue prediction")