import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# 訓練済みモデルとスケーラーを保存するスクリプト
def save_cluster_model():
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

    # クラスタリング用の特徴量選択
    clustering_features = [
        'Revenue_M_JPY', 'Employees', 'Past_Transactions',
        'Unit_Price_10K_JPY', 'Customer_Satisfaction', 'Company_Age'
    ]

    # 業種をダミー変数化
    industry_dummies = pd.get_dummies(df['Industry'], prefix='Industry')
    region_dummies = pd.get_dummies(df['Region'], prefix='Region')

    # 特徴量データフレーム作成
    X = df[clustering_features].copy()
    X = pd.concat([X, industry_dummies, region_dummies], axis=1)

    # データ標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # クラスタリングモデル訓練
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # モデルとスケーラーを保存
    joblib.dump(kmeans, 'cluster_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns.tolist(), 'feature_names.pkl')

    # クラスター特徴量も保存
    df['Cluster'] = cluster_labels
    cluster_stats = df.groupby('Cluster').agg({
        'Recent_Amount_10K_JPY': 'mean',
        'Revenue_M_JPY': 'mean',
        'Employees': 'mean',
        'Company_Age': 'mean',
        'Customer_Satisfaction': 'mean'
    }).round(2)

    joblib.dump(cluster_stats, 'cluster_stats.pkl')

    print("Models saved successfully!")
    print("Cluster characteristics:")
    print(cluster_stats)

    return X.columns.tolist()

if __name__ == "__main__":
    feature_names = save_cluster_model()