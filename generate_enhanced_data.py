import pandas as pd
import numpy as np
import random

# 既存データを読み込み
df_original = pd.read_csv('company_cluster_demo.csv')

# 拡張データ生成用の設定
np.random.seed(42)
random.seed(42)

# 拡張業種リスト（より多様に）
industries = [
    '製造', 'IT', '小売', '飲食', '建設', 'サービス', '物流', '金融', '教育', '医療',
    '不動産', '広告', '農業', '通信', 'エネルギー', '化学', '自動車', '繊維', '印刷', '放送'
]

# 拡張地域リスト（より均等に）
regions = [
    '関東', '関西', '中部', '九州', '中国', '東北', '四国', '北海道', '沖縄'
]

# 企業名パターン
company_types = [
    '商事', 'テック', 'リテール', 'フーズ', '建設', 'マーケ', '物流', '金融', '教育', 'ヘルス',
    'システム', 'ソリューション', 'スーパー', 'レストラン', 'ハウジング', 'コンサル', '運輸', '証券',
    'スクール', 'ホスピタル', 'エナジー', 'ケミカル', 'オート', 'テキスタイル', 'プリント',
    'メディア', 'プロパティ', 'アド', 'ファーム', 'コム', 'ラボ', 'ファクトリー', 'ワークス',
    'グループ', 'ホールディングス', 'トレード', 'インダストリー'
]

company_prefixes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'アルファ', 'ベータ', 'ガンマ',
    'デルタ', 'シグマ', 'オメガ', 'ユニ', 'マルチ', 'トップ', 'エース', 'プライム', 'マックス'
]

def generate_company_data(n_companies=80):
    data = []

    for i in range(n_companies):
        # 企業名生成
        prefix = random.choice(company_prefixes)
        suffix = random.choice(company_types)
        company_name = f"{prefix}{suffix}"

        # 業種（より均等に分散）
        industry = industries[i % len(industries)]

        # 地域（より均等に分散）
        region = regions[i % len(regions)]

        # 企業規模に基づいたパラメータ生成
        # 小規模（30%）、中規模（50%）、大規模（20%）の分布
        size_category = np.random.choice(['small', 'medium', 'large'], p=[0.3, 0.5, 0.2])

        if size_category == 'small':
            revenue = np.random.normal(300, 100)  # 200-500百万円
            employees = np.random.normal(80, 30)  # 30-150人
            past_transactions = np.random.poisson(3) + 1  # 1-8回
            unit_price = np.random.normal(40, 15)  # 20-70万円
        elif size_category == 'medium':
            revenue = np.random.normal(800, 300)  # 400-1500百万円
            employees = np.random.normal(200, 80)  # 100-400人
            past_transactions = np.random.poisson(8) + 3  # 3-15回
            unit_price = np.random.normal(80, 25)  # 40-120万円
        else:  # large
            revenue = np.random.normal(2000, 800)  # 1000-4000百万円
            employees = np.random.normal(500, 200)  # 200-1000人
            past_transactions = np.random.poisson(15) + 5  # 5-30回
            unit_price = np.random.normal(150, 50)  # 80-250万円

        # 値の範囲制限と整数化
        revenue = max(100, int(revenue))
        employees = max(10, int(employees))
        past_transactions = max(1, int(past_transactions))
        unit_price = max(10, int(unit_price))

        # 設立年（多様性を持たせる）
        if np.random.random() < 0.1:  # 10%の企業は非常に古い
            founded_year = np.random.randint(1950, 1980)
        elif np.random.random() < 0.3:  # 30%の企業は比較的新しい
            founded_year = np.random.randint(2010, 2020)
        else:  # 60%の企業は中間
            founded_year = np.random.randint(1980, 2010)

        # 顧客満足度（業種による傾向を加味）
        if industry in ['IT', '教育', '医療']:
            satisfaction = np.random.normal(4.2, 0.4)
        elif industry in ['金融', 'サービス']:
            satisfaction = np.random.normal(3.8, 0.5)
        else:
            satisfaction = np.random.normal(3.4, 0.6)

        satisfaction = np.clip(satisfaction, 1.0, 5.0)
        satisfaction = round(satisfaction, 1)

        # 直近取引金額（過去取引回数と単価に基づく）
        # 一部に外れ値を含める
        if np.random.random() < 0.05:  # 5%の企業は異常に高い取引
            recent_amount = unit_price * past_transactions * np.random.uniform(2.0, 5.0)
        elif np.random.random() < 0.1:  # 10%の企業は取引なし/低い
            recent_amount = unit_price * 0.1 * np.random.uniform(0.1, 0.5)
        else:  # 通常の取引
            recent_amount = unit_price * np.random.uniform(0.5, 2.0)

        recent_amount = max(5, int(recent_amount))

        data.append({
            '企業名': company_name,
            '売上(百万円)': revenue,
            '業種': industry,
            '従業員数': employees,
            '過去取引回数': past_transactions,
            '取引単価(万円)': unit_price,
            '直近取引金額(万円)': recent_amount,
            '設立年': founded_year,
            '顧客満足度': satisfaction,
            '地域': region
        })

    return pd.DataFrame(data)

# 拡張データ生成
df_expanded = generate_company_data(80)

print("=== Generated Enhanced Dataset ===")
print(f"Total companies: {len(df_expanded)}")
print(f"\nIndustry distribution:")
print(df_expanded['業種'].value_counts().sort_index())
print(f"\nRegion distribution:")
print(df_expanded['地域'].value_counts().sort_index())

# 基本統計
print(f"\n=== Basic Statistics ===")
numeric_cols = ['売上(百万円)', '従業員数', '過去取引回数', '取引単価(万円)', '直近取引金額(万円)', '設立年', '顧客満足度']
print(df_expanded[numeric_cols].describe())

# 外れ値とエッジケースの確認
print(f"\n=== Outliers and Edge Cases ===")
print(f"Companies with very high revenue (>3000M): {len(df_expanded[df_expanded['売上(百万円)'] > 3000])}")
print(f"Companies with very low revenue (<150M): {len(df_expanded[df_expanded['売上(百万円)'] < 150])}")
print(f"Companies with very high recent amount (>500万円): {len(df_expanded[df_expanded['直近取引金額(万円)'] > 500])}")
print(f"Companies with very low recent amount (<50万円): {len(df_expanded[df_expanded['直近取引金額(万円)'] < 50])}")
print(f"Very old companies (founded before 1980): {len(df_expanded[df_expanded['設立年'] < 1980])}")
print(f"Very new companies (founded after 2015): {len(df_expanded[df_expanded['設立年'] > 2015])}")

# CSVファイルに保存
df_expanded.to_csv('company_cluster_demo_enhanced.csv', index=False, encoding='utf-8-sig')
print(f"\n=== Enhanced dataset saved as 'company_cluster_demo_enhanced.csv' ===")

# 元データとの比較
print(f"\n=== Comparison with Original Dataset ===")
print(f"Original: {len(df_original)} companies")
print(f"Enhanced: {len(df_expanded)} companies")
print(f"Increase: {len(df_expanded) - len(df_original)} companies ({((len(df_expanded) - len(df_original)) / len(df_original) * 100):.1f}% increase)")

# 多様性指標
original_industries = len(df_original['業種'].unique())
enhanced_industries = len(df_expanded['業種'].unique())
original_regions = len(df_original['地域'].unique())
enhanced_regions = len(df_expanded['地域'].unique())

print(f"\nDiversity improvements:")
print(f"Industries: {original_industries} → {enhanced_industries} (+{enhanced_industries - original_industries})")
print(f"Regions: {original_regions} → {enhanced_regions} (+{enhanced_regions - original_regions})")

# データ範囲の比較
print(f"\nData range improvements:")
for col in ['売上(百万円)', '従業員数', '顧客満足度']:
    orig_range = df_original[col].max() - df_original[col].min()
    enhanced_range = df_expanded[col].max() - df_expanded[col].min()
    print(f"{col}: Range {orig_range:.1f} → {enhanced_range:.1f}")