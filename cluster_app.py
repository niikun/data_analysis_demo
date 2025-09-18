import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ページ設定
st.set_page_config(
    page_title="企業クラスター分析アプリ",
    page_icon="🏢",
    layout="wide"
)

# モデルとデータの読み込み
@st.cache_resource
def load_models():
    try:
        kmeans = joblib.load('cluster_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        cluster_stats = joblib.load('cluster_stats.pkl')
        return kmeans, scaler, feature_names, cluster_stats
    except FileNotFoundError:
        st.error("モデルファイルが見つかりません。cluster_model.pyを実行してください。")
        return None, None, None, None

# データ準備関数
def prepare_input_data(company_data):
    """入力データを特徴量に変換"""
    # 基本数値特徴量
    base_features = {
        'Revenue_M_JPY': company_data['revenue'],
        'Employees': company_data['employees'],
        'Past_Transactions': company_data['past_transactions'],
        'Unit_Price_10K_JPY': company_data['unit_price'],
        'Customer_Satisfaction': company_data['satisfaction'],
        'Company_Age': 2024 - company_data['founded_year']
    }

    # 業種ダミー変数
    industries = ['Construction', 'Education', 'Finance', 'Food_Service', 'Healthcare',
                 'IT', 'Logistics', 'Manufacturing', 'Retail', 'Services']
    for industry in industries:
        base_features[f'Industry_{industry}'] = 1 if company_data['industry'] == industry else 0

    # 地域ダミー変数
    regions = ['Chubu', 'Chugoku', 'Kansai', 'Kanto', 'Kyushu']
    for region in regions:
        base_features[f'Region_{region}'] = 1 if company_data['region'] == region else 0

    return pd.DataFrame([base_features])

# クラスター解釈関数
def interpret_cluster(cluster_id, cluster_stats):
    """クラスター結果の解釈"""
    cluster_names = {
        0: "中堅企業層",
        1: "小規模企業層",
        2: "大手企業層",
        3: "一般企業層"
    }

    cluster_descriptions = {
        0: "物流・医療・小売業中心の中堅企業。中部・九州地域に多く、安定した取引実績を持つ。",
        1: "サービス・教育業中心の小規模企業。顧客満足度が最も高く、若い企業が多い。",
        2: "建設・金融業中心の大手企業。関東立地の老舗企業で、最高水準の取引金額を誇る。",
        3: "多様な業種の一般企業。関東中心で、標準的な取引パターンを示す最大グループ。"
    }

    stats = cluster_stats.loc[cluster_id]

    return {
        'name': cluster_names[cluster_id],
        'description': cluster_descriptions[cluster_id],
        'avg_recent_amount': stats['Recent_Amount_10K_JPY'],
        'avg_revenue': stats['Revenue_M_JPY'],
        'avg_employees': stats['Employees'],
        'avg_age': stats['Company_Age'],
        'avg_satisfaction': stats['Customer_Satisfaction']
    }

# メイン関数
def main():
    st.title("🏢 企業クラスター分析アプリ")
    st.markdown("### 企業情報を入力して、4つのクラスターのどれに分類されるかを予測します")

    # モデル読み込み
    kmeans, scaler, feature_names, cluster_stats = load_models()

    if kmeans is None:
        return

    # サイドバーで入力
    st.sidebar.header("📊 企業情報入力")

    # 企業名
    company_name = st.sidebar.text_input("企業名", "新規企業")

    # 基本情報
    revenue = st.sidebar.number_input("売上 (百万円)", min_value=1, max_value=10000, value=500)
    employees = st.sidebar.number_input("従業員数", min_value=1, max_value=1000, value=100)
    founded_year = st.sidebar.slider("設立年", min_value=1950, max_value=2024, value=2000)

    # 取引情報
    past_transactions = st.sidebar.number_input("過去取引回数", min_value=0, max_value=50, value=5)
    unit_price = st.sidebar.number_input("取引単価 (万円)", min_value=1, max_value=500, value=50)
    satisfaction = st.sidebar.slider("顧客満足度", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

    # 業種選択
    industry_options = {
        'Manufacturing': '製造業',
        'IT': 'IT業',
        'Retail': '小売業',
        'Food_Service': '飲食業',
        'Construction': '建設業',
        'Services': 'サービス業',
        'Logistics': '物流業',
        'Finance': '金融業',
        'Education': '教育業',
        'Healthcare': '医療業'
    }

    industry = st.sidebar.selectbox(
        "業種",
        options=list(industry_options.keys()),
        format_func=lambda x: industry_options[x]
    )

    # 地域選択
    region_options = {
        'Kanto': '関東',
        'Kansai': '関西',
        'Chubu': '中部',
        'Kyushu': '九州',
        'Chugoku': '中国'
    }

    region = st.sidebar.selectbox(
        "地域",
        options=list(region_options.keys()),
        format_func=lambda x: region_options[x]
    )

    # 予測ボタン
    if st.sidebar.button("🔍 クラスター予測", type="primary"):
        # 入力データ準備
        company_data = {
            'revenue': revenue,
            'employees': employees,
            'founded_year': founded_year,
            'past_transactions': past_transactions,
            'unit_price': unit_price,
            'satisfaction': satisfaction,
            'industry': industry,
            'region': region
        }

        # 特徴量変換
        input_df = prepare_input_data(company_data)

        # 標準化
        input_scaled = scaler.transform(input_df)

        # 予測
        cluster_pred = kmeans.predict(input_scaled)[0]
        cluster_info = interpret_cluster(cluster_pred, cluster_stats)

        # 結果表示
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader(f"📈 予測結果: {cluster_info['name']}")
            st.write(f"**企業名**: {company_name}")
            st.write(f"**クラスター**: Cluster {cluster_pred}")
            st.info(cluster_info['description'])

            # 予測した直近取引金額
            expected_amount = cluster_info['avg_recent_amount']
            st.metric(
                label="予想直近取引金額",
                value=f"{expected_amount:.0f}万円",
                delta=f"クラスター平均"
            )

        with col2:
            st.subheader("📊 クラスター特性比較")

            # 入力企業とクラスター平均の比較
            comparison_data = {
                '指標': ['売上(百万円)', '従業員数', '企業年数', '顧客満足度'],
                '入力企業': [revenue, employees, 2024-founded_year, satisfaction],
                'クラスター平均': [
                    cluster_info['avg_revenue'],
                    cluster_info['avg_employees'],
                    cluster_info['avg_age'],
                    cluster_info['avg_satisfaction']
                ]
            }

            comparison_df = pd.DataFrame(comparison_data)

            # レーダーチャート用データ準備
            categories = comparison_df['指標'].tolist()

            # 正規化（0-1スケール）
            max_values = [5000, 600, 60, 5.0]  # 各指標の最大値
            input_normalized = [comparison_df['入力企業'][i] / max_values[i] for i in range(4)]
            cluster_normalized = [comparison_df['クラスター平均'][i] / max_values[i] for i in range(4)]

            # レーダーチャート作成
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=input_normalized + [input_normalized[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='入力企業',
                line_color='blue'
            ))

            fig.add_trace(go.Scatterpolar(
                r=cluster_normalized + [cluster_normalized[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='クラスター平均',
                line_color='red'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # 全クラスター比較
        st.subheader("🏆 全クラスター比較")

        # クラスター統計の可視化
        fig_bar = px.bar(
            cluster_stats.reset_index(),
            x='Cluster',
            y='Recent_Amount_10K_JPY',
            title='クラスター別平均直近取引金額',
            labels={'Recent_Amount_10K_JPY': '平均直近取引金額(万円)', 'Cluster': 'クラスター'}
        )

        # 予測されたクラスターをハイライト
        colors = ['lightblue'] * 4
        colors[cluster_pred] = 'red'
        fig_bar.update_traces(marker_color=colors)

        st.plotly_chart(fig_bar, use_container_width=True)

        # クラスター詳細テーブル
        cluster_details = cluster_stats.copy()
        cluster_details.index = [f"Cluster {i}" for i in cluster_details.index]
        cluster_details.columns = ['平均直近取引金額(万円)', '平均売上(百万円)', '平均従業員数', '平均企業年数', '平均顧客満足度']

        st.dataframe(cluster_details, use_container_width=True)

if __name__ == "__main__":
    main()