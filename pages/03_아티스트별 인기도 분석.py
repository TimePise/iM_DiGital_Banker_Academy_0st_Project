import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="🎵아티스트별 인기도 분석", page_icon="📈", layout="wide")

# =========================================================================
# 스타일 & 헬퍼
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 2rem; border-radius: 10px; color: white;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: white; padding: 1rem; border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center;
    }
    .section-divider {
        border-top: 2px solid #667eea;
        margin: 2rem 0 1rem 0; padding-top: 1rem;
    }
    .insight {
        background: #f0f7ff; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #667eea; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def make_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family="맑은고딕", title_font_size=16,
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title_text=""
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    return fig

def show_insight(title, content):
    st.markdown(f"""
    <div class="insight">
        <strong>💡 {title}</strong><br>{content}
    </div>
    """, unsafe_allow_html=True)

# =========================================================================
# 데이터
df = pd.read_csv("data/kpop_2010_2025_curated_final.csv")

st.markdown("""
<div class="main-header">
    <h1>🎵아티스트별 인기도 분석</h1>
    <p>2010 ~ 2025 사이의 각 아티스트(BTS, BLACKPINK, TXT, AESPA)의 발매곡 인기도를 분석합니다.</p>
</div>
""", unsafe_allow_html=True)

# 기본 설정/가공
TARGET_ARTISTS = ["BTS", "Blackpink", "TXT", "aespa"]
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df = df.dropna(subset=["artist", "track_name", "popularity", "release_date"])
df = df[df["artist"].isin(TARGET_ARTISTS)].copy()

# 동일 발매일 평균 인기도
sorted_overview_df = (
    df.groupby(['artist', 'release_date'], as_index=False)['popularity']
      .mean()
)

# 아티스트별 데이터 분리
sorted_BTS       = sorted_overview_df[sorted_overview_df['artist'] == 'BTS']
sorted_Blackpink = sorted_overview_df[sorted_overview_df['artist'] == 'Blackpink']
sorted_TXT       = sorted_overview_df[sorted_overview_df['artist'] == 'TXT']
sorted_aespa     = sorted_overview_df[sorted_overview_df['artist'] == 'aespa']

ARTIST_COLORS = {
    "BTS": "#7C4DFF",
    "Blackpink": "#E91E63",
    "TXT": "#FF9800",
    "aespa": "#00AA44"
}

# =========================================================================
# KPI
avg_tracks = df.groupby('artist')['track_name'].count().mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #667eea; margin: 0;">{len(df):,}</h3>
        <p style="margin: 0;">총 트랙 수</p>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea; margin: 0;">2010 ~ 2025</h3>
        <p style="margin: 0;">분석 기간</p>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea; margin: 0;">4</h3>
        <p style="margin: 0;">분석 아티스트 수</p>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #667eea; margin: 0;">{avg_tracks:.0f}</h3>
        <p style="margin: 0;">아티스트별 평균 트랙 수</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# =========================================================================
# 탭 구성
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "① 개요(라인)", "② 아티스트별 추세", "③ 분포(박스플롯)",
    "④ 추세 분석(산점도+회귀)", "⑤ Top 5", "⑥ 인사이트"
])

# ① 개요: 전체 라인차트
with tab1:
    st.subheader("📈 발매곡 인기도 추세 — 개요")
    fig_1_overview = px.line(
        sorted_overview_df,
        x="release_date", y="popularity", color="artist",
        color_discrete_map=ARTIST_COLORS, title="Overview (동일 발매일 평균)"
    )
    fig_1_overview.update_yaxes(range=[0, 100])
    st.plotly_chart(make_chart(fig_1_overview), use_container_width=True)

# ② 아티스트별 라인 2x2
with tab2:
    st.subheader("📈 아티스트별 발매곡 인기도 추세")
    c1, c2 = st.columns(2)
    with c1:
        fig_bts = px.line(sorted_BTS, x="release_date", y="popularity", title="BTS")
        fig_bts.update_yaxes(range=[0, 100])
        st.plotly_chart(make_chart(fig_bts), use_container_width=True)
    with c2:
        fig_bp = px.line(sorted_Blackpink, x="release_date", y="popularity", title="Blackpink")
        fig_bp.update_yaxes(range=[0, 100])
        st.plotly_chart(make_chart(fig_bp), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_txt = px.line(sorted_TXT, x="release_date", y="popularity", title="TXT")
        fig_txt.update_yaxes(range=[0, 100])
        st.plotly_chart(make_chart(fig_txt), use_container_width=True)
    with c4:
        fig_aespa = px.line(sorted_aespa, x="release_date", y="popularity", title="aespa")
        fig_aespa.update_yaxes(range=[0, 100])
        st.plotly_chart(make_chart(fig_aespa), use_container_width=True)

# ③ 박스플롯
with tab3:
    st.subheader("📦 발매곡 인기도 분포 (박스플롯)")
    fig_box = px.box(
        df, x="artist", y="popularity", color="artist",
        color_discrete_map=ARTIST_COLORS, category_orders={"artist": TARGET_ARTISTS},
        title="Popularity Distribution by Artist"
    )
    st.plotly_chart(make_chart(fig_box), use_container_width=True)

# ④ 산점도 + 회귀선
with tab4:
    st.subheader("📉 발매곡 인기도 추세 분석 (산점도 + 회귀선)")
    def add_regression_line(fig: go.Figure, dfx: pd.DataFrame, name: str, color: str):
        if len(dfx) < 2:
            return
        # datetime → epoch seconds (float)
        x_time = pd.to_datetime(dfx['release_date']).astype('int64') / 1e9
        y = dfx['popularity'].values
        try:
            k, b = np.polyfit(x_time, y, 1)
        except Exception:
            return
        x_line = np.linspace(x_time.min(), x_time.max(), 100)
        y_line = k * x_line + b
        x_line_dates = pd.to_datetime(x_line * 1e9)

        fig.add_trace(go.Scatter(
            x=x_line_dates, y=y_line, mode='lines',
            name=f"{name} Trend", line=dict(color=color, width=2, dash="dash"))
        )

    c1, c2 = st.columns(2)
    with c1:
        fig_s1 = px.scatter(sorted_BTS, x="release_date", y="popularity", title="BTS")
        add_regression_line(fig_s1, sorted_BTS, "BTS", "#111827")
        st.plotly_chart(make_chart(fig_s1), use_container_width=True)
    with c2:
        fig_s2 = px.scatter(sorted_Blackpink, x="release_date", y="popularity", title="Blackpink")
        add_regression_line(fig_s2, sorted_Blackpink, "Blackpink", "#111827")
        st.plotly_chart(make_chart(fig_s2), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_s3 = px.scatter(sorted_TXT, x="release_date", y="popularity", title="TXT")
        add_regression_line(fig_s3, sorted_TXT, "TXT", "#111827")
        st.plotly_chart(make_chart(fig_s3), use_container_width=True)
    with c4:
        fig_s4 = px.scatter(sorted_aespa, x="release_date", y="popularity", title="aespa")
        add_regression_line(fig_s4, sorted_aespa, "aespa", "#111827")
        st.plotly_chart(make_chart(fig_s4), use_container_width=True)

# ⑤ Top 5
with tab5:
    st.subheader("🏆 각 그룹별 인기도 상위 5개 곡")
    def get_top_n_songs(dfx: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        return dfx.nlargest(n, 'popularity')[['artist', 'album_type', 'track_name', 'popularity']]

    top_5_BTS       = get_top_n_songs(df[df['artist'] == 'BTS'])
    top_5_Blackpink = get_top_n_songs(df[df['artist'] == 'Blackpink'])
    top_5_TXT       = get_top_n_songs(df[df['artist'] == 'TXT'])
    top_5_aespa     = get_top_n_songs(df[df['artist'] == 'aespa'])

    st.markdown("**BTS Top 5 Songs**")
    st.dataframe(top_5_BTS, use_container_width=True)

    st.markdown("**Blackpink Top 5 Songs**")
    st.dataframe(top_5_Blackpink, use_container_width=True)

    st.markdown("**TXT Top 5 Songs**")
    st.dataframe(top_5_TXT, use_container_width=True)

    st.markdown("**aespa Top 5 Songs**")
    st.dataframe(top_5_aespa, use_container_width=True)

# ⑥ 인사이트
with tab6:
    st.subheader("📈 Insights")
    show_insight(
        "각 아티스트별 발매 선호 앨범 타입과 인기도",
        "- BTS: 앨범 단위 발매곡 중 일부 곡의 인기도가 높음<br>"
        "- BLACKPINK: 싱글 앨범 발매곡의 인기도가 높음<br>"
        "- TXT: 싱글/앨범 모두 고르게 분포<br>"
        "- aespa: 싱글 위주지만, *Drama*는 앨범 수록곡임에도 높은 인기도"
    )
    show_insight(
        "정리",
        "- 발매곡 인기도는 그룹별 전략 차이에 의해 큰 영향을 받음<br>"
        "- ‘많이 낸다’보다 **어떻게 기획했는가**가 중요"
    )
