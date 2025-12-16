# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------- 1) 페이지 설정 -------------------------
st.set_page_config(
    page_title="🎵 K-POP 인기곡 트렌드: 곡 길이 변화",
    page_icon="🎵",
    layout="wide",
)

# ------------------------- 2) CSS 스타일 -------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 2rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: white; padding: 1rem; border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.10); text-align: center;
        border: 1px solid #eef2ff;
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

# ------------------------- 3) 헬퍼 함수 -------------------------
def make_chart(fig: go.Figure) -> go.Figure:
    """차트 스타일 통일"""
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="맑은고딕", title_font_size=18,
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title_text=""
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    return fig

def show_insight(title: str, content: str):
    """인사이트 박스"""
    st.markdown(f"""
    <div class="insight">
        <strong>💡 {title}</strong><br>{content}
    </div>
    """, unsafe_allow_html=True)

def convert_ms_to_min_sec(ms: int) -> str:
    """밀리초 → 'M분 S초'"""
    if pd.isna(ms):
        return "-"
    total_seconds = int(ms) // 1000
    return f"{total_seconds // 60}분 {total_seconds % 60}초"

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame | None:
    """CSV 로드 & 기본 전처리"""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 앱 폴더의 data 하위에 두거나 경로를 확인하세요.")
        return None

    df.columns = df.columns.str.lower().str.strip()
    # 필수 컬럼 존재 체크
    needed = {"artist", "track_name", "popularity", "duration_ms", "release_date"}
    missing = needed - set(df.columns)
    if missing:
        st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing))}")
        return None

    # 타입 처리
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    df["year"] = df["release_date"].dt.year
    return df

# ------------------------- 4) 헤더 -------------------------
st.markdown("""
<div class="main-header">
    <h1>🎵 K-POP 인기곡 트렌드 분석: 곡 길이의 변화</h1>
    <p>연도·아티스트별 최고 인기곡을 기준으로, 곡 길이(분)의 변화를 시각화하고 인사이트를 정리했습니다.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------- 5) 데이터 불러오기 -------------------------
# ✅ CSV 경로 입력 제거: 기본 경로만 사용
default_path = os.path.join("data", "kpop_2010_2025_curated_final.csv")
df = load_data(default_path)
if df is None:
    st.stop()

# ------------------------- 6) 분석 준비 -------------------------
# 아티스트-연도별 popularity 최대 행 추출
df_pop = df.dropna(subset=["popularity"]).copy()
grp = df_pop.groupby(["artist", "year"], as_index=False)
idx = grp["popularity"].idxmax()
top_songs_by_year = df_pop.loc[idx["popularity"]].copy()

top_songs_by_year["duration_min"] = top_songs_by_year["duration_ms"] / 60000.0
top_songs_by_year["duration_min_sec"] = top_songs_by_year["duration_ms"].apply(convert_ms_to_min_sec)

# 테이블에 보여줄 최소 컬럼
table_cols = ["artist", "year", "track_name", "popularity", "duration_min_sec"]
table_data = top_songs_by_year[table_cols].sort_values(["artist", "year"]).reset_index(drop=True)

# 공통 메트릭
years = sorted(df["year"].dropna().unique())
year_min, year_max = (years[0], years[-1]) if years else (None, None)
avg_len_min = top_songs_by_year["duration_min"].mean()
artist_cnt = top_songs_by_year["artist"].nunique()
row_cnt = len(df)

# 카드형 메트릭
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color:#667eea;margin:0;">{row_cnt:,}</h3>
        <p style="margin:0;">전체 행 수</p>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color:#667eea;margin:0;">{artist_cnt:,}</h3>
        <p style="margin:0;">아티스트 수(최고 인기곡 기준)</p>
    </div>
    """, unsafe_allow_html=True)
with c3:
    yr = f"{year_min}–{year_max}" if year_min is not None else "-"
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color:#667eea;margin:0;">{yr}</h3>
        <p style="margin:0;">데이터 연도 범위</p>
    </div>
    """, unsafe_allow_html=True)
with c4:
    avg_txt = f"{avg_len_min:.2f} 분" if pd.notna(avg_len_min) else "-"
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color:#667eea;margin:0;">{avg_txt}</h3>
        <p style="margin:0;">최고 인기곡 평균 길이</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ------------------------- 7) 탭 레이아웃 -------------------------
tab1, tab2 = st.tabs(["📋 테이블 보기", "📈 시각화"])

# ------------------------- 7-1) 테이블 탭 -------------------------
with tab1:
    st.subheader("연도별 그룹별 가장 인기 있는 곡")
    st.caption("각 아티스트의 연도별 최고 인기곡과 길이(분:초)를 표시합니다.")

    # 기본 4개 아티스트 (기존 유지)
    artists_to_show = ["BTS", "Blackpink", "TXT", "aespa"]
    col_left, col_right = st.columns(2)

    # 좌측 두 개
    with col_left:
        for a in artists_to_show[0:2]:
            st.write(f"### {a} 인기곡 데이터")
            st.dataframe(table_data[table_data["artist"] == a], use_container_width=True)
    # 우측 두 개
    with col_right:
        for a in artists_to_show[2:4]:
            st.write(f"### {a} 인기곡 데이터")
            st.dataframe(table_data[table_data["artist"] == a], use_container_width=True)

# ------------------------- 7-2) 시각화 탭 -------------------------
with tab2:
    st.subheader("연도별 인기곡 길이 변화 (산점도)")
    st.caption("버블 크기는 popularity(인기 지표)에 비례합니다. 빨간 점선은 전체 추세선입니다.")

    # 데이터에 존재하는 아티스트만 대상으로 색 매핑 생성
    uniq_artists = top_songs_by_year["artist"].dropna().unique().tolist()
    palette = px.colors.qualitative.Set2
    color_map = {a: palette[i % len(palette)] for i, a in enumerate(sorted(uniq_artists))}

    fig = px.scatter(
        top_songs_by_year.sort_values("year"),
        x="year", y="duration_min",
        color="artist",
        size="popularity", size_max=28,
        hover_data={"track_name": True, "popularity": True, "duration_min": ":.2f"},
        color_discrete_map=color_map,
        title="연도별 그룹별 최고 인기곡 길이(분)"
    )

    # 전체 회귀선(1차)
    x = top_songs_by_year["year"].values
    y = top_songs_by_year["duration_min"].values
    if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        coeff = np.polyfit(x, y, 1)
        p = np.poly1d(coeff)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = p(xs)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="전체 곡 길이 추세선"
        ))

    fig.update_yaxes(title_text="곡 길이 (분)", range=[1, 5])
    fig.update_xaxes(dtick=1, title_text="연도")
    fig = make_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    # 인사이트 박스
    show_insight(
    "곡 길이 단축 트렌드",
    """이 그래프는 K-POP 인기곡의 길이가 **점점 짧아지는 추세**를 시각적으로 보여줍니다.
    **빨간색 점선 회귀선**이 이 추세를 명확하게 보여주고 있습니다.
    
    이러한 현상은 단순히 트렌드를 넘어, 음악 소비 환경의 근본적인 변화를 반영합니다.
    
    * **스트리밍 환경의 영향**: 2000년대 후반 유료 음원 사이트가 정착하면서, 무료 미리 듣기 1분 안에 주요 멜로디와 후렴을 넣어 청자의 관심을 끌어야 했습니다.
    * **쇼트폼 콘텐츠의 부상**: 최근 몇 년간 틱톡, 유튜브 쇼츠 등 **숏폼 콘텐츠**가 신곡 홍보의 필수 코스로 떠오르며, 노래를 각인시키는 시간이 기존 미리 듣기 1분에서 수십 초로 줄어들었습니다.
    * **전주(Intro)의 단축**: 과거 긴 전주를 찾아보기 어려워졌고, 2~4마디로 주된 비트만 소개하는 수준으로 바뀌었습니다.
    * **안무와 실용성**: K팝의 격한 안무를 소화하려면 노래 길이가 짧을수록 유리하다는 실용적인 이유도 있습니다.
    
    결론적으로, 노래 길이가 짧아지는 것은 **리스너들의 집중 시간을 빠르게 사로잡고** 다양한 플랫폼에 효과적으로 노출시키기 위한 전략적인 선택으로 보입니다."""
)

