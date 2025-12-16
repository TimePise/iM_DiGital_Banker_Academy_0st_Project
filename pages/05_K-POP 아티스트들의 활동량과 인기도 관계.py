# -*- coding: utf-8 -*-
"""
K-POP 활동량(연도별 발매 곡 수) vs 인기도(평균 popularity)
- 사이드바 제거: 기본 CSV만 사용 (data/kpop_2010_2025_curated_final.csv)
- 시각화 탭형 UI
대상: BTS / BLACKPINK / TXT / aespa
"""

from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dateutil import parser as dateparser

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(page_title="K_POP 활동량 vs 인기도", page_icon="🎵", layout="wide")

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
    .main-header { background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 2rem; border-radius: 10px; color: white;
        text-align: center; margin-bottom: 2rem; }
    .metric-card { background: #ffffff; padding: 1rem; border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; color: #0f172a; }
    .section-divider { border-top: 2px solid #667eea; margin: 2rem 0 1rem 0; padding-top: 1rem; }
    .insight { background: #f0f7ff; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #667eea; margin: 1rem 0; color: #0f172a; }
    .insight, .insight * { color: #0f172a !important; }
    @media (prefers-color-scheme: dark) {
        .main-header { background: linear-gradient(90deg, #4f46e5, #6d28d9); }
        .metric-card { background: #111827; border: 1px solid #1f2937; box-shadow: none; color: #e5e7eb; }
        .insight { background: #0b1220; border-left-color: #8ab4f8; color: #e5e7eb; }
        .insight, .insight * { color: #e5e7eb !important; }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 헬퍼
# -----------------------------
def make_chart(fig: go.Figure, dark: bool = False) -> go.Figure:
    if dark:
        bg, fg, sub, grid = "#0b1220", "#e5e7eb", "#9ca3af", "#1f2937"
    else:
        bg, fg, sub, grid = "#ffffff", "#0f172a", "#475569", "#e5e7eb"
    fig.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=fg, family="맑은고딕"),
        title_font=dict(size=16, color=fg),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=fg)),
        margin=dict(t=60, r=20, b=40, l=60)
    )
    fig.for_each_xaxis(lambda ax: ax.update(
        title_font=dict(color=fg), tickfont=dict(color=sub),
        gridcolor=grid, zerolinecolor=grid, linecolor=grid))
    fig.for_each_yaxis(lambda ay: ay.update(
        title_font=dict(color=fg), tickfont=dict(color=sub),
        gridcolor=grid, zerolinecolor=grid, linecolor=grid))
    try:
        fig.update_coloraxes(colorbar=dict(
            bgcolor=bg, tickfont=dict(color=sub), outlinecolor=grid,
            title=dict(font=dict(color=fg))
        ))
    except Exception:
        pass
    return fig

def show_insight(title: str, content: str):
    st.markdown(f"""
    <div class="insight"><strong>💡 {title}</strong><br>{content}</div>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def _to_year(x) -> Optional[int]:
    if pd.isna(x): return None
    try: return int(x)
    except Exception:
        for fn in (lambda v: dateparser.parse(str(v)).year, lambda v: int(str(v)[:4])):
            try: return fn(x)
            except Exception: pass
    return None

@st.cache_data(show_spinner=False)
def gini(array: np.ndarray) -> float:
    x = np.array(array, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0: return np.nan
    if np.any(x < 0): x -= x.min()
    if np.all(x == 0): return 0.0
    x_sorted = np.sort(x); n = x_sorted.size; cumx = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

# -----------------------------
# 헤더
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>🎵 K-POP 활동량 vs 인기도 분석</h1>
    <p>CSV만으로 2013–2025 사이의 활동량과 인기도 관계를 시각화합니다 (BTS / BLACKPINK / TXT / aespa)</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 상수/타겟 & 색상
# -----------------------------
MIN_YEAR, MAX_YEAR = 2013, 2025
TARGET = ["BTS", "BLACKPINK", "TXT", "aespa"]
TREND_COLORS = {"증가": "#10B981", "감소": "#EF4444", "유지": "#6B7280"}
ARTIST_COLORS = {"BTS": "#7C4DFF", "BLACKPINK": "#E91E63", "TXT": "#45B7D1", "aespa": "#FF9800"}
ALIASES = {
    "bts": "BTS", "방탄소년단": "BTS",
    "blackpink": "BLACKPINK", "블랙핑크": "BLACKPINK",
    "txt": "TXT", "tomorrow x together": "TXT",
    "tomorrow xtogether": "TXT", "tomorrowxtogether": "TXT", "투모로우바이투게더": "TXT",
    "aespa": "aespa", "æspa": "aespa", "에스파": "aespa",
}

def canonicalize(name: str) -> Optional[str]:
    if not isinstance(name, str): return None
    key = name.strip().lower()
    if "tomorrow x together" in key or "투모로우" in key: return "TXT"
    return ALIASES.get(key, name.strip())

# -----------------------------
# 데이터 로드 (기본 CSV만 사용)
# -----------------------------
default_path = Path.cwd() / "data" / "kpop_2010_2025_curated_final.csv"
if not default_path.exists():
    st.error(f"CSV를 찾을 수 없습니다: {default_path}")
    st.stop()

@st.cache_data(show_spinner=True)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

df_raw = load_csv(default_path)

# -----------------------------
# 전처리
# -----------------------------
required = {"artist", "track_name", "popularity"}
missing = required - set(df_raw.columns)
if missing:
    st.error(f"필수 컬럼 누락: {missing}")
    st.stop()

df = df_raw.copy()
df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")

if "year" not in df.columns:
    if "release_date" in df.columns:
        df["year"] = df["release_date"].apply(_to_year)
    else:
        st.error("year 컬럼이 없고 release_date도 없어 연도 산출이 불가합니다.")
        st.stop()

before_na = df["artist"].isna().sum()
df["artist"] = df["artist"].apply(canonicalize)
after_na = df["artist"].isna().sum()

n_rows_before = len(df)
df = df.dropna(subset=["artist", "track_name", "popularity", "year"]).copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)

# 연도/타겟 필터(고정): 사이드바 슬라이더 제거
f = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR) & (df["artist"].isin(TARGET))].copy()
if f.empty:
    st.warning("필터 결과가 없습니다. 기본 데이터 범위를 확인하세요.")
    st.stop()

# -----------------------------
# 집계
# -----------------------------
annual = (
    f.groupby(["artist", "year"], observed=True)
     .agg(tracks_count=("track_name", "count"),
          avg_popularity=("popularity", "mean"))
     .reset_index()
)
by_artist = (
    f.groupby("artist", observed=True)
     .agg(total_tracks=("track_name", "count"),
          mean_pop=("popularity", "mean"),
          med_pop=("popularity", "median"),
          gini_pop=("popularity", lambda x: gini(np.array(x))))
     .reindex(TARGET)
     .reset_index()
)

# -----------------------------
# KPI 카드
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><h3 style="color:#667eea;margin:0;">{len(f):,}</h3><p style="margin:0;">총 트랙 수</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><h3 style="color:#667eea;margin:0;">{len(annual):,}</h3><p style="margin:0;">연도×아티스트 집계</p></div>', unsafe_allow_html=True)
with c3:
    yr_rng = f"{int(annual['year'].min())}–{int(annual['year'].max())}" if not annual.empty else "-"
    st.markdown(f'<div class="metric-card"><h3 style="color:#667eea;margin:0;">{yr_rng}</h3><p style="margin:0;">연도 범위</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><h3 style="color:#667eea;margin:0;">{len(TARGET)}</h3><p style="margin:0;">아티스트 수</p></div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# -----------------------------
# 탭 구성
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "① 히트맵", "② 트렌드", "③ 전년대비(YoY)", "④ 산점도+회귀",
    "⑤ 분포·Gini·Top5", "⑥ 사분위 vs 인기도", "⑦ 코호트", "⑧ 상관분석"
])

# ① 히트맵
with tab1:
    st.subheader("연도×아티스트 히트맵")
    pivot_tracks = annual.pivot(index="artist", columns="year", values="tracks_count").reindex(TARGET)
    fig_ht = px.imshow(pivot_tracks, aspect="auto",
                       title="Tracks per year (연도×아티스트)",
                       labels=dict(x="Year", y="Artist", color="Tracks"))
    fig_ht.update_yaxes(autorange="reversed")
    st.plotly_chart(make_chart(fig_ht), use_container_width=True)

    pivot_pop = annual.pivot(index="artist", columns="year", values="avg_popularity").reindex(TARGET)
    fig_hp = px.imshow(pivot_pop, aspect="auto",
                       title="Avg. Popularity (연도×아티스트)",
                       labels=dict(x="Year", y="Artist", color="Avg. Pop"))
    fig_hp.update_yaxes(autorange="reversed")
    st.plotly_chart(make_chart(fig_hp), use_container_width=True)

# ② 트렌드
with tab2:
    st.subheader("연도별 트렌드")
    fig_t1 = px.line(annual, x="year", y="tracks_count", color="artist",
                     color_discrete_map=ARTIST_COLORS, markers=True, title="Tracks per year")
    st.plotly_chart(make_chart(fig_t1), use_container_width=True)

    fig_t2 = px.line(annual, x="year", y="avg_popularity", color="artist",
                     color_discrete_map=ARTIST_COLORS, markers=True, title="Avg. Popularity (0–100)")
    st.plotly_chart(make_chart(fig_t2), use_container_width=True)

# ③ YoY
with tab3:
    st.subheader("전년 대비 변화 (YoY)")
    annual_sorted = annual.sort_values(["artist", "year"]).copy()
    annual_sorted["tracks_yoy"] = annual_sorted.groupby("artist", observed=True)["tracks_count"].diff()
    annual_sorted["pop_yoy"] = annual_sorted.groupby("artist", observed=True)["avg_popularity"].diff()
    def flag(x: float) -> str:
        if pd.isna(x) or abs(x) < 1e-9: return "유지"
        return "증가" if x > 0 else "감소"
    annual_sorted["tracks_yoy_flag"] = annual_sorted["tracks_yoy"].apply(flag)
    annual_sorted["pop_yoy_flag"] = annual_sorted["pop_yoy"].apply(flag)

    fig_y1 = px.bar(annual_sorted, x="year", y="tracks_yoy", color="tracks_yoy_flag",
                    color_discrete_map=TREND_COLORS, facet_col="artist",
                    category_orders={"artist": TARGET}, title="Δ Tracks (YoY)")
    st.plotly_chart(make_chart(fig_y1), use_container_width=True)

    fig_y2 = px.bar(annual_sorted, x="year", y="pop_yoy", color="pop_yoy_flag",
                    color_discrete_map=TREND_COLORS, facet_col="artist",
                    category_orders={"artist": TARGET}, title="Δ Avg. Pop (YoY)")
    st.plotly_chart(make_chart(fig_y2), use_container_width=True)

# ④ 산점도 + 회귀선
with tab4:
    st.subheader("활동량 vs 평균 인기도 (연도×아티스트)")
    reg_opt = st.radio("회귀선 옵션", ["전체(단일)", "아티스트별"], horizontal=True)
    fig_scatter = px.scatter(
        annual, x="tracks_count", y="avg_popularity", color="artist",
        color_discrete_map=ARTIST_COLORS, size_max=12,
        hover_data=["artist", "year", "tracks_count", "avg_popularity"],
        title="Tracks per year vs Avg. Popularity"
    )
    def add_reg_line(fig: go.Figure, df: pd.DataFrame, name: str, color: str):
        if len(df) < 2: return
        x, y = df["tracks_count"].values, df["avg_popularity"].values
        try: k, b = np.polyfit(x, y, 1)
        except Exception: return
        xs = np.linspace(x.min(), x.max(), 100)
        ys = k * xs + b
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name,
                                 line=dict(color=color, width=2, dash="dash")))
    if reg_opt == "전체(단일)":
        add_reg_line(fig_scatter, annual, "Regression (All)", "#111827")
    else:
        for a, g in annual.groupby("artist", observed=True):
            add_reg_line(fig_scatter, g, f"Regression ({a})", ARTIST_COLORS.get(a, "#111827"))
    st.plotly_chart(make_chart(fig_scatter), use_container_width=True)

# ⑤ 분포·Gini·Top5
with tab5:
    st.subheader("인기 분포(박스플롯) · Gini(쏠림) · Top5 인기곡")
    fig_box = px.box(f, x="artist", y="popularity", color="artist",
                     color_discrete_map=ARTIST_COLORS, category_orders={"artist": TARGET},
                     title="Popularity Distribution by Artist")
    st.plotly_chart(make_chart(fig_box), use_container_width=True)
    st.markdown("**아티스트별 요약 (필터 적용)**")
    st.dataframe(by_artist.round({"mean_pop": 2, "med_pop": 2, "gini_pop": 3}),
                 use_container_width=True)

    def topk(g, k=5):
        return g.nlargest(k, "popularity")[["track_name", "popularity", "year"]]
    top5_table = (f.groupby("artist", group_keys=False, observed=True)
                    .apply(topk).reset_index().rename(columns={"artist": "artist"}))
    st.markdown("**Top5 인기곡 (필터 적용)**")
    st.dataframe(top5_table, use_container_width=True)

# ⑥ 사분위 vs 인기도
with tab6:
    st.subheader("활동량 사분위별 평균 인기도")
    qcut = annual.copy()
    if len(qcut) >= 4:
        qcut["activity_quartile"] = pd.qcut(qcut["tracks_count"], 4,
                                            labels=["Q1(낮음)", "Q2", "Q3", "Q4(높음)"])
        qtab = (qcut.groupby(["artist", "activity_quartile"], observed=True)
                    .agg(avg_pop=("avg_popularity", "mean")).reset_index())
        fig_q = px.bar(qtab, x="activity_quartile", y="avg_pop", color="artist",
                       color_discrete_map=ARTIST_COLORS, facet_col="artist",
                       category_orders={"artist": TARGET,
                                        "activity_quartile": ["Q1(낮음)", "Q2", "Q3", "Q4(높음)"]},
                       title="Avg. Popularity by Activity Quartile")
        st.plotly_chart(make_chart(fig_q), use_container_width=True)
    else:
        st.info("사분위 분석을 하려면 연도×아티스트 집계가 최소 4개 이상 필요합니다.")

# ⑦ 코호트
with tab7:
    st.subheader("코호트 비교 (2010–15 / 2016–20 / 2021–25)")
    st.info("코호트는 활동 시작 시점이 아닌, 곡 발매 연도를 기준으로 구분합니다.")
    coh = f.copy()
    coh["cohort"] = pd.cut(coh["year"], bins=[2009, 2015, 2020, 2025],
                           labels=["2010–15", "2016–20", "2021–25"], include_lowest=True)
    coh_agg = (coh.groupby(["artist", "cohort"], observed=True)
                   .agg(tracks=("track_name", "count"), avg_pop=("popularity", "mean")).reset_index())
    fig_ct = px.bar(coh_agg, x="cohort", y="tracks", color="artist",
                    color_discrete_map=ARTIST_COLORS, facet_col="artist",
                    category_orders={"artist": TARGET}, title="Cohort vs Tracks")
    st.plotly_chart(make_chart(fig_ct), use_container_width=True)
    fig_cp = px.bar(coh_agg, x="cohort", y="avg_pop", color="artist",
                    color_discrete_map=ARTIST_COLORS, facet_col="artist",
                    category_orders={"artist": TARGET}, title="Cohort vs Avg. Popularity")
    st.plotly_chart(make_chart(fig_cp), use_container_width=True)

# ⑧ 상관분석
with tab8:
    st.subheader("상관분석 (활동량↔인기도)")
    if len(annual) >= 3:
        pearson = annual[["tracks_count", "avg_popularity"]].corr(method="pearson").iloc[0, 1]
        spearman = annual[["tracks_count", "avg_popularity"]].corr(method="spearman").iloc[0, 1]
        c1, c2 = st.columns(2)
        with c1: st.metric("Pearson r (전체)", f"{pearson:.3f}")
        with c2: st.metric("Spearman ρ (전체)", f"{spearman:.3f}")
        rows: List[Dict] = []
        for a, g in annual.groupby("artist", observed=True):
            if len(g) >= 3:
                r = g[["tracks_count", "avg_popularity"]].corr(method="pearson").iloc[0, 1]
                s = g[["tracks_count", "avg_popularity"]].corr(method="spearman").iloc[0, 1]
                rows.append({"artist": a, "pearson_r": r, "spearman_rho": s, "n_years": len(g)})
        if rows:
            st.dataframe(pd.DataFrame(rows)[["artist", "pearson_r", "spearman_rho", "n_years"]]
                         .sort_values("artist"), use_container_width=True)
    else:
        st.info("상관분석을 위해서는 연도×아티스트 집계가 최소 3개 이상 필요합니다.")

# -----------------------------
# 인사이트
# -----------------------------
show_insight(
    "전체 요약",
    "활동량(연도별 곡 수)과 평균 인기도는 단순 비례하지 않았습니다. "
    "BLACKPINK는 소수 발매에도 높은 평균 인기도의 '히트 집중형', "
    "BTS는 다작에도 평균 인기도를 안정적으로 유지하는 '팬덤 안정형' 특성이 보입니다. "
    "TXT·aespa는 2021–25 코호트에서 상승세가 두드러집니다."
)
show_insight(
    "아티스트별 스냅샷",
    "BTS는 2015–2019년 동시 상승, BLACKPINK는 희소성 기반의 높은 집중도(Gini↑), "
    "TXT는 최근 중앙값 상승, aespa는 2021년 이후 효율 높은 타이틀 중심 전략이 관찰됩니다."
)
show_insight(
    "마무리",
    "평균 인기도는 '얼마나 많이'보다 <b>'어떻게 기획했는가'</b>에 좌우됩니다. "
    "향후 전략은 발매량 확대보다 그룹 특성과 타이밍에 맞춘 기획 최적화가 핵심입니다."
)
