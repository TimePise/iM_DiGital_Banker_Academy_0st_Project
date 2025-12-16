# app_long_loved_dashboard.py
# ⏱️ 세대×성별 지속력 대시보드 (업로드/사이드바 제거 버전)

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ---------------- 1) 페이지 설정 ----------------
st.set_page_config(page_title="세대×성별 지속력 대시보드", page_icon="⏱️", layout="wide")

# ---------------- 2) CSS ----------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 2rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 1.6rem;
    }
    /* 기존 metric-card + 통일 타이포/높이 적용 */
    .metric-card {
        background: #fff; padding: 1rem; border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #eef2ff; text-align: center;
        display: flex; flex-direction: column; justify-content: center;
        min-height: 110px;
    }
    .metric-value {
        margin: 0; line-height: 1.1;
        font-weight: 700; font-size: 28px; color: #667eea;
        word-break: keep-all; white-space: normal;
    }
    .metric-value.small {
        font-size: 16px; line-height: 1.25;
    }
    .metric-label {
        margin: 6px 0 0 0; font-size: 13px; color: #475569;
    }
    @media (max-width: 1200px){
        .metric-value { font-size: 24px; }
        .metric-value.small { font-size: 15px; }
    }

    .section-divider {
        border-top: 2px solid #667eea; margin: 1.6rem 0 1rem 0;
        padding-top: 0.6rem;
    }
    .insight {
        background: #f0f7ff; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #667eea; margin: 0.6rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- 3) 헬퍼 ----------------
def make_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="맑은고딕", title_font_size=18,
        margin=dict(l=10, r=10, t=55, b=10), legend_title_text=""
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    return fig

def show_insight(title: str, content: str):
    st.markdown(f"""
    <div class="insight">
        <strong>💡 {title}</strong><br>{content}
    </div>
    """, unsafe_allow_html=True)

# ---------------- 4) 헤더 ----------------
st.markdown("""
<div class="main-header">
    <h1>⏱️ 세대와 성별에 따른 K-POP 히트곡의 지속력 차이</h1>
    <p>대상: <b>BTS · BLACKPINK · TXT · AESPA</b> — 체류지표로 롱런 특성을 비교합니다.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- 5) 데이터 로드 ----------------
DEFAULT_PATH = os.path.join("data", "kpop_2010_2025_curated_final.csv")

TEAM_META = {
    "BTS": {"성별": "남자", "세대": "3세대"},
    "BLACKPINK": {"성별": "여자", "세대": "3세대"},
    "TXT (TOMORROW X TOGETHER)": {"성별": "남자", "세대": "4세대"},
    "AESPA": {"성별": "여자", "세대": "4세대"},
}
TEAMS_STD = list(TEAM_META.keys())

def normalize_artist(s):
    if pd.isna(s): return np.nan
    t = str(s).strip()
    tu = t.upper()
    if tu in ["BTS", "BLACKPINK", "TXT", "AESPA"]:
        return tu if tu != "TXT" else "TXT (TOMORROW X TOGETHER)"
    if t.lower() == "aespa":
        return "AESPA"
    return t

def coalesce(df, candidates, newname):
    for c in candidates:
        if c in df.columns:
            df[newname] = df[c]
            return df
    df[newname] = np.nan
    return df

@st.cache_data
def load_csv(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def prep(raw: pd.DataFrame, min_pop=0, yr_min=2015, yr_max=2025):
    df = raw.copy()
    df = coalesce(df, ["artist", "artist_name", "main_artist"], "artist")
    df = coalesce(df, ["track_name", "track", "song"], "track_name")
    df = coalesce(df, ["album_release_date", "release_date"], "album_release_date")

    if "release_year" not in df.columns or df["release_year"].isna().all():
        rel = pd.to_datetime(df["album_release_date"], errors="coerce")
        df["release_year"] = rel.dt.year
    if "popularity" not in df.columns:
        df["popularity"] = np.nan

    df["artist_std"] = df["artist"].apply(normalize_artist)
    df = df[df["artist_std"].isin(TEAMS_STD)].copy()

    rel = pd.to_datetime(df["album_release_date"], errors="coerce")
    now = pd.Timestamp.utcnow().tz_localize(None)
    df["age_years"] = ((now - rel).dt.days / 365).round(2)
    df["staying_index"] = (df["popularity"] / (1 + np.log1p(df["age_years"]))).round(2)

    df["세대"] = df["artist_std"].map(lambda a: TEAM_META[a]["세대"])
    df["성별"] = df["artist_std"].map(lambda a: TEAM_META[a]["성별"])

    df = df.dropna(subset=["staying_index","age_years","popularity","track_name","release_year","세대","성별"])
    df = df[(df["popularity"]>=min_pop) & (df["release_year"].between(yr_min, yr_max))].copy()

    return df.rename(columns={
        "artist_std": "아티스트",
        "track_name": "곡명",
        "album_release_date": "발매일",
        "release_year": "발매연도",
        "popularity": "인기도",
        "age_years": "연식(년)",
        "staying_index": "체류지표",
    }).sort_values(["체류지표","인기도"], ascending=[False, False]).reset_index(drop=True)

raw_df = load_csv(DEFAULT_PATH)
if raw_df is None:
    st.error(f"기본 CSV 파일을 찾을 수 없습니다: {DEFAULT_PATH}")
    st.stop()
data = prep(raw_df)

# ---------------- 7) 메트릭 카드 ----------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="metric-card">'
        f'  <div class="metric-value">{len(data):,}</div>'
        f'  <div class="metric-label">총 곡 수</div>'
        f'</div>', unsafe_allow_html=True)

with c2:
    avg_age = "-" if data.empty else f"{data['연식(년)'].mean():.1f} 년"
    st.markdown(
        f'<div class="metric-card">'
        f'  <div class="metric-value">{avg_age}</div>'
        f'  <div class="metric-label">평균 연식</div>'
        f'</div>', unsafe_allow_html=True)

with c3:
    avg_stay = "-" if data.empty else f"{data['체류지표'].mean():.1f}"
    st.markdown(
        f'<div class="metric-card">'
        f'  <div class="metric-value">{avg_stay}</div>'
        f'  <div class="metric-label">평균 체류지표</div>'
        f'</div>', unsafe_allow_html=True)

with c4:
    st.markdown(
        '<div class="metric-card">'
        '  <div class="metric-value small">BTS·BLACKPINK·TXT·AESPA</div>'
        '  <div class="metric-label">대상 팀</div>'
        '</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------------- 8) 탭 ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① 세대 분포", "② 성별 하락 곡선", "③ 명곡 비율", "④ 코호트 추이", "⑤ 통계검정"
])

# ---------- 탭1: 세대 분포 ----------
with tab1:
    st.subheader("📦 세대별 체류지표 분포")
    st.markdown(r"""
**체류지표(Staying Index)**  
발매 후 시간이 지났음에도 현재 인기도가 얼마나 유지되는지 보기 위한 간단 지표입니다.  
\[
\text{staying\_index} = \frac{\text{popularity}}{1 + \ln(1 + \text{age\_years})}
\]
→ 같은 인기라도 오래된 곡일수록 분모가 커지므로 **진짜 ‘롱런’ 곡**이 상위로 떠오릅니다.
""")
    if data.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        fig = px.box(data, x="세대", y="체류지표", color="세대",
                     title="세대별 Staying Index 분포", points="outliers")
        st.plotly_chart(make_chart(fig), use_container_width=True)

        grp = data.groupby("세대")["체류지표"].agg(평균="mean", 중앙값="median", 표준편차="std", n="size").round(3)
        st.dataframe(grp)
        if "3세대" in grp.index and "4세대" in grp.index:
            st.caption(f"→ 평균 기준: 3세대 {grp.loc['3세대','평균']} vs 4세대 {grp.loc['4세대','평균']}")

# ---------- 탭2: 성별 하락 곡선 ----------
with tab2:
    st.subheader("📉 성별 연식→인기도 하락 곡선")
    if data.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        fig = px.scatter(
            data, x="연식(년)", y="인기도",
            color="성별", symbol="아티스트",
            hover_data=["아티스트", "곡명", "발매연도", "체류지표"],
            title="연식 증가에 따른 인기도 변화 (남/여)"
        )
        # 성별별 1차 회귀선
        for g, gdf in data.groupby("성별"):
            x = gdf["연식(년)"].to_numpy(dtype=float)
            y = gdf["인기도"].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 2:
                b, a = np.polyfit(x[m], y[m], 1)  # y = a + b*x
                xs = np.linspace(x[m].min(), x[m].max(), 100)
                ys = a + b * xs
                fig.add_traces(go.Scatter(x=xs, y=ys, mode="lines", name=f"{g} 회귀선", line=dict(dash="dash")))
        st.plotly_chart(make_chart(fig), use_container_width=True)

        rows = []
        for g, gdf in data.groupby("성별"):
            x = gdf["연식(년)"].astype(float).to_numpy()
            y = gdf["인기도"].astype(float).to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 2:
                b, a = np.polyfit(x[m], y[m], 1)
                rows.append({"성별": g, "기울기(β_age)": round(b, 3), "절편": round(a, 2), "n": int(m.sum())})
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("기울기(β_age)"))
            st.caption("→ |β_age|가 작을수록 시간 경과에 따른 인기도 하락이 완만(지속력 ↑).")

# ---------- 탭3: 명곡 비율 ----------
with tab3:
    st.subheader("🏆 명곡 비율 (체류지표 상위 20%)")
    st.markdown("""
**명곡 정의(본 대시보드 기준)**  
- 전체 체류지표 분포의 **상위 20% (quantile 0.80 이상)** 를 명곡으로 정의합니다.  
- 상대 기준이므로 표본이 바뀌면 컷오프도 함께 변합니다.
""")
    if data.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        cutoff = data["체류지표"].quantile(0.80)
        tmp = data.assign(명곡=lambda d: np.where(d["체류지표"] >= cutoff, 1, 0))

        by_team = tmp.groupby("아티스트")["명곡"].mean().mul(100).round(1).reset_index(name="명곡 비율(%)")
        by_gen = tmp.groupby("세대")["명곡"].mean().mul(100).round(1).reset_index(name="명곡 비율(%)")

        fig1 = px.bar(by_team.sort_values("명곡 비율(%)", ascending=False),
                      x="아티스트", y="명곡 비율(%)",
                      title=f"팀별 명곡 비율 (컷오프={cutoff:.2f})")
        fig2 = px.bar(by_gen.sort_values("명곡 비율(%)", ascending=False),
                      x="세대", y="명곡 비율(%)",
                      title="세대별 명곡 비율")
        st.plotly_chart(make_chart(fig1), use_container_width=True)
        st.plotly_chart(make_chart(fig2), use_container_width=True)

        st.dataframe(by_team)
        st.dataframe(by_gen)
        if set(["3세대", "4세대"]).issubset(by_gen["세대"].unique()):
            g3 = float(by_gen.loc[by_gen["세대"] == "3세대", "명곡 비율(%)"].iloc[0])
            g4 = float(by_gen.loc[by_gen["세대"] == "4세대", "명곡 비율(%)"].iloc[0])
            st.caption(f"→ 명곡 비율: 3세대 {g3}% vs 4세대 {g4}%")

# ---------- 탭4: 코호트 추이 ----------
with tab4:
    st.subheader("📈 연도별 평균 체류지표 추이 (코호트)")
    st.markdown("""
**코호트(Cohort)**: 발매연도로 묶은 집단.  
연도별 평균 체류지표를 비교하면, 세대 교체 전후 **지속력 패턴 변화**를 볼 수 있습니다.
""")
    if data.empty or data["발매연도"].isna().all():
        st.info("연도 정보가 없어 추이를 표시할 수 없습니다.")
    else:
        yearly = data.groupby(["발매연도", "세대"])["체류지표"].mean().reset_index()
        fig = px.line(yearly, x="발매연도", y="체류지표", color="세대", markers=True,
                      title="연도별 평균 Staying Index — 세대 비교")
        st.plotly_chart(make_chart(fig), use_container_width=True)
        st.dataframe(yearly.sort_values(["세대", "발매연도"]))

# ---------- 탭5: 통계검정 ----------
with tab5:
    st.subheader("🧪 3세대 vs 4세대 ‘지속력’ 차이 검정 (Mann–Whitney U)")
    with st.expander("📘 Mann–Whitney U 검정: 무엇이고 왜 쓰나요? (클릭)", expanded=True):
        st.markdown(r"""
**만–휘트니 U**는 두 집단의 **분포 위치(중앙 경향)** 차이를 보는 **비모수** 검정입니다.  
정규성 가정이 어렵고 이상값에 민감할 수 있는 지표(체류지표)에 적합합니다.

- H0: 3세대와 4세대의 체류지표 분포 위치가 동일
- H1: 두 분포의 위치가 다름

**해석**  
- p < 0.05 → 유의미한 차이  
- 효과크기 보조:
  - Cohen’s d (양수면 3세대 > 4세대)
  - Cliff’s delta δ ∈ [-1,1] (양수면 3세대 > 4세대)
""")
    if data.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        g3 = data.loc[data["세대"] == "3세대", "체류지표"].astype(float).dropna().to_numpy()
        g4 = data.loc[data["세대"] == "4세대", "체류지표"].astype(float).dropna().to_numpy()

        if len(g3) >= 2 and len(g4) >= 2:
            mw_stat, mw_p = stats.mannwhitneyu(g3, g4, alternative="two-sided")

            def cohens_d(x, y):
                x, y = np.asarray(x), np.asarray(y)
                nx, ny = len(x), len(y)
                sx, sy = x.std(ddof=1), y.std(ddof=1)
                sp = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / max(nx+ny-2, 1))
                return (x.mean() - y.mean()) / sp if sp > 0 else np.nan

            def cliffs_delta(x, y):
                x, y = np.asarray(x), np.asarray(y)
                gt = sum((xi > y).sum() for xi in x)
                lt = sum((xi < y).sum() for xi in x)
                n_pairs = len(x) * len(y)
                return (gt - lt) / n_pairs if n_pairs > 0 else np.nan

            d_val = cohens_d(g3, g4)      # 3세대 - 4세대
            delta = cliffs_delta(g3, g4)

            res = pd.DataFrame({
                "검정": ["Mann–Whitney U"],
                "통계량": [mw_stat],
                "p값": [mw_p],
                "Cohen's d (3세대-4세대)": [d_val],
                "Cliff's delta": [delta],
                "3세대 평균": [np.mean(g3)],
                "4세대 평균": [np.mean(g4)],
                "3세대 n": [len(g3)],
                "4세대 n": [len(g4)],
            }).round(4)
            st.dataframe(res, use_container_width=True)

            fig = px.box(data, x="세대", y="체류지표", color="세대",
                         title="세대별 체류지표 분포 (검정 보조)", points=False)
            st.plotly_chart(make_chart(fig), use_container_width=True)

            st.caption("→ p<0.05면 분포 차이 유의. d(0.2/0.5/0.8), δ(0.147/0.33/0.474) 기준으로 효과크기 해석. d·δ **양수**면 3세대가 더 큼.")
        else:
            st.info("양 집단의 표본 수가 부족합니다.")
