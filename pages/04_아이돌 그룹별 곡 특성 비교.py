# pages/05_group_features.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------- Page Setup --------------------
st.set_page_config(page_title="아이돌 그룹별 곡 특성", page_icon="🎵", layout="wide")

# -------------------- CSS --------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 1.6rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 1.4rem;
    }
    .metric-card {
        background: #fff; padding: 1rem; border-radius: 10px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08); text-align: center;
        border: 1px solid #eef2ff;
    }
    .section-divider {
        border-top: 2px solid #667eea; margin: 1.6rem 0 0.8rem 0; padding-top: 0.6rem;
    }
    .insight {
        background: #f0f7ff; padding: 1rem; border-radius: 10px;
        border-left: 5px solid #667eea; margin: 0.6rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Helpers --------------------
def make_chart(fig):
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Malgun Gothic", title_font_size=16,
        legend_title_text=None, margin=dict(t=50, r=10, b=20, l=10)
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    return fig

def show_insight(title, content):
    st.markdown(f"""<div class="insight"><strong>💡 {title}</strong><br>{content}</div>""", unsafe_allow_html=True)

def agg_table(df, by, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols: return pd.DataFrame()
    return df.groupby(by, dropna=False)[cols].agg(["count","mean","median","min","max"]).round(3)

# -------------------- Load & Prepare --------------------
CSV_PATH = "data/kpop_2010_2025_curated_final.csv"
TARGET_GROUPS = ["BTS", "Blackpink", "Bigbang", "TXT", "TOMORROW X TOGETHER", "aespa"]
GENERATION_MAP = {
    "BTS": "3rd", "Blackpink": "3rd", "Bigbang": "2nd",
    "TXT": "4th", "TOMORROW X TOGETHER": "4th", "aespa": "4th"
}
ARTIST_COLORS = {
    "BTS": "#7C4DFF", "Blackpink": "#E91E63", "Bigbang": "#FF9800",
    "TXT": "#2196F3", "TOMORROW X TOGETHER": "#2196F3", "aespa": "#9C27B0"
}
GEN_COLORS = {"2nd": "#667eea", "3rd": "#764ba2", "4th": "#10b981"}

df = pd.read_csv(CSV_PATH)
df = df[df["artist"].isin(TARGET_GROUPS)].copy()
if df.empty:
    st.error("선택된 아티스트(BTS / Blackpink / Bigbang / TXT / aespa) 데이터가 비어 있습니다. CSV를 확인하세요.")
    st.stop()

# 타입 정리
for c in ["release_year", "duration_min", "duration_ms", "popularity"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# duration_min 보완
if "duration_min" not in df.columns and "duration_ms" in df.columns:
    df["duration_min"] = df["duration_ms"] / 60000.0

# 메타 파생
df["generation"] = df["artist"].map(GENERATION_MAP)
CURRENT_YEAR = pd.Timestamp.now().year
df["song_age_years"] = (CURRENT_YEAR - df["release_year"]).clip(lower=0) if "release_year" in df.columns else np.nan

# 연도 보정 Z-인기도
if {"release_year","popularity"}.issubset(df.columns) and df["popularity"].notna().any():
    year_stats = df.groupby("release_year")["popularity"].agg(["mean","std"]).rename(columns={"mean":"y_mean","std":"y_std"})
    df = df.merge(year_stats, left_on="release_year", right_index=True, how="left")
    df["popularity_z_by_year"] = (df["popularity"] - df["y_mean"]) / df["y_std"].replace(0, np.nan)
else:
    df["popularity_z_by_year"] = np.nan

# 오래된 히트곡(≥5년 & 상위 인기 or Z≥+1)
pop_q75 = df["popularity"].quantile(0.75) if "popularity" in df else np.nan
df["is_long_run_hit"] = (
    (df["song_age_years"] >= 5)
    & (
        (df["popularity"] >= pop_q75)
        | (df["popularity_z_by_year"] >= 1.0)
    )
)
df["legacy_hit_score"] = np.where(
    df["popularity"] >= 70,
    np.where(
        df["popularity_z_by_year"].notna(),
        df["popularity_z_by_year"],
        (df["popularity"] - df["popularity"].mean()) / (df["popularity"].std(ddof=0) or np.nan)
    ) * np.log1p(df["song_age_years"].fillna(0)),
    np.nan
)

# -------------------- Header --------------------
st.markdown("""
<div class="main-header">
  <h2 style="margin:0;">🎵아이돌 그룹별 곡 특성은 어떻게 구분되는가?</h2>
  <p style="margin:4px 0 0 0;">분석 범위: BTS, Blackpink, Bigbang, TXT, aespa · 기준: 세대(3rd vs 4th) 비교</p>
</div>
""", unsafe_allow_html=True)

# -------------------- KPI (cards) --------------------
k1,k2,k3 = st.columns(3)
with k1:
    st.markdown(f"""<div class="metric-card"><h3 style="color:#667eea;margin:.2rem 0;">{len(df):,}</h3><p style="margin:0;">총 트랙 수</p></div>""", unsafe_allow_html=True)
with k2:
    v = f"{df['popularity'].mean():.1f}" if "popularity" in df else "N/A"
    st.markdown(f"""<div class="metric-card"><h3 style="color:#667eea;margin:.2rem 0;">{v}</h3><p style="margin:0;">평균 인기도</p></div>""", unsafe_allow_html=True)
with k3:
    v = int(df["is_long_run_hit"].sum()) if "is_long_run_hit" in df else 0
    st.markdown(f"""<div class="metric-card"><h3 style="color:#667eea;margin:.2rem 0;">{v}</h3><p style="margin:0;">오래된 히트곡 수</p></div>""", unsafe_allow_html=True)

# -------------------- Summary Tables --------------------
SUMMARY_COLS = ["popularity","duration_min","song_age_years","popularity_z_by_year"]
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("아티스트별 요약")
st.dataframe(agg_table(df, ["artist"], SUMMARY_COLS), use_container_width=True)

st.subheader("세대별 요약")
gen_tbl = agg_table(df, ["generation"], SUMMARY_COLS)
gen_tbl = gen_tbl.reindex(["2nd","3rd","4th"], level=0).dropna(how="all")
st.dataframe(gen_tbl, use_container_width=True)

# -------------------- Tabs (Charts + Tab-only Insights) --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① 연도별 활동 추세", "② 인기도 분포 (세대)", "③ 곡 길이 vs 인기도",
    "④ 연도 대비 인기(Z) 분포", "⑤ 오래된 히트곡"
])

with tab1:
    if "release_year" in df.columns:
        fig = px.histogram(df, x="release_year", color="artist", barmode="group",
                           color_discrete_map=ARTIST_COLORS, title="연도별 곡 수 분포")
        st.plotly_chart(make_chart(fig), use_container_width=True)

        # Tab1 Insight
        yr_counts = df["release_year"].value_counts().sort_index()
        peak_year = int(yr_counts.idxmax()) if not yr_counts.empty else None
        per_artist_peak = (
            df.groupby(["artist","release_year"])["track_name"].count()
              .reset_index().sort_values(["artist","track_name"], ascending=[True,False])
              .groupby("artist").first()
        ) if "track_name" in df.columns else pd.DataFrame()
        lines = []
        if peak_year:
            lines.append(f"- 전체 최다 발매 연도: {peak_year}년")
        if not per_artist_peak.empty:
            for a, row in per_artist_peak.iterrows():
                lines.append(f"- {a} 최다 발매 연도: {int(row['release_year'])}년 ({int(row['track_name'])}곡)")
        show_insight("연도 추세 인사이트", "<br>".join(lines) if lines else "추출 가능한 인사이트가 없습니다.")

with tab2:
    if "popularity" in df.columns:
        fig = px.box(df, x="generation", y="popularity", color="generation",
                     color_discrete_map=GEN_COLORS, points="all", title="세대별 인기도 분포")
        st.plotly_chart(make_chart(fig), use_container_width=True)

        # Tab2 Insight
        pop_by_gen = df.groupby("generation")["popularity"].mean().round(1)
        existing = [g for g in ["2nd","3rd","4th"] if g in pop_by_gen.index]
        lines = [f"- {g} 세대 평균 인기도: {pop_by_gen[g]}" for g in existing]
        if existing:
            best = pop_by_gen[existing].idxmax()
            lines.append(f"- 평균 인기도 우세: {best} 세대")
        show_insight("인기도 인사이트", "<br>".join(lines) if lines else "인기도 계산 데이터가 부족합니다.")

with tab3:
    has_cols = {"duration_min","popularity"}.issubset(df.columns)
    valid_rows = df[["duration_min","popularity"]].dropna().shape[0] if has_cols else 0
    if has_cols and valid_rows > 0:
        fig = px.scatter(
            df, x="duration_min", y="popularity", color="artist",
            color_discrete_map=ARTIST_COLORS,
            hover_data=[c for c in ["track_name","album_name","release_year"] if c in df.columns],
            title="곡 길이(분) vs 인기도"
        )
        st.plotly_chart(make_chart(fig), use_container_width=True)

        # Tab3 Insight (상관)
        corr = df[["duration_min","popularity"]].dropna().corr().iloc[0,1]
        lines = [f"- 전체 상관계수(길이→인기도): {corr:.2f}"]
        by_gen_corr = []
        if "generation" in df.columns:
            for g, sub in df.groupby("generation"):
                if sub[["duration_min","popularity"]].dropna().shape[0] >= 3:
                    c = sub[["duration_min","popularity"]].corr().iloc[0,1]
                    by_gen_corr.append((g, c))
            by_gen_corr.sort(key=lambda x: str(x[0]))
            lines += [f"- {g} 세대 상관: {c:.2f}" for g,c in by_gen_corr]
        show_insight("길이-인기도 인사이트", "<br>".join(lines))
    else:
        st.info("곡 길이 또는 인기도에 유효한 데이터가 부족해 산점도를 표시할 수 없습니다. (duration_min/duration_ms 확인)")

with tab4:
    if "popularity_z_by_year" in df.columns and df["popularity_z_by_year"].notna().any():
        fig = px.box(df, x="generation", y="popularity_z_by_year", color="generation",
                     color_discrete_map=GEN_COLORS, points="all", title="세대별 연도 대비 인기(Z) 분포")
        st.plotly_chart(make_chart(fig), use_container_width=True)

        # Tab4 Insight
        z_by_gen = df.groupby("generation")["popularity_z_by_year"].mean().round(2)
        share_above1 = df.assign(flag=(df["popularity_z_by_year"]>=1)).groupby("generation")["flag"].mean().round(2)
        lines = []
        for g in ["2nd","3rd","4th"]:
            if g in z_by_gen.index:
                lines.append(f"- {g} 세대 평균 Z: {z_by_gen[g]}")
        for g in ["2nd","3rd","4th"]:
            if g in share_above1.index:
                lines.append(f"- {g} 세대 Z≥+1 비중: {share_above1[g]*100:.0f}%")
        show_insight("연도 보정 인기도 인사이트", "<br>".join(lines) if lines else "Z-인기도 계산 데이터가 부족합니다.")
    else:
        st.info("연도 대비 인기(Z) 계산을 위한 데이터가 부족합니다. release_year / popularity를 확인하세요.")

with tab5:
    legacy_df = df[df["is_long_run_hit"]].copy()
    c1, c2 = st.columns(2)

    with c1:
        if not legacy_df.empty:
            counts = legacy_df.groupby("artist")["track_name"].count().reset_index().rename(columns={"track_name":"long_run_hits"})
            fig = px.bar(counts.sort_values("long_run_hits", ascending=False),
                         x="artist", y="long_run_hits", color="artist",
                         color_discrete_map=ARTIST_COLORS, text="long_run_hits",
                         title="아티스트별 오래된 히트곡 수")
            st.plotly_chart(make_chart(fig), use_container_width=True)
        else:
            st.info("오래된 히트곡 조건을 만족하는 곡이 없습니다.")

    with c2:
        if not legacy_df.empty:
            gen_counts = (legacy_df.groupby("generation")["track_name"]
                          .count().reindex(["2nd","3rd","4th"]).fillna(0)
                          .reset_index().rename(columns={"track_name":"long_run_hits"}))
            fig = px.bar(gen_counts, x="generation", y="long_run_hits", color="generation",
                         color_discrete_map=GEN_COLORS, text="long_run_hits",
                         title="세대별 오래된 히트곡 수")
            st.plotly_chart(make_chart(fig), use_container_width=True)

    cols = [c for c in ["artist","track_name","album_name","release_year","song_age_years","popularity","popularity_z_by_year","legacy_hit_score"] if c in legacy_df.columns]
    top = (legacy_df.sort_values(["legacy_hit_score","popularity"], ascending=[False, False])
                   .loc[:, cols].head(10).reset_index(drop=True))
    st.markdown("**Top 10 오래된 히트곡 (legacy_hit_score 기준)**")
    st.dataframe(top, use_container_width=True)

    # Tab5 Insight
    lines = []
    if not legacy_df.empty:
        total_hits = int(legacy_df.shape[0])
        top_artist = legacy_df["artist"].value_counts().idxmax()
        top_artist_n = int(legacy_df["artist"].value_counts().max())
        best_gen = legacy_df["generation"].value_counts().idxmax()
        lines.append(f"- 총 오래된 히트곡: {total_hits}곡")
        lines.append(f"- 아티스트 최다 보유: {top_artist} ({top_artist_n}곡)")
        lines.append(f"- 세대 최다 보유: {best_gen}")
        top_row = top.iloc[0] if not top.empty else None
        if top_row is not None:
            yr = int(top_row.get("release_year")) if pd.notna(top_row.get("release_year")) else "N/A"
            lines.append(f"- 대표 사례: {top_row.get('artist')} – {top_row.get('track_name')} ({yr}년)")
    show_insight("오래된 히트곡 인사이트", "<br>".join(lines) if lines else "조건에 맞는 히트곡이 없습니다.")