import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------
# 기본 설정 (새 코드 적용)
# ------------------------
st.set_page_config(
    page_title="K-POP 인기곡 분석", # 페이지 제목은 기존 내용 유지
    page_icon="🎵",
    layout="wide"
)

# ------------------------
# 스타일 정의 (새 코드 적용)
# ------------------------
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


# ------------------------
# 유틸 함수 (새 코드로 교체 및 추가)
# ------------------------
def make_chart(fig):
    """차트 스타일 통일"""
    # 폰트가 없는 환경을 고려하여 font_family는 주석 처리. 필요시 해제하여 사용하세요.
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        # font_family="맑은고딕", 
        title_font_size=16
    )
    return fig

def show_insight(title, content):
    """인사이트 박스"""
    st.markdown(f"""
    <div class="insight">
        <strong>💡 {title}</strong><br>{content}
    </div>
    """, unsafe_allow_html=True)

# ------------------------
# 데이터 로드 (기존 코드 유지)
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data\kpop_2010_2025_curated_final.csv")
    # 사전 데이터 처리
    df['duration_sec'] = df['duration_ms'] / 1000
    bins = [0, 120, 180, 240, 300, 600]
    labels = ['0-2분', '2-3분', '3-4분', '4-5분', '5분+']
    df['duration_bin'] = pd.cut(df['duration_sec'], bins=bins, labels=labels, right=False)
    
    df['is_collab'] = df['track_name'].str.contains('feat\.|ft\.', case=False, regex=True)
    df['collab_label'] = df['is_collab'].map({True: '협업곡', False: '단독곡'})

    # 날짜 및 계절 컬럼 생성
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_month'] = df['release_date'].dt.month
    
    def month_to_season(month):
        if month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        elif month in [9, 10, 11]: return 'Fall'
        else: return 'Winter'
    df['season'] = df['release_month'].apply(month_to_season)
    
    return df

df = load_data()

# ------------------------
# 색상 팔레트 정의 (기존 코드 유지)
# ------------------------
artists = df['artist'].unique()
colors = ["#7C4DFF", "#E91E63", "#FF9800", "#00BCD4"]
artist_palette = dict(zip(artists, colors))

season_palette = {
    'Spring': '#10B981', 'Summer': '#EF4444',
    'Fall': '#F59E0B', 'Winter': '#6B7280'
}

collab_palette = {'단독곡': '#a6a6a6', '협업곡': '#2ca02c'}


# ------------------------
# 헤더 (새 코드로 교체)
# ------------------------
st.markdown("""
<div class="main-header">
    <h1>🎵 K-POP 인기곡 분석 대시보드</h1>
    <p>인기곡 기준: <strong>popularity >= 60</strong></p>
</div>
""", unsafe_allow_html=True)

# ------------------------
# Metric Cards (새로운 스타일 적용)
# ------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #667eea; margin: 0;">{len(df)}</h3>
        <p style="margin: 0;">전체 곡 수</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #48bb78; margin: 0;">{df['artist'].nunique()}</h3>
        <p style="margin: 0;">아티스트 수</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #ed8936; margin: 0;">{df['popularity'].mean():.1f}</h3>
        <p style="margin: 0;">평균 인기도</p>
    </div>
    """, unsafe_allow_html=True)
    

# ------------------------
# 인기곡 필터링 (기존 코드 유지)
# ------------------------
popular_songs = df[df['popularity'] >= 60].copy()

# ------------------------
# 탭 구성 (기존 코드 유지)
# ------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 연도별 인기곡", "🍂 계절별 인기곡", "📅 연도×계절 트렌드",
    "🤝 협업 vs 단독곡", "⏱️ 곡 길이 비교", "📈 곡 길이 vs 인기도", "🎵인사이트"
])

# ------------------------
# Tab1 ~ Tab6 (기존 차트 코드 전체 유지)
# ------------------------
with tab1:
    fig = px.histogram(
        popular_songs, x="release_year", color="artist",
        color_discrete_map=artist_palette, title="그룹별 연도별 인기곡 발매 수 (popularity >= 60)",
        barmode='group', labels={'release_year': '발매 연도', 'count': '인기곡 수', 'artist': '아티스트'}
    )
    st.plotly_chart(make_chart(fig), use_container_width=True)



with tab2:
    fig = px.histogram(
        popular_songs, x="season", color="artist",
        color_discrete_map=artist_palette, title="그룹별 계절별 인기곡 발매 수 (popularity >= 60)",
        barmode='group', category_orders={"season": ['Spring', 'Summer', 'Fall', 'Winter']},
        labels={'season': '계절', 'count': '인기곡 수', 'artist': '아티스트'}
    )
    st.plotly_chart(make_chart(fig), use_container_width=True)
  


with tab3:
    release_counts = popular_songs.groupby(['artist', 'release_year', 'season']).size().reset_index(name='count')
    fig = px.bar(
        release_counts, x="release_year", y="count", color="season",
        facet_col="artist", facet_col_wrap=2, color_discrete_map=season_palette,
        category_orders={"season": ['Spring', 'Summer', 'Fall', 'Winter']},
        title="그룹별 연도별 계절별 인기곡 수 (popularity >= 60)",
        labels={'release_year': '발매 연도', 'count': '인기곡 수', 'season': '계절'}
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(make_chart(fig), use_container_width=True)
  

with tab4:
    collab_popularity = df.groupby(['artist', 'collab_label'])['popularity'].mean().reset_index()
    fig = px.bar(
        collab_popularity, x="artist", y="popularity", color="collab_label",
        color_discrete_map=collab_palette, barmode='group',
        title="그룹별 단독곡 vs 협업곡 평균 인기도 비교",
        labels={'artist': '그룹', 'popularity': '평균 인기도', 'collab_label': '곡 유형'}
    )
    st.plotly_chart(make_chart(fig), use_container_width=True)


with tab5:
    fig = px.box(
        popular_songs, x="artist", y="duration_sec", color="artist",
        color_discrete_map=artist_palette, title="그룹별 인기곡 곡 길이 비교 (popularity >= 60)",
        labels={'artist': '그룹', 'duration_sec': '곡 길이 (초)'}
    )
    st.plotly_chart(make_chart(fig), use_container_width=True)
  

with tab6:
    line_data = df.groupby(['artist', 'duration_bin'], observed=True)['popularity'].mean().reset_index()
    fig = px.line(
        line_data, x='duration_bin', y='popularity', color='artist',
        facet_col='artist', facet_col_wrap=2, color_discrete_map=artist_palette,
        markers=True, title='그룹별 곡 길이에 따른 평균 인기도',
        labels={'duration_bin': '곡 길이 구간', 'popularity': '평균 인기도', 'artist': '아티스트'}
    )
    fig.update_yaxes(range=[0, 100])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(make_chart(fig), use_container_width=True)

with tab7:
    st.subheader("🎵 분석 인사이트 정리")
    
    show_insight("계절별 발매 패턴",
                 "겨울 시즌에는 인기곡 발매 비중이 다른 계절 대비 눈에 띄게 낮습니다.")
    show_insight("곡 길이와 인기도의 상관성",
                 "평균적으로 3~4분대 곡에서 인기도가 높게 나타납니다.")
    show_insight("아티스트별 특이 패턴",
                 "<ul>"
                 "<li><b>BTS</b>: 여름 시즌에 인기곡 집중</li>"
                 "<li><b>BLACKPINK</b>: 가을 발매곡 많음, 단독곡 인기도 높음</li>"
                 "<li><b>aespa</b>: 협업곡 인기도가 단독곡보다 높음</li>"
                 "<li><b>TXT</b>: 최근 발매곡 높은 인기도</li>"
                 "</ul>")

