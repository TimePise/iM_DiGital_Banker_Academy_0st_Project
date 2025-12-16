# utils_no_audio.py
import os, time
from datetime import date
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# 1) 환경변수 로드 (.env 파일에서 Client ID/Secret 읽기)
load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Spotify API 인증
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
)

# 2) 설정
ARTISTS = {
    "BTS":"3Nrfpe0tUJi4K4DXYWgMUX",
    "Blackpink":"41MozSoPIsD1dJM0CLPjZF",
    "TXT": "0ghlgldX5Dd6720Q3qFyQB",
    "aespa": "6YVMFz59CuY7ngCxTxjpxE",
}
YEAR_START, YEAR_END = 2010, 2025
SLEEP = 0.15   # 요청 간 대기 (API rate limit 방지)

# 리스트를 50개 단위로 끊는 유틸 (Spotify API 제한 때문)
def batched(xs, n=50):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

# 발매 연도가 지정 범위 안에 있는지 체크
def in_year_range(release_date: str) -> bool:
    try:
        y = int(release_date.split("-")[0])
        return YEAR_START <= y <= YEAR_END
    except Exception:
        return False

# 특정 아티스트의 앨범 불러오기 (album, single만)
def fetch_albums_in_range(artist_id: str):
    seen, albums, offset = set(), [], 0
    while True:
        res = sp.artist_albums(
            artist_id,
            include_groups="album,single",
            country="KR",
            limit=50,
            offset=offset,
        )
        items = res.get("items", [])
        if not items: break
        for a in items:
            if a["id"] not in seen and in_year_range(a["release_date"]):
                albums.append(a)
                seen.add(a["id"])
        offset += len(items)
        time.sleep(SLEEP)
    return albums

# 앨범 ID로 해당 앨범의 모든 트랙 ID 가져오기
def fetch_track_ids(album_id: str):
    ids, offset = [], 0
    while True:
        res = sp.album_tracks(album_id, limit=50, offset=offset)
        items = res.get("items", [])
        if not items: break
        ids += [t["id"] for t in items if t.get("id")]
        offset += len(items)
        time.sleep(SLEEP)
    return ids

# 3) 실행
rows = []
today = date.today()

for artist, aid in ARTISTS.items():
    albums = fetch_albums_in_range(aid)         # 아티스트의 앨범 가져오기
    track_ids = []
    for alb in albums:
        track_ids += fetch_track_ids(alb["id"]) # 각 앨범에서 트랙 수집
    track_ids = list(dict.fromkeys(track_ids))  # track_id 중복 제거

    # 트랙 정보를 50개 단위로 가져오기
    for chunk in batched(track_ids, 50):
        tracks = sp.tracks(chunk)["tracks"]
        for t in tracks:
            alb = t["album"]
            if not in_year_range(alb["release_date"]): continue
            release_year = int(alb["release_date"].split("-")[0])
            song_age = today.year - release_year
            rows.append({
                "artist": artist,
                "artist_id": aid,
                "album_id": alb["id"],
                "album_name": alb.get("name"),
                "album_type": alb.get("album_type"),
                "track_id": t["id"],
                "track_name": t["name"],
                "isrc": (t.get("external_ids") or {}).get("isrc"),
                "release_date": alb["release_date"],
                "release_year": release_year,
                "popularity": t.get("popularity"),
                "duration_ms": t.get("duration_ms"),
                "duration_min": (t.get("duration_ms") or 0) / 60000,
                "explicit": t.get("explicit"),
                "disc_number": t.get("disc_number"),
                "track_number": t.get("track_number"),
                "song_age_years": song_age,
                "staying_index": (t.get("popularity") or 0) / (1 + song_age)
            })
        time.sleep(SLEEP)

# 4) DataFrame 변환 및 저장
df = pd.DataFrame(rows).drop_duplicates(subset=["track_id"])
df.to_csv("kpop_2010_2025_curated.csv", index=False, encoding="utf-8-sig")
print("✅ saved:", df.shape, "rows -> kpop_2010_2025_curated.csv")
