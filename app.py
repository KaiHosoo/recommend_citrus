# app.py
import math, uuid
from typing import List, Dict
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client, Client

st.set_page_config(page_title="柑橘レコメンダ", page_icon="🍊", layout="wide")

# ---------------- 基本設定 ----------------
FEATURES = ["brix", "acid", "bitterness", "aroma", "moisture", "texture"]
ALIASES = {
    "brix": ["brix", "sweet", "sweetness", "sugar"],
    "acid": ["acid", "acidity", "sour", "sourness"],
    "bitterness": ["bitterness", "bitter"],
    "aroma": ["aroma", "smell", "fragrance", "flavor", "flavour"],
    "moisture": ["moisture", "juicy", "juiciness"],
    "texture": ["texture", "elastic", "firmness", "pulpiness"],
}

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    if "season" not in df.columns:
        for cand in ["seasons","season_pref","in_season"]:
            if cand in df.columns: df = df.rename(columns={cand:"season"}); break
    if "image_url" not in df.columns:
        for cand in ["image","img","img_url","photo_url","picture"]:
            if cand in df.columns: df = df.rename(columns={cand:"image_url"}); break
    for std, cands in ALIASES.items():
        if std in df.columns: continue
        for cand in cands:
            if cand in df.columns: df = df.rename(columns={cand: std}); break
    if "name" not in df.columns:
        for cand in ["品種名","citrus_name","item_name","title"]:
            if cand in df.columns: df = df.rename(columns={cand:"name"}); break
        if "name" not in df.columns: df["name"] = [f"item_{i}" for i in range(len(df))]
    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df)+1)
    return df

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize_columns(df)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"必要カラムが足りません: {missing} / 取得={list(df.columns)}")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(1,6)
    if "season" not in df.columns:
        df["season"] = ""
    df["season"] = df["season"].fillna("").astype(str)

    if "image_url" not in df.columns:
        df["image_url"] = ""
    df["image_url"] = df["image_url"].fillna("").astype(str)
    return df.dropna(subset=FEATURES)

def parse_seasons(cell: str) -> List[str]:
    if not cell: return []
    return [s.strip().lower() for s in str(cell).split(",") if s.strip()]

def score_items(df: pd.DataFrame, user_vec: np.ndarray, season_pref: str="", 
                weights: Dict[str,float]|None=None, season_boost: float=0.03) -> pd.DataFrame:
    if weights is None: weights = {k:1.0 for k in FEATURES}
    w = np.array([weights[k] for k in FEATURES], dtype=float)
    max_dist = float(np.sqrt(np.sum((w*5)**2)))
    X = df[FEATURES].to_numpy(float)
    dists = np.sqrt(np.sum(((X - user_vec)*w)**2, axis=1))
    scores = 1.0 - (dists / max_dist)
    add = 0.0
    if season_pref:
        m = df["season"].fillna("").map(lambda s: season_pref.lower() in parse_seasons(s)).to_numpy(bool)
        scores = np.clip(scores + np.where(m, season_boost, 0.0), 0.0, 1.0)
    out = df.copy()
    out["distance"] = dists
    out["score"] = scores
    return out.sort_values(["score","name"], ascending=[False, True]).reset_index(drop=True)

# ---------------- Supabase client ----------------
def get_sb() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])
SB = get_sb()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

def save_user_inputs_to_supabase(nickname: str, sliders: dict, season_pref: str, 
                                 weights: dict, topk: int, ranked_df: pd.DataFrame) -> tuple[bool,str]:
    payload = {
        "session_id": st.session_state["session_id"],
        "nickname": nickname or "",
        "brix": int(sliders["brix"]), "acid": int(sliders["acid"]), "bitterness": int(sliders["bitterness"]),
        "aroma": int(sliders["aroma"]), "moisture": int(sliders["moisture"]), "texture": int(sliders["texture"]),
        "season_pref": season_pref or "",
        "weights": weights,
        "topk": int(topk),
        "top_recos": [
            {"name": row["name"], "score": float(row["score"])}
            for _, row in ranked_df.head(int(topk)).iterrows()
        ],
    }
    try:
        SB.table("user_pref_logs").insert(payload).execute()
        return True, "保存しました。"
    except Exception as e:
        return False, f"保存に失敗しました: {e}"

# ---------------- UI ----------------
st.title("🍊 柑橘レコメンダ（CSV版 + Supabase保存）")

with st.sidebar:
    st.header("1) 好みを入力")
    sliders = {
        "brix": st.slider("甘さ(brix)", 1, 6, 4),
        "acid": st.slider("酸味(acid)", 1, 6, 3),
        "bitterness": st.slider("苦味(bitterness)", 1, 6, 2),
        "aroma": st.slider("香り(aroma)", 1, 6, 3),
        "moisture": st.slider("ジューシーさ(moisture)", 1, 6, 4),
        "texture": st.slider("食感(texture)", 1, 6, 3),
    }
    season_pref = st.selectbox("季節（任意）", ["", "winter", "spring", "summer", "autumn"])
    with st.expander("重み（任意）"):
        weights = {
            "brix": st.number_input("甘さの重み", 0.0, 3.0, 1.0, 0.1),
            "acid": st.number_input("酸味の重み", 0.0, 3.0, 1.0, 0.1),
            "bitterness": st.number_input("苦味の重み", 0.0, 3.0, 1.0, 0.1),
            "aroma": st.number_input("香りの重み", 0.0, 3.0, 1.0, 0.1),
            "moisture": st.number_input("ジューシーさの重み", 0.0, 3.0, 1.0, 0.1),
            "texture": st.number_input("食感の重み", 0.0, 3.0, 1.0, 0.1),
        }
    topk = st.number_input("表示件数", 1, 20, 5)

colL, colR = st.columns([1,2], gap="large")

with colL:
    st.subheader("2) データ読み込み（CSV）")
    path = st.text_input("CSVパス", "citrus_features.csv")
    try:
        df = load_csv(path)
        st.success(f"読み込み成功: {len(df)} 品種")
        st.dataframe(df[["name","season"]+FEATURES], use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"読み込み失敗: {e}")
        st.stop()

with colR:
    st.subheader("3) 結果")
    user_vec = np.array([sliders[k] for k in FEATURES], dtype=float)
    ranked = score_items(df, user_vec, season_pref=season_pref, weights=weights)
    for i, row in ranked.head(int(topk)).iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([1,3])
            with c1:
                if isinstance(row.get("image_url",""), str) and row["image_url"].strip():
                    st.image(row["image_url"], use_container_width=True)
                st.metric("スコア", f"{row['score']:.3f}")
            with c2:
                st.markdown(f"### {i+1}. {row['name']}")
                st.caption(f"季節: {row['season'] or '-'}")

st.divider()
st.subheader("4) 入力をSupabaseに保存")
with st.form("save"):
    nickname = st.text_input("ニックネーム（任意）", "")
    agree = st.checkbox("入力内容をSupabaseに保存することに同意します", True)
    submitted = st.form_submit_button("保存")
    if submitted:
        if not agree:
            st.warning("保存には同意が必要です。")
        else:
            ok, msg = save_user_inputs_to_supabase(nickname, sliders, season_pref, weights, int(topk), ranked)
            (st.success if ok else st.error)(msg)