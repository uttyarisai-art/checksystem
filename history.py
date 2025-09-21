# -*- coding: utf-8 -*-
# CSV（data/history.csv）に蓄積したチェック履歴を一覧表示・簡易分析するページ

import pandas as pd
import streamlit as st
from common import load_history, ensure_data_dirs

def render():
    ensure_data_dirs()
    st.title("📜 チェック履歴")

    df = load_history()
    if df.empty:
        st.info("まだ履歴がありません。先にチェックを実行してください。")
        return

    # 絞り込みフィルタ
    st.subheader("🔎 絞り込み")
    col1, col2, col3 = st.columns(3)
    with col1:
        companies = ["（すべて）"] + sorted([c for c in df["company_name"].dropna().unique().tolist() if str(c).strip()])
        sel_company = st.selectbox("企業名", companies)
    with col2:
        modes = ["（すべて）"] + sorted(df["mode"].dropna().unique().tolist())
        sel_mode = st.selectbox("チェック種別", modes)
    with col3:
        filename = st.text_input("ファイル名に含む文字（部分一致）", value="")

    _df = df.copy()
    if sel_company != "（すべて）":
        _df = _df[_df["company_name"] == sel_company]
    if sel_mode != "（すべて）":
        _df = _df[_df["mode"] == sel_mode]
    if filename.strip():
        _df = _df[_df["filename"].astype(str).str.contains(filename.strip(), na=False)]

    st.subheader("📄 履歴一覧")
    st.dataframe(_df, use_container_width=True, hide_index=True)

    # スコアの推移チャート（審査項目のみを対象）
    st.subheader("📈 スコア推移（審査項目のみ）")
    df_score = df[df["mode"] == "審査項目"].copy()
    try:
        df_score["score"] = pd.to_numeric(df_score["score"], errors="coerce")
        df_score = df_score.dropna(subset=["score"])
        if not df_score.empty:
            # 日時でソート
            df_score = df_score.sort_values("timestamp")
            st.line_chart(df_score.set_index("timestamp")["score"])
        else:
            st.caption("表示可能なスコアがありません。")
    except Exception:
        st.caption("スコアの描画に失敗しました。")

    # ダウンロード
    st.download_button(
        "📥 CSV をダウンロード",
        data=_df.to_csv(index=False),
        file_name="history_filtered.csv",
        mime="text/csv"
    )
