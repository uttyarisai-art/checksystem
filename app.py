# -*- coding: utf-8 -*-
# 事業計画書チェックシステム - メイン起動ファイル
# Streamlit 単一アプリ上で、サイドバーのラジオ切替により
# [check] / [history] / [prompt] の3画面を表示する。

import os
import streamlit as st

# ローカルモジュール読み込み（同一ディレクトリ想定）
import check
import history
import prompt
from common import ensure_data_dirs

# ページ全体の設定
st.set_page_config(
    page_title="事業計画書チェックシステム",
    page_icon="🧭",
    layout="wide"
)

# データ格納用のディレクトリ作成（存在しない場合のみ）
ensure_data_dirs()

# サイドバー：アプリ全体のナビゲーション
st.sidebar.title("📂 メニュー")
page = st.sidebar.radio(
    "ページ切替",
    options=["check", "history", "prompt"],
    format_func=lambda x: {"check": "✅ チェック", "history": "📜 履歴", "prompt": "🧩 プロンプト編集"}[x],
    index=0  # 初期表示を check にする
)

# メイン画面の切替
if page == "check":
    check.render()
elif page == "history":
    history.render()
elif page == "prompt":
    prompt.render()

# フッター情報（使い方のヒント）
st.sidebar.markdown("---")
st.sidebar.caption(
    "🔑 OpenAI APIキーは環境変数 `OPENAI_API_KEY` に設定してください。\n"
    "⚙️ モデルは環境変数 `OPENAI_MODEL`（省略時 gpt-4o-mini）を使用します。"
)
