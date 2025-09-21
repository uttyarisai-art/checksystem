# -*- coding: utf-8 -*-
# PDF アップロード→RAG（TF-IDF検索）→GPT-4系で評価/誤字脱字チェック
# 結果と得点は CSV（data/history.csv）に追記して、[history]で閲覧可能

import io
import json
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from common import (
    extract_text_from_pdf,
    chunk_text_for_japanese,
    build_tfidf_index,
    retrieve_top_k,
    load_prompts,
    call_openai_with_context,
    append_history,
    guess_company_name_from_text,
    ensure_data_dirs,
    parse_score_safely
)


def _render_settings_box():
    """チェックの共通設定 UI。返り値は辞書。"""
    with st.expander("⚙️ 高度な設定（必要な場合のみ）", expanded=False):
        top_k = st.slider("コンテキストとして LLM に渡すチャンク数（Top-K）", 3, 12, 6)
        chunk_chars = st.slider("チャンク最大文字数", 500, 2000, 1200, step=100)
        overlap = st.slider("チャンクのオーバーラップ文字数", 0, 600, 200, step=50)
        model = st.text_input("使用モデル（空欄でデフォルト）", value="")
    return {
        "top_k": top_k,
        "chunk_chars": chunk_chars,
        "overlap": overlap,
        "model": model.strip() or None
    }


def _render_result_box(title: str, content: str, parsed_json: dict | None):
    """LLM の出力を表示。JSONに見えれば展開、無理なら原文表示。"""
    st.subheader(title)
    col1, col2 = st.columns([2, 1])

    with col1:
        if parsed_json:
            st.success("構造化出力（JSON）を検出しました。")
            st.json(parsed_json)
        else:
            st.info("構造化（JSON）で受け取れませんでした。原文を表示します。")
            st.write(content)

    with col2:
        # スコア抽出の試行（審査項目モードを想定）
        score = parse_score_safely(parsed_json or content)
        st.metric("抽出スコア（推定）", value="-" if score is None else f"{score} 点")
        return score


def render():
    """チェック画面の描画（初期表示ページ）"""
    ensure_data_dirs()
    st.title("✅ 事業計画書チェック")

    # チェック種別の切替（サイドバーではなく、画面内で切替）
    mode = st.radio(
        "チェック種別",
        options=["criteria", "typo"],
        format_func=lambda x: "審査項目チェック" if x == "criteria" else "誤字脱字・整合性チェック",
        horizontal=True
    )

    # PDF のアップローダ
    uploaded = st.file_uploader("PDFファイルをアップロード（10〜20ページ想定）", type=["pdf"])

    # 企業名（PDF から推定 or 手入力）
    company_name = st.text_input("企業名（空欄ならPDFから自動推定を試行）", value="")

    # 設定
    settings = _render_settings_box()

    # 実行ボタン
    run = st.button("🚀 この内容でチェックを実行", use_container_width=True, type="primary")

    if not run:
        st.caption("※ PDF をアップロードし、ボタンを押すとチェックが始まります。")
        return

    if not uploaded:
        st.error("PDF がアップロードされていません。先にファイルを選択してください。")
        return

    # PDF → 全文テキスト抽出
    with st.spinner("PDF を解析しています…"):
        try:
            pdf_bytes = uploaded.read()
            text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
        except Exception as e:
            st.error(f"PDF の読み込みでエラーが発生しました: {e}")
            return

    if not text or len(text.strip()) == 0:
        st.error("PDF からテキストを抽出できませんでした（スキャンPDFの可能性）。OCRを通して再試行してください。")
        return

    # 企業名の推定
    if not company_name.strip():
        company_name = guess_company_name_from_text(text) or "（企業名不明）"
    st.write(f"推定企業名: **{company_name}**")

    # チャンク化 → TF-IDF インデックス作成
    with st.spinner("RAG 用のインデックスを作成しています…"):
        chunks = chunk_text_for_japanese(text, max_chars=settings["chunk_chars"], overlap=settings["overlap"])
        vectorizer, matrix = build_tfidf_index(chunks)

    # プロンプト取得
    prompts = load_prompts()
    if mode == "criteria":
        task_prompt = prompts["criteria_prompt"]
        title = "審査項目チェック結果"
        system_hint = (
            "あなたは日本の中小企業向け補助金（例：ものづくり補助金）の審査員AIです。"
            "与えられたコンテキスト（申請書抜粋）に基づき、評価基準に沿って厳格に判定します。"
            "出力はできる限り JSON 形式で返してください。"
        )
    else:
        task_prompt = prompts["typo_prompt"]
        title = "誤字脱字・整合性チェック結果"
        system_hint = (
            "あなたは日本語文書の校正AIです。誤字脱字、表記ゆれ、数値・単位の不整合、社名・人名の不一致、"
            "日付の矛盾などを、コンテキスト（申請書抜粋）に基づいて検出し、提案を返してください。"
            "出力は可能なら JSON 形式にしてください。"
        )

    # RAG：プロンプトからクエリを作り、関連チャンクを検索
    with st.spinner("関連する記述を検索しています…"):
        # シンプルに「タスクプロンプトの冒頭300字」をクエリとして使用（実需なら観点ごとに複数クエリ推奨）
        query = task_prompt[:300]
        top_chunks_idx, top_chunks = retrieve_top_k(query, vectorizer, matrix, chunks, k=settings["top_k"])

    # LLM 呼び出し
    with st.spinner("LLM（GPT-4系）で評価しています…"):
        llm_text = call_openai_with_context(
            system_prompt=system_hint,
            user_task_prompt=task_prompt,
            context_chunks=top_chunks,
            model_override=settings["model"]
        )

    # 出力の表示
    parsed_json = None
    if llm_text:
        try:
            parsed_json = json.loads(llm_text)
        except Exception:
            parsed_json = None

    score = _render_result_box(title, llm_text or "(出力なし)", parsed_json)

    # 参照した抜粋（人間の根拠確認用）
    with st.expander("🔍 LLM に渡した参照抜粋（Top-K）", expanded=False):
        for i, (idx, ch) in enumerate(zip(top_chunks_idx, top_chunks), start=1):
            st.markdown(f"**[{i}] チャンク #{idx}**")
            st.write(ch)

    # 履歴 CSV への保存
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_history(
        timestamp=timestamp,
        company_name=company_name,
        score=score if score is not None else "",
        mode="審査項目" if mode == "criteria" else "誤字脱字",
        filename=getattr(uploaded, "name", "uploaded.pdf")
    )

    # ダウンロード用（JSON/Markdown）
    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            "📥 解析結果（JSON or 原文）を保存",
            data=(json.dumps(parsed_json, ensure_ascii=False, indent=2) if parsed_json else (llm_text or "")),
            file_name=f"analysis_{mode}_{timestamp.replace(' ','_')}.{'json' if parsed_json else 'txt'}",
            mime="application/json" if parsed_json else "text/plain"
        )
    with colB:
        st.success("履歴に記録しました。サイドバーの📜 履歴から一覧表示できます。")
