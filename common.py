# -*- coding: utf-8 -*-
# アプリ全体で共有する関数群：入出力、RAG、LLM呼び出し、履歴保存など

import json
import os
import re
from datetime import datetime
from io import BytesIO
from typing import List, Tuple

import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import dotenv
dotenv.load_dotenv()

# OpenAI v1 クライアント
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # importエラーでもアプリは起動し、UIから注意喚起する


# -----------------------------
# パス/ディレクトリ周り
# -----------------------------
def data_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "data")


def prompts_path() -> str:
    return os.path.join(data_dir(), "prompts.json")


def history_path() -> str:
    return os.path.join(data_dir(), "history.csv")


def ensure_data_dirs():
    os.makedirs(data_dir(), exist_ok=True)
    # プロンプト初期ファイルの用意
    if not os.path.exists(prompts_path()):
        save_prompts(default_prompts())
    # 履歴CSVの初期化
    if not os.path.exists(history_path()):
        pd.DataFrame(columns=["timestamp", "company_name", "score", "mode", "filename"]).to_csv(history_path(), index=False)


# -----------------------------
# プロンプトの既定値 / 読み書き
# -----------------------------
def default_prompts() -> dict:
    """初期状態のプロンプト（必要に応じて自由に編集可能）"""
    return {
        "criteria_prompt": (
            "あなたは日本の中小企業向け補助金（例：ものづくり補助金）の審査員です。"
            "以下の申請書コンテキストを根拠に、審査観点に沿って評価し、次の JSON 形式で返答してください。\n\n"
            "【審査観点（例）】\n"
            "- 技術面: 新規性/独自性（0-10）、優位性の根拠（0-10）、実現性（0-10）\n"
            "- 事業面: 市場性/顧客提供価値（0-10）、売上・付加価値の数値計画（0-10）\n"
            "- 体制面: 実施体制・スケジュール・リスク管理（0-10）\n"
            "- 政策適合: 政策目的との整合性、地域/雇用/賃上げ等（0-10）\n"
            "合計スコアは0-100点で、60=標準、70=採択ボーダー、80+=高評価の目安。\n\n"
            "【出力JSONフォーマット】\n"
            "{\n"
            '  "score": <0-100の整数>,\n'
            '  "summary": "全体総評（200-400字）",\n'
            '  "strengths": ["強み1", "強み2"],\n'
            '  "weaknesses": ["弱み1", "弱み2"],\n'
            '  "risks": ["リスク1", "リスク2"],\n'
            '  "missing_items": ["不足資料/不記載の可能性", "..."],\n'
            '  "recommendations": ["改善提案1", "改善提案2"]\n'
            "}\n"
            "根拠は要点を短く示し、推測は避け、コンテキストに無い事項は『不明』と記載してください。"
        ),
        "typo_prompt": (
            "以下の申請書コンテキストを校正してください。誤字脱字、表記ゆれ、単位や数値の不整合、"
            "社名・商品名・人名の不一致、日付/年度の矛盾、表/本文の齟齬、ページまたぎでの用語ゆれなどを検出し、"
            "次の JSON 形式で返答してください。\n\n"
            "{\n"
            '  "issues": [\n'
            '    {\n'
            '      "type": "誤字/表記ゆれ/数値矛盾 など",\n'
            '      "excerpt": "問題箇所の短い抜粋",\n'
            '      "detail": "何が問題か（できる限り具体的に）",\n'
            '      "suggestion": "どう直すべきかの提案"\n'
            "    }\n"
            "  ],\n"
            '  "summary": "全体所感（100-200字）"\n'
            "}\n"
            "コンテキストに無い情報での断定はせず、『不明』と明記してください。"
        )
    }


def load_prompts() -> dict:
    try:
        with open(prompts_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_prompts()


def save_prompts(prompts: dict):
    with open(prompts_path(), "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)


# -----------------------------
# 履歴 CSV
# -----------------------------
def append_history(timestamp: str, company_name: str, score, mode: str, filename: str):
    """履歴CSVに1行追記（存在しなければヘッダ付きで作成）"""
    p = history_path()
    row = {
        "timestamp": timestamp,
        "company_name": company_name,
        "score": score,
        "mode": mode,
        "filename": filename
    }
    try:
        df = pd.read_csv(p)
    except Exception:
        df = pd.DataFrame(columns=["timestamp", "company_name", "score", "mode", "filename"])
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(p, index=False)


def load_history() -> pd.DataFrame:
    try:
        return pd.read_csv(history_path())
    except Exception:
        return pd.DataFrame(columns=["timestamp", "company_name", "score", "mode", "filename"])


# -----------------------------
# PDF → テキスト抽出
# -----------------------------
def extract_text_from_pdf(file_like: BytesIO) -> str:
    """pypdf でシンプルにテキスト抽出（埋め込みテキストが無い場合は空文字）"""
    reader = PdfReader(file_like)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        texts.append(t)
    return "\n".join(texts)


def guess_company_name_from_text(text: str) -> str | None:
    """簡易ルールで社名を推定（『株式会社〇〇』など）"""
    # よくある法人表記パターンをいくつか試す
    patterns = [
        r"(?:応募者|申請者)[:：]\s*([^\n]{2,30})",
        r"(株式会社[^\s\n]{1,30})",
        r"([^\s\n]{1,30}株式会社)",
        r"(合同会社[^\s\n]{1,30})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            name = m.group(1).strip()
            # ノイズ除去（句読点など）
            name = re.sub(r"[。．,.、\s]+$", "", name)
            if 2 <= len(name) <= 40:
                return name
    return None


# -----------------------------
# チャンク化（日本語向け）
# -----------------------------
def chunk_text_for_japanese(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    日本語文の区切り（。！？\n）を活用して、指定長でチャンク化。
    overlap でチャンク間の重なりを持たせ、前後関係を多少維持する。
    """
    # 区切りで一次分割
    # 句点・改行などを区切りとする
    sentences = re.split(r"(。|！|？|\n)", text)
    # sentences は ["文", "。", "文", "。", ...] の形になるので結合
    units = []
    buf = ""
    for s in sentences:
        if s in ["。", "！", "？", "\n"]:
            buf += s
            units.append(buf)
            buf = ""
        else:
            buf += s
    if buf.strip():
        units.append(buf)

    # 指定長でまとめる
    chunks = []
    cur = ""
    for u in units:
        if len(cur) + len(u) <= max_chars:
            cur += u
        else:
            if cur.strip():
                chunks.append(cur.strip())
            # overlap 分だけ末尾から残す
            if overlap > 0 and len(cur) > overlap:
                cur = cur[-overlap:] + u
            else:
                cur = u
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# -----------------------------
# TF-IDF 検索（ローカルRAG簡易版）
# -----------------------------
def build_tfidf_index(chunks: List[str]) -> Tuple[TfidfVectorizer, any]:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        max_features=50000
    )
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def retrieve_top_k(query: str, vectorizer: TfidfVectorizer, matrix, chunks: List[str], k: int = 6) -> Tuple[List[int], List[str]]:
    if not query.strip():
        return list(range(min(k, len(chunks)))), chunks[:k]
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, matrix)[0]
    idx = sims.argsort()[::-1][:k]
    return idx.tolist(), [chunks[i] for i in idx]


# -----------------------------
# OpenAI 呼び出し（Chat Completions）
# -----------------------------
def call_openai_with_context(system_prompt: str, user_task_prompt: str, context_chunks: List[str], model_override: str | None = None) -> str | None:
    """
    Context（Top-K 抜粋）を添えて ChatCompletion を実行。
    APIキー未設定やエラー時は None を返す。
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        # Streamlit 側でユーザーに知らせる（raise はしない）
        import streamlit as st
        st.error("OPENAI_API_KEY が未設定です。環境変数に API キーを設定してください。")
        return None

    if OpenAI is None:
        import streamlit as st
        st.error("`openai` パッケージの読み込みに失敗しました。`pip install openai` を実行してください。")
        return None

    model = model_override or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # コンテキストの整形（識別可能な区切り付き）
    context_text = "\n\n---\n\n".join([f"[CONTEXT #{i+1}]\n{c}" for i, c in enumerate(context_chunks)])

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "次のコンテキスト（申請書の抜粋）に基づいてタスクを実施してください。\n"
                        "【重要】コンテキスト外の情報で断定せず、不明点は『不明』と記載。\n\n"
                        f"{context_text}\n\n"
                        "----\n"
                        f"【タスク】\n{user_task_prompt}"
                    )
                }
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        import streamlit as st
        st.error(f"OpenAI 呼び出しでエラー: {e}")
        return None


# -----------------------------
# スコア抽出の安全化
# -----------------------------
def parse_score_safely(obj_or_text) -> int | None:
    """
    JSON / テキストから 0-100 の整数スコアを可能な限り抽出。
    - JSONで "score" キーが整数ならそれを返す
    - テキストなら 'score": 85' や '85点' などを探索
    """
    # JSONの場合
    if isinstance(obj_or_text, dict):
        v = obj_or_text.get("score")
        try:
            iv = int(v)
            if 0 <= iv <= 100:
                return iv
        except Exception:
            pass

    # テキストの場合
    text = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text, ensure_ascii=False)
    # 例: "score": 85
    m = re.search(r'"score"\s*:\s*(\d{1,3})', text)
    if m:
        iv = int(m.group(1))
        if 0 <= iv <= 100:
            return iv
    # 例: 85点
    m2 = re.search(r'(\d{1,3})\s*点', text)
    if m2:
        iv = int(m2.group(1))
        if 0 <= iv <= 100:
            return iv
    return None
