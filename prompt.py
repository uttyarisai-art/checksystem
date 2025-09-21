# -*- coding: utf-8 -*-
# 審査項目チェック／誤字脱字チェックのプロンプトを編集・保存するページ

import streamlit as st
from common import load_prompts, save_prompts, default_prompts, ensure_data_dirs

def render():
    ensure_data_dirs()
    st.title("🧩 プロンプト編集")

    st.caption("ここで編集したプロンプトが、[✅ チェック] 実行時に使用されます。")

    prompts = load_prompts()
    criteria_prompt = st.text_area(
        "審査項目チェック用プロンプト",
        value=prompts["criteria_prompt"],
        height=300,
        help="審査観点・配点・出力形式（JSON推奨）などを記述"
    )
    typo_prompt = st.text_area(
        "誤字脱字・整合性チェック用プロンプト",
        value=prompts["typo_prompt"],
        height=300,
        help="検出したい表記ゆれ・数値矛盾・固有名詞の不一致などを記述"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存", type="primary", use_container_width=True):
            save_prompts({"criteria_prompt": criteria_prompt, "typo_prompt": typo_prompt})
            st.success("プロンプトを保存しました。")
    with col2:
        if st.button("↩️ 既定値に戻す", use_container_width=True):
            defaults = default_prompts()
            save_prompts(defaults)
            st.info("既定のプロンプトに戻しました。ページを再読み込みすると反映されます。")

    with st.expander("ℹ️ プロンプト作成のコツ", expanded=False):
        st.markdown(
            "- **出力形式は JSON** を強く推奨。キー例: `score`, `summary`, `strengths`, `weaknesses`, `risks`, `missing_items` 等\n"
            "- **根拠の明示**: 回答内に、どの抜粋（コンテキスト）を根拠にしたかを短く示させる\n"
            "- **採点基準の定義**: 例：60点＝標準、70点＝採択ボーダー、80点以上＝高評価… 等\n"
        )
