"""Streamlit Interactive Web UI for Fake News Classification & Explainability.

Provides real-time article analysis, probability gauges, explainability token
highlighting, stylometric signal breakdown, and EDA exploration.
"""

import json
import logging
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, REPORTS_DIR, MODELS_DIR
from src.inference.pipeline import get_inference_engine
from src.eda.analyzer import NewsEDAAnalyzer

# Page configuration
st.set_page_config(
    page_title="Fake News Detection & Explainability Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
        margin-bottom: 12px;
    }
    .fake-badge {
        background-color: #ffebee;
        color: #c62828;
        font-weight: 700;
        font-size: 1.4rem;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid #ef9a9a;
        display: inline-block;
    }
    .real-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        font-weight: 700;
        font-size: 1.4rem;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid #a5d6a7;
        display: inline-block;
    }
    .highlight-fake {
        background-color: #ffcdd2;
        color: #b71c1c;
        padding: 2px 5px;
        border-radius: 4px;
        font-weight: 600;
    }
    .highlight-real {
        background-color: #c8e6c9;
        color: #1b5e20;
        padding: 2px 5px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Caches the loaded inference engine."""
    return get_inference_engine()


def render_highlighted_text(text: str, fake_words: list, real_words: list) -> str:
    """Highlights influential words in HTML."""
    fake_set = {w.lower(): score for w, score in fake_words}
    real_set = {w.lower(): score for w, score in real_words}
    
    tokens = text.split()
    html_tokens = []
    
    for t in tokens:
        clean_t = "".join(c for c in t if c.isalnum()).lower()
        if clean_t in fake_set:
            score = fake_set[clean_t]
            html_tokens.append(f"<span class='highlight-fake' title='Misinformation Driver (+{score})'>{t}</span>")
        elif clean_t in real_set:
            score = real_set[clean_t]
            html_tokens.append(f"<span class='highlight-real' title='Credibility Indicator (+{score})'>{t}</span>")
        else:
            html_tokens.append(t)
            
    return " ".join(html_tokens)


# Sample article presets
PRESETS = {
    "Select a preset...": {
        "title": "",
        "author": "",
        "text": ""
    },
    "🚨 Sensational Conspiracy (Fake News)": {
        "title": "BOMBSHELL: Proof of Secret Weather Weapons Creating Artificial Disasters",
        "author": "UndergroundPatriot",
        "text": "Military whistleblowers expose leaked documents proving secret scalar transmitters trigger synthetic hurricanes to terrorize the populace. Mainstream media continues covering up the massive conspiracy."
    },
    "🏛️ Institutional Science & Space (Real News)": {
        "title": "NASA James Webb Space Telescope Observes Atmospheric Composition of Distant Exoplanet",
        "author": "Dr. Sarah Jenkins",
        "text": "Astronomers utilizing the James Webb Space Telescope have detected definitive spectral signatures of water vapor and carbon dioxide in the upper atmosphere of a gas giant exoplanet located 700 light-years away in peer-reviewed findings."
    },
    "⚖️ Economic & Policy Announcement (Real News)": {
        "title": "Federal Reserve Holds Benchmark Interest Rate Steady Amid Moderate Growth",
        "author": "Reuters Economics Desk",
        "text": "The Federal Reserve announced Wednesday that it will maintain its benchmark federal funds rate at current levels following a two-day policy meeting. Central bank officials pointed to continued labor market resilience and moderate economic expansion."
    },
    "⚠️ Medical Miracle Hoax (Fake News)": {
        "title": "SHOCKING: Secret Government Vault Discovered Containing Banned 100% Cancer Cure!",
        "author": "PatriotTruthNetwork",
        "text": "Whistleblowers have finally broken their silence regarding a massive underground vault containing natural miracle frequencies that eliminate all diseases overnight! Mainstream pharmaceutical cartels and corrupt elites have colluded for decades to hide this miraculous truth."
    }
}


def main():
    st.title("🛡️ Production Fake News Detection & Explainability Engine")
    st.caption("End-to-End NLP Pipeline with Stylometry, Attribution Heatmaps, and False-Negative Mitigation")

    try:
        engine = load_engine()
    except Exception as e:
        st.error(f"Failed to load trained model artifact: {e}. Please ensure `scripts/train_and_evaluate.py` has run.")
        return

    # Tabs
    tab_inspect, tab_eda, tab_about = st.tabs([
        "🔍 News Verification & Explainability",
        "📊 Exploratory Data Analysis (EDA)",
        "⚙️ Model & Pipeline Specs"
    ])

    with tab_inspect:
        col1, col2 = st.columns([1.2, 1.0])

        with col1:
            st.subheader("Input News Article")
            preset_choice = st.selectbox("Load Example Article Preset:", list(PRESETS.keys()))

            default_title = PRESETS[preset_choice]["title"]
            default_author = PRESETS[preset_choice]["author"]
            default_text = PRESETS[preset_choice]["text"]

            input_title = st.text_input("Headline / Title:", value=default_title, placeholder="Enter article headline...")
            input_author = st.text_input("Author / Source Byline:", value=default_author, placeholder="e.g. Associated Press, Unknown, BlogName")
            input_text = st.text_area("Article Body Content:", value=default_text, height=220, placeholder="Paste full article text here...")

            analyze_button = st.button("🚀 Analyze Article", type="primary", use_container_width=True)

        with col2:
            st.subheader("Verification Analysis")
            if analyze_button or (input_title and input_text):
                if not input_title.strip() and not input_text.strip():
                    st.warning("Please enter at least a title or article body.")
                else:
                    with st.spinner("Executing NLP normalization and explainability attribution..."):
                        result = engine.predict_news(
                            title=input_title,
                            text=input_text,
                            author=input_author or "Unknown",
                            explain=True
                        )

                    is_fake = result["label"] == "fake"
                    confidence_pct = result["confidence"] * 100
                    fake_prob_pct = result["fake_probability"] * 100
                    real_prob_pct = result["real_probability"] * 100

                    # Result Badge
                    if is_fake:
                        st.markdown(
                            f"<div class='fake-badge'>⚠️ CLASSIFIED AS FAKE NEWS ({confidence_pct:.1f}% Confidence)</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='real-badge'>✅ CLASSIFIED AS REAL NEWS ({confidence_pct:.1f}% Confidence)</div>",
                            unsafe_allow_html=True
                        )

                    st.write("")
                    
                    # Probabilities
                    st.markdown(f"**Fake News Probability:** `{fake_prob_pct:.1f}%`")
                    st.progress(result["fake_probability"])

                    st.markdown(f"**Real News Probability:** `{real_prob_pct:.1f}%`")
                    st.progress(result["real_probability"])

                    # Decision threshold note
                    st.caption(f"🎯 Calibrated Decision Threshold: `{result['decision_threshold']:.2f}` (Optimized for $F_2$ to minimize False Negatives)")

                    # Rationale Box
                    st.info(f"**Automated Verification Rationale:**\n\n{result['rationale']}")

                    # Top driving features
                    st.markdown("#### Top Contributing Lexical Features")
                    keywords = result["top_contributing_keywords"]
                    if keywords:
                        kw_df = pd.DataFrame(keywords, columns=["Token / Feature", "Attribution Weight"])
                        st.dataframe(kw_df, use_container_width=True, hide_index=True)
                    else:
                        st.write("Neutral aggregate profile.")

        # Full Document Token Highlighting
        if (analyze_button or (input_title and input_text)) and input_text:
            st.divider()
            st.subheader("📝 Explainability Token Attribution Heatmap")
            st.caption("Visualizing tokens that drove classification: Red = Fake indicators, Green = Credibility indicators.")
            
            fake_kws = result["top_contributing_keywords"] if is_fake else []
            real_kws = result["top_contributing_keywords"] if not is_fake else []

            highlighted_html = render_highlighted_text(input_text, fake_kws, real_kws)
            st.markdown(
                f"<div style='background-color:#ffffff; padding:18px; border-radius:8px; border:1px solid #ddd; line-height:1.7;'>{highlighted_html}</div>",
                unsafe_allow_html=True
            )

    with tab_eda:
        st.subheader("Exploratory Data Analysis & Linguistic Patterns")
        eda_file = REPORTS_DIR / "eda_report.json"
        if eda_file.exists():
            with open(eda_file, "r", encoding="utf-8") as f:
                eda_data = json.load(f)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Benchmark Samples", eda_data["total_samples"])
            c2.metric("Real News Proportion", f"{eda_data['class_percentages']['Real (%)']}%")
            c3.metric("Fake News Proportion", f"{eda_data['class_percentages']['Fake (%)']}%")
            c4.metric("Lexical Diversity (TTR)", f"{eda_data['lexical_diversity']['Fake News (TTR)']:.3f} (Fake)")

            st.write("---")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("##### 📌 Stylometric & Affect Markers")
                sens_df = pd.DataFrame(eda_data["sensationalism_signals"]).T
                st.dataframe(sens_df, use_container_width=True)

            with col_b:
                st.markdown("##### 📖 Readability & Token Length")
                read_df = pd.DataFrame(eda_data["readability_scores"]).T
                st.dataframe(read_df, use_container_width=True)

            st.write("---")
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                st.markdown("##### Top Fake News Discriminative Unigrams")
                fake_unigrams = pd.DataFrame(eda_data["top_fake_unigrams"], columns=["Unigram", "Count"])
                st.dataframe(fake_unigrams, use_container_width=True, hide_index=True)

            with col_u2:
                st.markdown("##### Top Real News Discriminative Unigrams")
                real_unigrams = pd.DataFrame(eda_data["top_real_unigrams"], columns=["Unigram", "Count"])
                st.dataframe(real_unigrams, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python scripts/run_eda.py` to generate the EDA report.")

    with tab_about:
        st.subheader("System Architecture & Specifications")
        st.markdown("""
        - **Pipeline Workflow**: Text Normalization $\\rightarrow$ Contraction Expansion $\\rightarrow$ Morphological Lemmatization $\\rightarrow$ Metadata Synthesis (`[AUTHOR]`, `[TITLE]`, `[CONTENT]`) $\\rightarrow$ TF-IDF / Transformer.
        - **False Negative Mitigation**: Optimized decision boundary minimizing undetected fake news ($F_2$ weighting).
        - **Explainability Engine**: Combines linear attribution and local perturbation surrogates (LIME).
        - **Hardware Acceleration**: Ready for NVIDIA RTX GPUs with mixed precision (`torch.cuda.amp`) and gradient accumulation.
        """)


if __name__ == "__main__":
    main()
