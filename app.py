import streamlit as st
from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer
import time

# --------- Simple Minimalist Custom CSS ---------
st.markdown("""
    <style>
    body { background: #fafbfc !important; }
    .main-title {
        color: #22223b;
        font-size: 2.1em;
        font-weight: 600;
        letter-spacing: -1px;
        margin-bottom: 0.2em;
        font-family: 'Segoe UI', sans-serif;
    }
    .subtitle {
        color: #555;
        font-size: 1.04em;
        margin-bottom: 1.3em;
        margin-top: -14px;
    }
    .token-box {
        font-family: 'Fira Mono', monospace;
        font-size: 1em;
        background: #f4f4f8;
        border-radius: 8px;
        padding: 0.7em 1em;
        margin-bottom: 0.5em;
        color: #333;
        border: 1px solid #ececec;
        overflow-x: auto;
    }
    .metric-min {
        display: inline-block;
        background: #f7f7fa;
        border-radius: 6px;
        padding: 0.5em 1.2em;
        margin-right: 1em;
        margin-bottom: 0.7em;
        font-size: 1.06em;
        color: #374151;
        border: 1px solid #efefef;
    }
    .footer-min {
        color: #aaa;
        font-size: 0.95em;
        margin-top: 2em;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# --------- Main Title ---------
st.markdown('<div class="main-title">Tokenization Demo for LLMs</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Compare three tokenization methods. Paste your text, pick a tokenizer, and see the difference.</div>', unsafe_allow_html=True)

# --------- Tokenizer Selection ---------
col1, col2 = st.columns([1,3])
with col1:
    tokenizer_type = st.radio(
        "Tokenizer",
        ["Character", "Regex", "GPT4Pattern"],
        index=1,
        format_func=lambda x: {"Character": "Caractère", "Regex": "Regex", "GPT4Pattern": "GPT-4 Pattern"}[x]
    )

tokenizer_cls = {
    "Caractère": BasicTokenizer,
    "Regex": RegexTokenizer,
    "GPT-4 Pattern": GPT4Tokenizer,
    "Character": BasicTokenizer,
    "GPT4Pattern": GPT4Tokenizer
}[tokenizer_type]

# --------- Input Area ---------
default_text = (
    "Le traitement du langage naturel est essentiel pour les modèles de langage modernes. "
    "Essayez ce texte pour observer les différences de tokenisation."
)
with col2:
    text = st.text_area("Texte à tokeniser", default_text, height=100)

# --------- Run Tokenizer ---------
tokenizer = tokenizer_cls()
start = time.time()
tokens = tokenizer.encode(text)
encode_time = (time.time() - start) * 1000
detok = tokenizer.decode(tokens)

# --------- Show Metrics ---------
st.markdown(
    f"""
    <div class="metric-min"><b>Tokens</b>: {len(tokens)}</div>
    <div class="metric-min"><b>Encodage</b>: {encode_time:.1f} ms</div>
    <div class="metric-min"><b>Caractères</b>: {len(text)}</div>
    """, unsafe_allow_html=True
)

# --------- Show Tokens ---------
st.markdown("<b>Tokens générés :</b>", unsafe_allow_html=True)
st.markdown(f"<div class='token-box'>{tokens}</div>", unsafe_allow_html=True)

# --------- Show Detokenized ---------
st.markdown("<b>Texte après décodage :</b>", unsafe_allow_html=True)
st.code(detok, language="markdown")

# --------- Footer ---------
st.markdown(
    "<div class='footer-min'>LLM Tokenizer Demo &mdash; Elkadiri Anas</div>",
    unsafe_allow_html=True
)
