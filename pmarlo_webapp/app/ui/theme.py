from __future__ import annotations

import streamlit as st


_SHARD_TOKEN_STYLES = """
<style>
:root {
    --pmarlo-shard-metabiased-bg: #f472b6;
    --pmarlo-shard-metabiased-fg: #4a0433;
    --pmarlo-shard-unbiased-bg: #16a34a;
    --pmarlo-shard-unbiased-fg: #022313;
}

/* Selected tags in multiselect widgets */
[data-testid="stMultiSelect"] div[data-baseweb="tag"][aria-label*="[CV-BIASED]"] {
    background-color: var(--pmarlo-shard-metabiased-bg) !important;
    border-color: var(--pmarlo-shard-metabiased-bg) !important;
    color: var(--pmarlo-shard-metabiased-fg) !important;
}

[data-testid="stMultiSelect"] div[data-baseweb="tag"][aria-label*="[UNBIASED]"] {
    background-color: var(--pmarlo-shard-unbiased-bg) !important;
    border-color: var(--pmarlo-shard-unbiased-bg) !important;
    color: var(--pmarlo-shard-unbiased-fg) !important;
}

[data-testid="stMultiSelect"] div[data-baseweb="tag"][aria-label*="[CV-BIASED]"] svg,
[data-testid="stMultiSelect"] div[data-baseweb="tag"][aria-label*="[UNBIASED]"] svg {
    color: inherit !important;
    fill: currentColor !important;
}

/* Options rendered inside the dropdown menu */
[data-baseweb="menu"] [role="option"][aria-label*="[CV-BIASED]"]::before,
[data-baseweb="menu"] [role="option"][aria-label*="[UNBIASED]"]::before {
    content: "";
    display: inline-block;
    width: 0.6rem;
    height: 0.6rem;
    border-radius: 999px;
    margin-right: 0.5rem;
}

[data-baseweb="menu"] [role="option"][aria-label*="[CV-BIASED]"]::before {
    background-color: var(--pmarlo-shard-metabiased-bg);
}

[data-baseweb="menu"] [role="option"][aria-label*="[UNBIASED]"]::before {
    background-color: var(--pmarlo-shard-unbiased-bg);
}
</style>
"""


def inject_global_styles() -> None:
    """Inject shared Streamlit style overrides once per session."""

    flag = "_pmarlo_styles_injected"
    if st.session_state.get(flag):
        return
    st.markdown(_SHARD_TOKEN_STYLES, unsafe_allow_html=True)
    st.session_state[flag] = True
