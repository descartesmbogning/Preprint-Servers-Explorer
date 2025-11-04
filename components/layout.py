# components/layout.py
from __future__ import annotations
import os
import streamlit as st

# ---------- CSS loader ----------
def load_css(path: str = "assets/brand.css", extra: list[str] | None = None) -> None:
    """Inject CSS file(s) into the Streamlit app."""
    paths = [path] + (extra or [])
    blocks: list[str] = []
    for p in paths:
        if not p:
            continue
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                blocks.append(f.read())
        else:
            st.warning(f"CSS file not found: {p}")

    if blocks:
        css = "\n".join(blocks)   # ✅ fix: compute first (Windows-safe)
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ---------- Internal helpers ----------
def _nav_link(label: str, key: str, active_key: str) -> str:
    """Build a nav link that changes `?section=…` in-place (same tab)."""
    is_active = key == active_key
    cls = "active" if is_active else ""
    aria = ' aria-current="page"' if is_active else ""
    return (
        f'<a class="{cls}" href="?section={key}" target="_self"{aria}>'
        f'{label}</a>'
    )

def _img_if(path: str, alt: str, cls: str = "") -> str:
    if os.path.exists(path):
        klass = f' class="{cls}"' if cls else ""
        return f'<img src="{path}" alt="{alt}"{klass}>'
    return ""

# ---------- Header ----------
def render_sticky_header(active_section: str = "overview") -> None:
    """ScholCommLab-styled fixed header."""
    logo_tag = _img_if("assets/scholcommlab-logo.png", "ScholCommLab logo")
    if not logo_tag:
        logo_tag = '<span style="font-weight:700;color:var(--scl-blue)">ScholCommLab</span>'

    nav_html = " ".join([
        _nav_link("Overview", "overview", active_section),
        _nav_link("Source Explorer", "explorer", active_section),
        _nav_link("Compare", "compare", active_section),
        _nav_link("Data", "data", active_section),
        _nav_link("About", "about", active_section),
    ])

    st.markdown(
        f"""
        <header class="app-header" role="banner">
          <div class="row">
            <div class="app-brand">
              {logo_tag}
              <div class="title">Preprints Tracker</div>
            </div>
            <nav class="nav" role="navigation" aria-label="Primary">
              {nav_html}
            </nav>
            <div class="right-links" role="navigation" aria-label="External">
              <a href="https://www.scholcommlab.ca/" rel="noopener noreferrer">scholcommlab.ca ↗</a>
            </div>
          </div>
        </header>
        """,
        unsafe_allow_html=True
    )

# ---------- Footer ----------
def render_sticky_footer(last_dt=None) -> None:
    """Fixed footer with brand links."""
    last_html = f"Last updated: <b>{last_dt.date()}</b>" if last_dt else "Last updated: —"
    sfu = "assets/marks/sfu.png"        # optional partner marks
    uo  = "assets/marks/uottawa.png"

    logos = ""
    for p, alt in [(sfu, "SFU")]: #, (uo, "uOttawa")]:
        if os.path.exists(p):
            logos += f'<img src="{p}" alt="{alt}" class="partner-mark">'

    st.markdown(
        f"""
        <footer class="app-footer" role="contentinfo">
          <div class="row">
            <div class="left">
              <span>© ScholCommLab</span>
              <span class="partner-marks">{logos}</span>
            </div>
            <div class="links">
              <span class="muted">{last_html}</span>
              <a href="https://www.scholcommlab.ca/" target="_blank" rel="noopener">Website</a>
              <a href="https://twitter.com/scholcommlab" target="_blank" rel="noopener">X/Twitter</a>
              <a href="?section=about">About this app</a>
              <a class="cta" href="https://www.scholcommlab.ca/join-us/" target="_blank" rel="noopener">Join us ↗</a>
            </div>
          </div>
        </footer>
        """,
        unsafe_allow_html=True
    )