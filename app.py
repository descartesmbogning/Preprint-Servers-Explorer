import io
import os
from urllib.parse import quote, unquote

import pandas as pd
import plotly.express as px
import streamlit as st

from components.layout import load_css, render_sticky_header, render_sticky_footer

st.set_page_config(
    page_title="Preprints Tracker — ScholCommLab",
    page_icon="assets/scholcommlab-favicon.png",
    layout="wide",
)

load_css(extra=["assets/layout.css"])

DATA_DIR = "data"
GOOGLE_SHEET_META_URL = "https://docs.google.com/spreadsheets/d/1qrWIC8II0HEzETcp1nFpJEVEKxYS0CkE/export?format=csv&gid=261027131"
GOOGLE_SHEET_TTL_SECONDS = 300

qp = st.query_params
section_key = qp.get("section", "overview")
render_sticky_header(active_section=section_key)

FILTER_CONFIG = {
    "is_active_2026": {
        "label": "Active in 2026",
        "type": "boolean_ynu",
        "group": "quick",
        "widget": "checkbox_yes_only",
    },

    "domain_tags": {"label": "Domain", "type": "multivalue", "group": "quick"},
    "region_scope_type": {"label": "Region scope", "type": "multivalue", "group": "quick"},
    "region_label": {"label": "Region", "type": "multivalue", "group": "quick"},
    "ownership_group": {"label": "Ownership", "type": "multivalue", "group": "quick"},
    "acceptance_group": {"label": "Acceptance", "type": "multivalue", "group": "quick"},
    "moderation_group": {"label": "Moderation", "type": "multivalue", "group": "quick"},
    "language_group": {"label": "Language", "type": "multivalue", "group": "quick"},
    "fee_model": {"label": "Fees", "type": "multivalue", "group": "quick"},
    "source_role": {"label": "Source role", "type": "multivalue", "group": "quick"},

    "submission_term_group": {"label": "Submission terminology", "type": "multivalue", "group": "advanced"},
    "journal_integration_group": {"label": "Journal integration", "type": "multivalue", "group": "advanced"},
    "versioning_group": {"label": "Versioning", "type": "multivalue", "group": "advanced"},
    "indexing_group": {"label": "Indexing", "type": "multivalue", "group": "advanced"},
    "preservation_group": {"label": "Preservation", "type": "multivalue", "group": "advanced"},
    "peer_review_group": {"label": "Peer review", "type": "multivalue", "group": "advanced"},
}

PROFILE_CONFIG = {
    "Identity": ["server_url", "is_active_2026", "region_scope_type", "region_label", "source_role"],
    "Scope & governance": ["domain_tags", "domain_primary", "ownership_group", "moderation_group", "language_group", "fee_model"],
    "Submission & content": ["acceptance_group", "submission_term_group", "versioning_group"],
    "Trust & review": ["peer_review_group"],
    "Ecosystem & visibility": ["journal_integration_group", "indexing_group", "preservation_group", "preservation_partner"],
}

COMPARE_PROFILE_CONFIG = {
    "Identity": ["is_active_2026", "region_scope_type", "region_label", "source_role"],
    "Scope & governance": ["domain_tags", "ownership_group", "moderation_group", "language_group", "fee_model"],
    "Submission & content": ["acceptance_group", "submission_term_group", "versioning_group"],
    "Trust & review": ["peer_review_group"],
    "Ecosystem & visibility": ["journal_integration_group", "indexing_group", "preservation_group"],
}


def _find_file(basename_no_ext: str):
    csv_path = os.path.join(DATA_DIR, f"{basename_no_ext}.csv")
    xlsx_path = os.path.join(DATA_DIR, f"{basename_no_ext}.xlsx")
    if os.path.exists(csv_path):
        return csv_path
    if os.path.exists(xlsx_path):
        return xlsx_path
    return None


def _fmt_count_pct(n, denom):
    try:
        n = int(n)
    except Exception:
        return "—"
    if denom and denom > 0:
        return f"{n:,} ({n/denom:.1%})"
    return f"{n:,}"


def _int_or_zero(x):
    try:
        return int(x)
    except Exception:
        return 0


def _safe_int_series_sum(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns or df.empty:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def normalize_multivalue_cell(x: object) -> list[str]:
    if pd.isna(x):
        return []

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []

    parts = [p.strip() for p in s.split(";")]
    parts = [p for p in parts if p not in ("", "nan", "None")]

    seen = set()
    clean = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            clean.append(p)

    return clean


def format_profile_value(val):
    if pd.isna(val):
        return None
    sval = str(val).strip()
    if sval == "" or sval.lower() == "nan":
        return None
    return sval


def prettify_label(col_name: str) -> str:
    return col_name.replace("_", " ").strip().title()


def _qp_bool(key: str, default: bool) -> bool:
    v = qp.get(key)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def _encode_list(vals):
    return "|".join(quote(str(v), safe="") for v in vals)


def _decode_list(s):
    if not s:
        return []
    return [unquote(x) for x in str(s).split("|") if x != ""]


def _update_qp_if_changed(**kvs):
    new_map = dict(qp)
    changed = False
    for k, v in kvs.items():
        sval = "" if v is None else str(v)
        if new_map.get(k, "") != sval:
            new_map[k] = sval
            changed = True
    if changed:
        try:
            st.query_params.update(new_map)
        except Exception:
            pass


def reset_all_filters(current_section: str):
    try:
        st.query_params.clear()
        st.query_params["section"] = current_section
    except Exception:
        pass


def init_filter_state_from_query(yr_min: int, yr_max: int, yr_from_default: int, yr_to_default: int):
    if "filter_year_range" not in st.session_state:
        st.session_state["filter_year_range"] = (
            max(yr_min, yr_from_default),
            min(yr_max, yr_to_default),
        )

    for col, cfg in FILTER_CONFIG.items():
        key = f"filter_{col}"
        qp_key = f"flt_{col}"

        if key in st.session_state:
            continue

        if cfg.get("widget") == "checkbox_yes_only":
            st.session_state[key] = _qp_bool(qp_key, False)
        else:
            st.session_state[key] = _decode_list(qp.get(qp_key, ""))


@st.cache_data
def read_any(path_or_buffer):
    if isinstance(path_or_buffer, str):
        if path_or_buffer.endswith(".csv"):
            return pd.read_csv(path_or_buffer)
        return pd.read_excel(path_or_buffer)
    name = getattr(path_or_buffer, "name", "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(path_or_buffer)
    return pd.read_excel(path_or_buffer)


@st.cache_data(ttl=GOOGLE_SHEET_TTL_SECONDS)
def load_google_sheet_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


@st.cache_data
def clean_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "server_name" not in df.columns:
        raise ValueError("Summary file must include 'server_name'.")
    df["server_name"] = df["server_name"].astype(str).str.strip()
    for c in ["n_records", "n_is_version_of", "n_unique", "n_published", "count_2024", "count_2025"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "collection_date" in df.columns:
        df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
    return df


@st.cache_data
def clean_yearly(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "server_name" not in df.columns:
        for cand in ["server", "serverName", "Server Name"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "server_name"})
                break
    if "server_name" not in df.columns:
        raise ValueError("Yearly file must have a 'server_name' column.")
    df["server_name"] = df["server_name"].astype(str).str.strip()
    year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    long_df = df.melt(id_vars=["server_name"], value_vars=year_cols, var_name="year", value_name="count")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce").fillna(0).astype(int)
    return long_df.dropna(subset=["year"]).sort_values(["server_name", "year"]).reset_index(drop=True)


@st.cache_data
def clean_yearly_enriched(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = ["server_name", "year", "count_preprints"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Yearly enriched file missing required columns: {missing}")
    df["server_name"] = df["server_name"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for c in ["count_preprints", "count_versioned", "count_published", "count_cross_server"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["year"]).sort_values(["server_name", "year"]).reset_index(drop=True)


@st.cache_data
def clean_metadata(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "server_name" not in df.columns:
        raise ValueError("Metadata file must include 'server_name'.")
    df["server_name"] = df["server_name"].astype(str).str.strip()
    return df.drop_duplicates(subset=["server_name"]).reset_index(drop=True)


@st.cache_data
def build_range_summary(yearly_enriched_rng: pd.DataFrame) -> pd.DataFrame:
    if yearly_enriched_rng.empty:
        return pd.DataFrame(
            columns=["server_name", "count_preprints", "count_versioned", "count_published", "count_cross_server"]
        )
    return (
        yearly_enriched_rng.groupby("server_name", as_index=False)
        .agg(
            count_preprints=("count_preprints", "sum"),
            count_versioned=("count_versioned", "sum"),
            count_published=("count_published", "sum"),
            count_cross_server=("count_cross_server", "sum"),
        )
        .sort_values("count_preprints", ascending=False)
        .reset_index(drop=True)
    )


def get_filter_options(df: pd.DataFrame, col: str, filter_type: str) -> list[str]:
    if col not in df.columns:
        return []

    if filter_type == "multivalue":
        vals = (
            df[col]
            .dropna()
            .apply(normalize_multivalue_cell)
            .explode()
            .dropna()
            .astype(str)
            .str.strip()
        )
        vals = vals[vals != ""]
        return sorted(vals.unique().tolist())

    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    return sorted(vals.unique().tolist())


def apply_dynamic_filter(
    df: pd.DataFrame,
    col: str,
    selected,
    filter_type: str,
    widget_type: str | None = None
) -> pd.DataFrame:
    if col not in df.columns:
        return df

    if widget_type == "checkbox_yes_only":
        if not selected:
            return df
        return df[df[col].astype(str).str.strip() == "Yes"]

    if not selected:
        return df

    if filter_type == "multivalue":
        selected_set = {str(v).strip() for v in selected if str(v).strip()}
        mask = df[col].apply(
            lambda x: len(selected_set.intersection(set(normalize_multivalue_cell(x)))) > 0
        )
        return df[mask]

    return df[df[col].astype(str).isin(selected)]


def render_profile_sections(row: pd.Series, profile_config: dict):
    for section, fields in profile_config.items():
        shown = []
        for field in fields:
            if field in row.index:
                val = format_profile_value(row[field])
                if val is not None:
                    shown.append((field, val))
        if shown:
            st.markdown(f"#### {section}")
            for field, val in shown:
                st.caption(f"**{prettify_label(field)}:** {val}")
            st.markdown("")


def get_server_range_row(range_summary: pd.DataFrame, server_name: str):
    row = range_summary.loc[range_summary["server_name"] == server_name]
    if row.empty:
        return None
    return row.iloc[0]


def get_compare_metric_row(summary_df: pd.DataFrame, range_summary_df: pd.DataFrame, server_name: str) -> dict:
    sum_row = summary_df.loc[summary_df["server_name"] == server_name].head(1)
    rng_row = range_summary_df.loc[range_summary_df["server_name"] == server_name].head(1)

    preprints_all = _int_or_zero(sum_row.iloc[0]["n_unique"]) if not sum_row.empty and "n_unique" in sum_row.columns else 0
    versioned_all_n = _int_or_zero(sum_row.iloc[0]["n_is_version_of"]) if not sum_row.empty and "n_is_version_of" in sum_row.columns else 0
    published_all_n = _int_or_zero(sum_row.iloc[0]["n_published"]) if not sum_row.empty and "n_published" in sum_row.columns else 0

    preprints_range = _int_or_zero(rng_row.iloc[0]["count_preprints"]) if not rng_row.empty else 0
    versioned_range_n = _int_or_zero(rng_row.iloc[0]["count_versioned"]) if not rng_row.empty else 0
    published_range_n = _int_or_zero(rng_row.iloc[0]["count_published"]) if not rng_row.empty else 0

    return {
        "server_name": server_name,
        "preprints_all": preprints_all,
        "versioned_all": _fmt_count_pct(versioned_all_n, preprints_all if preprints_all else None),
        "published_all": _fmt_count_pct(published_all_n, preprints_all if preprints_all else None),
        "preprints_range": preprints_range,
        "versioned_range": _fmt_count_pct(versioned_range_n, preprints_range if preprints_range else None),
        "published_range": _fmt_count_pct(published_range_n, preprints_range if preprints_range else None),
    }


def render_compare_cards(summary_df: pd.DataFrame, range_summary_df: pd.DataFrame, picked_servers: list[str]):
    cols = st.columns(len(picked_servers))
    for i, server in enumerate(picked_servers):
        metrics = get_compare_metric_row(summary_df, range_summary_df, server)
        with cols[i]:
            st.markdown(f"### {server}")
            st.metric("Preprints (all-time)", f"{metrics['preprints_all']:,}")
            st.metric("Versioned (all-time)", metrics["versioned_all"])
            st.metric("Published links (all-time)", metrics["published_all"])
            st.metric("Preprints (range)", f"{metrics['preprints_range']:,}")
            st.metric("Versioned (range)", metrics["versioned_range"])
            st.metric("Published links (range)", metrics["published_range"])


def build_metadata_compare_table(summary_df: pd.DataFrame, picked_servers: list[str], profile_config: dict) -> pd.DataFrame:
    rows = []
    for section, fields in profile_config.items():
        for field in fields:
            row = {"Section": section, "Field": prettify_label(field)}
            has_any = False
            for server in picked_servers:
                server_row = summary_df.loc[summary_df["server_name"] == server].head(1)
                if not server_row.empty and field in server_row.columns:
                    val = format_profile_value(server_row.iloc[0][field])
                else:
                    val = None
                row[server] = val if val is not None else "—"
                if val is not None:
                    has_any = True
            if has_any:
                rows.append(row)
    return pd.DataFrame(rows)


def build_compare_summary_table(summary_df: pd.DataFrame, range_summary_df: pd.DataFrame, picked_servers: list[str]) -> pd.DataFrame:
    rows = []
    for server in picked_servers:
        m = get_compare_metric_row(summary_df, range_summary_df, server)
        rows.append({
            "Server": server,
            "Preprints (all-time)": m["preprints_all"],
            "Versioned (all-time)": m["versioned_all"],
            "Published links (all-time)": m["published_all"],
            "Preprints (range)": m["preprints_range"],
            "Versioned (range)": m["versioned_range"],
            "Published links (range)": m["published_range"],
        })
    return pd.DataFrame(rows)


def read_md(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return f"⚠️ Missing file: {path}"


sum_path = _find_file("summary")
yr_path = _find_file("yearly")
yr_enriched_path = _find_file("yearly_long_enriched")
meta_path = _find_file("server_metadata_clean")

if not sum_path or not yr_path or not yr_enriched_path:
    st.error("Missing files in data/: summary.csv, yearly.csv, yearly_long_enriched.csv")
    st.stop()

summary_raw = read_any(sum_path)
yearly_raw = read_any(yr_path)
yearly_enriched_raw = read_any(yr_enriched_path)

if GOOGLE_SHEET_META_URL:
    try:
        meta_raw = load_google_sheet_csv(GOOGLE_SHEET_META_URL)
    except Exception:
        if meta_path:
            meta_raw = read_any(meta_path)
        else:
            st.error("Google Sheet metadata failed and no local fallback metadata file was found.")
            st.stop()
else:
    if not meta_path:
        st.error("Missing metadata file: data/server_metadata_clean.csv")
        st.stop()
    meta_raw = read_any(meta_path)

summary = clean_summary(summary_raw)
yearly = clean_yearly(yearly_raw)
yearly_enriched = clean_yearly_enriched(yearly_enriched_raw)
metadata = clean_metadata(meta_raw)

summary["server_name"] = summary["server_name"].astype(str).str.strip()
metadata["server_name"] = metadata["server_name"].astype(str).str.strip()
summary = summary.merge(metadata, on="server_name", how="left")

# preferred start year logic
qp_yr_from = qp.get("yr_from")
qp_yr_to = qp.get("yr_to")

yr_min = int(yearly["year"].min()) if len(yearly) else 2000
yr_max = int(yearly["year"].max()) if len(yearly) else 2025
preferred_start_year = 1990

yr_from_default = int(qp_yr_from) if qp_yr_from and str(qp_yr_from).isdigit() else max(yr_min, preferred_start_year)
yr_to_default = int(qp_yr_to) if qp_yr_to and str(qp_yr_to).isdigit() else yr_max

# initialize from query params
init_filter_state_from_query(yr_min, yr_max, yr_from_default, yr_to_default)

# sidebar
sb_title_col, sb_btn_col = st.sidebar.columns([2, 1])

with sb_title_col:
    st.markdown("## Quick filters")

with sb_btn_col:
    st.markdown('<div class="compact-reset-wrap">', unsafe_allow_html=True)
    if st.button("Reset", key="reset_filters_sidebar", use_container_width=True):
        reset_all_filters(section_key)
        st.session_state.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

yr_from, yr_to = st.sidebar.slider(
    "Year range",
    yr_min,
    yr_max,
    st.session_state["filter_year_range"],
    step=1,
    key="filter_year_range",
)

filter_state = {}

for col, cfg in FILTER_CONFIG.items():
    if cfg["group"] != "quick" or col not in summary.columns:
        continue

    label = cfg["label"]
    ftype = cfg["type"]
    widget = cfg.get("widget")
    widget_key = f"filter_{col}"

    if widget == "checkbox_yes_only":
        st.sidebar.checkbox(label, key=widget_key)
    else:
        options = get_filter_options(summary, col, ftype)
        st.sidebar.multiselect(label, options=options, key=widget_key)

    filter_state[col] = st.session_state.get(widget_key)

with st.sidebar.expander("Advanced filters", expanded=False):
    for col, cfg in FILTER_CONFIG.items():
        if cfg["group"] != "advanced" or col not in summary.columns:
            continue

        widget_key = f"filter_{col}"
        options = get_filter_options(summary, col, cfg["type"])
        st.multiselect(cfg["label"], options=options, key=widget_key)

        filter_state[col] = st.session_state.get(widget_key)

# mirror all filters to query params
qp_updates = {
    "section": section_key,
    "yr_from": yr_from,
    "yr_to": yr_to,
}

for col, cfg in FILTER_CONFIG.items():
    qp_key = f"flt_{col}"
    widget = cfg.get("widget")
    val = filter_state.get(col)

    if widget == "checkbox_yes_only":
        qp_updates[qp_key] = "1" if val else ""
    else:
        qp_updates[qp_key] = _encode_list(val or [])

_update_qp_if_changed(**qp_updates)

show = summary.copy()
for col, cfg in FILTER_CONFIG.items():
    show = apply_dynamic_filter(show, col, filter_state.get(col), cfg["type"], cfg.get("widget"))

filtered_servers = sorted(show["server_name"].dropna().astype(str).unique().tolist())

with st.sidebar.expander("Active filters", expanded=False):
    active_labels = []

    if (yr_from, yr_to) != (yr_min, yr_max):
        active_labels.append(f"Year range: {yr_from}–{yr_to}")

    for col, cfg in FILTER_CONFIG.items():
        val = filter_state.get(col)
        if not val:
            continue
        if cfg.get("widget") == "checkbox_yes_only":
            active_labels.append(cfg["label"])
        else:
            active_labels.append(f"{cfg['label']}: {', '.join(val)}")

    if active_labels:
        for item in active_labels:
            st.caption(f"• {item}")
    else:
        st.caption("No filters applied.")

yearly_rng = yearly[(yearly["year"] >= yr_from) & (yearly["year"] <= yr_to)]
yearly_rng = yearly_rng[yearly_rng["server_name"].isin(filtered_servers)]

yearly_enriched_rng = yearly_enriched[(yearly_enriched["year"] >= yr_from) & (yearly_enriched["year"] <= yr_to)]
yearly_enriched_rng = yearly_enriched_rng[yearly_enriched_rng["server_name"].isin(filtered_servers)]

range_summary = build_range_summary(yearly_enriched_rng)
active_servers_in_range = sorted(
    yearly_rng.loc[yearly_rng["count"] > 0, "server_name"].dropna().astype(str).unique().tolist()
)

base_palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set3
color_map = {name: base_palette[i % len(base_palette)] for i, name in enumerate(filtered_servers)}

if "collection_date" in summary.columns and summary["collection_date"].notna().any():
    last_dt = pd.to_datetime(summary["collection_date"], errors="coerce").max()
    if pd.notna(last_dt):
        st.caption(f"Last updated (collection_date): **{last_dt.date()}**")

if section_key == "overview":
    if len(filtered_servers) == 0:
        st.warning("No servers match the current filters.")
        st.stop()

    servers_in_range = int(range_summary["server_name"].nunique()) if not range_summary.empty else 0
    total_preprints_range = _safe_int_series_sum(range_summary, "count_preprints")
    unique_all_time = int(pd.to_numeric(show.get("n_unique", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    servers_all_time = show["server_name"].nunique()

    total_versions_all_time = _safe_int_series_sum(show, "n_is_version_of")
    total_published_all_time = _safe_int_series_sum(show, "n_published")
    range_versioned = _safe_int_series_sum(range_summary, "count_versioned")
    range_published = _safe_int_series_sum(range_summary, "count_published")

    st.markdown("### Overview metrics")

    r1c1, r1c2 = st.columns(2)
    r1c1.metric("Sources (range)", servers_in_range)
    r1c2.metric("Preprints (range)", f"{total_preprints_range:,}")

    r2c1, r2c2 = st.columns(2)
    r2c1.metric("Sources (all-time)", servers_all_time)
    r2c2.metric("Preprints (all-time)", f"{unique_all_time:,}")

    r3c1, r3c2 = st.columns(2)
    r3c1.metric("Versioned (range)", _fmt_count_pct(range_versioned, total_preprints_range if total_preprints_range else None))
    r3c2.metric("Versioned (all-time)", _fmt_count_pct(total_versions_all_time, unique_all_time if unique_all_time else None))

    r4c1, r4c2 = st.columns(2)
    r4c1.metric("Published links (range)", _fmt_count_pct(range_published, total_preprints_range if total_preprints_range else None))
    r4c2.metric("Published links (all-time)", _fmt_count_pct(total_published_all_time, unique_all_time if unique_all_time else None))

    st.markdown("---")
    st.write("**Top sources**")

    ranking = (
        range_summary[["server_name", "count_preprints"]]
        .rename(columns={"count_preprints": "total"})
        .sort_values("total", ascending=False)
    )

    if ranking.empty:
        st.info("No sources to show.")
    else:
        top_df = ranking.head(min(15, len(ranking)))
        fig_bar = px.bar(
            top_df,
            x="total",
            y="server_name",
            orientation="h",
            title=f"Top {len(top_df)} sources ({yr_from}–{yr_to})",
            color="server_name",
            color_discrete_map=color_map,
        )
        fig_bar.update_layout(showlegend=False, height=min(220 + 28 * len(top_df), 900))
        st.plotly_chart(fig_bar, width="stretch")

elif section_key == "explorer":
    if len(filtered_servers) == 0:
        st.warning("No servers match the current filters.")
        st.stop()

    search_term = st.text_input("Search a server", "", key="explorer_search_term")
    explorer_servers = filtered_servers if not search_term else [s for s in filtered_servers if search_term.lower() in s.lower()]

    if not explorer_servers:
        st.info("No sources match the current search.")
        st.stop()

    sel = st.selectbox("Choose a source", explorer_servers, key="explorer_source_select")

    row_df = summary.loc[summary["server_name"] == sel].head(1)
    row = row_df.iloc[0] if not row_df.empty else pd.Series(dtype=object)

    sv_all = yearly[yearly["server_name"] == sel].sort_values("year")
    sv_nz = sv_all[sv_all["count"] > 0]
    sv_range_row = get_server_range_row(range_summary, sel)

    preprints_all = _int_or_zero(row.get("n_unique")) if not row_df.empty else 0
    versioned_all_n = _int_or_zero(row.get("n_is_version_of")) if not row_df.empty else 0
    published_all_n = _int_or_zero(row.get("n_published")) if not row_df.empty else 0

    preprints_range = int(sv_range_row["count_preprints"]) if sv_range_row is not None else 0
    versioned_range_n = int(sv_range_row["count_versioned"]) if sv_range_row is not None else 0
    published_range_n = int(sv_range_row["count_published"]) if sv_range_row is not None else 0

    st.markdown(f"## {sel}")
    st.caption(f"Selected range: {yr_from}–{yr_to}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Preprints (all-time)", f"{preprints_all:,}")
    c2.metric("Versioned (all-time)", _fmt_count_pct(versioned_all_n, preprints_all if preprints_all else None))
    c3.metric("Published links (all-time)", _fmt_count_pct(published_all_n, preprints_all if preprints_all else None))

    c4, c5, c6 = st.columns(3)
    c4.metric("Preprints (range)", f"{preprints_range:,}")
    c5.metric("Versioned (range)", _fmt_count_pct(versioned_range_n, preprints_range if preprints_range else None))
    c6.metric("Published links (range)", _fmt_count_pct(published_range_n, preprints_range if preprints_range else None))

    st.markdown("---")
    left, right = st.columns([1.05, 1.35], gap="large")

    with left:
        render_profile_sections(row, PROFILE_CONFIG)

    with right:
        if sv_nz.empty:
            st.info("No yearly preprints for this server.")
        else:
            fig = px.line(
                sv_nz,
                x="year",
                y="count",
                markers=True,
                title=f"{sel} • yearly preprints",
                color="server_name",
                color_discrete_map=color_map
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

elif section_key == "compare":
    if len(active_servers_in_range) < 2:
        st.info("Need at least two active sources in range.")
        st.stop()

    pick = st.multiselect(
        "Pick 2–4 sources",
        options=active_servers_in_range,
        default=active_servers_in_range[:min(3, len(active_servers_in_range))],
        max_selections=4,
        key="compare_pick_sources",
    )

    if len(pick) < 2:
        st.info("Select at least two sources to compare.")
        st.stop()

    render_compare_cards(summary, range_summary, pick)

    st.markdown("---")

    cmp = yearly_rng[yearly_rng["server_name"].isin(pick)].copy()
    cmp = cmp[cmp["count"] > 0]

    if cmp.empty:
        st.info("No non-zero data for chosen sources.")
    else:
        fig_cmp = px.line(
            cmp,
            x="year",
            y="count",
            color="server_name",
            markers=True,
            title=f"Comparison • {yr_from}–{yr_to}",
            color_discrete_map=color_map,
        )
        st.plotly_chart(fig_cmp, width="stretch")

    st.markdown("### Metric comparison")
    st.dataframe(build_compare_summary_table(summary, range_summary, pick), width="stretch", hide_index=True)

    st.markdown("### Metadata comparison")
    meta_cmp = build_metadata_compare_table(summary, pick, COMPARE_PROFILE_CONFIG)
    if meta_cmp.empty:
        st.info("No metadata available.")
    else:
        st.dataframe(meta_cmp, width="stretch", hide_index=True)

elif section_key == "data":
    st.header("🗂️ Data")
    st.dataframe(summary.head(200), width="stretch", hide_index=True)

elif section_key == "about":
    st.header("ℹ️ About this app")
    tabs = st.tabs(["Overview", "Methods", "Using the app", "Team & Contact", "Changelog"])
    with tabs[0]:
        st.markdown(read_md("about/overview.md"))
    with tabs[1]:
        st.markdown(read_md("about/methods.md"))
    with tabs[2]:
        st.markdown(read_md("about/use_app.md"))
    with tabs[3]:
        st.markdown(read_md("about/team_contact.md"))
    with tabs[4]:
        st.markdown(read_md("about/changelog.md"))

try:
    footer_dt = None
    if "collection_date" in summary.columns and summary["collection_date"].notna().any():
        footer_dt = pd.to_datetime(summary["collection_date"], errors="coerce").max()
    render_sticky_footer(last_dt=footer_dt)
except Exception:
    pass
