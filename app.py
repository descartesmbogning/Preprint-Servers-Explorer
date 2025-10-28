import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from urllib.parse import quote, unquote

st.set_page_config(page_title="Preprints Tracker", page_icon="📄", layout="wide")

DATA_DIR = "data"  # put your files here: data/summary.csv, data/yearly.csv

# ───────────────────────── helpers ─────────────────────────
def _find_file(basename_no_ext: str):
    """Try CSV first, then XLSX."""
    csv_path = os.path.join(DATA_DIR, f"{basename_no_ext}.csv")
    xlsx_path = os.path.join(DATA_DIR, f"{basename_no_ext}.xlsx")
    if os.path.exists(csv_path): return csv_path
    if os.path.exists(xlsx_path): return xlsx_path
    return None

@st.cache_data
def read_any(path_or_buffer):
    if isinstance(path_or_buffer, str):
        if path_or_buffer.endswith(".csv"):
            return pd.read_csv(path_or_buffer)
        return pd.read_excel(path_or_buffer)
    # file-like (upload)
    name = getattr(path_or_buffer, "name", "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(path_or_buffer)
    return pd.read_excel(path_or_buffer)

@st.cache_data
def clean_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "server_name": "server_name",
        "number_of_preprint_records": "n_records",
        "is_version_of": "n_is_version_of",
        "number_of_preprint_unique": "n_unique",
        "is_published": "n_published",
        "%_published": "pct_published",
        "count_in_2024": "count_2024",
        "count_in_2025": "count_2025",
        "source": "source",
        "collection_date": "collection_date",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    if "server_name" not in df.columns:
        raise ValueError("Summary file must include 'server_name'.")

    num_cols = ["n_records", "n_is_version_of", "n_unique", "n_published", "pct_published", "count_2024", "count_2025"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "collection_date" in df.columns:
        df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")

    # Normalize % published: handle 0–1 and 0–100; clamp at 0+
    if "pct_published" in df.columns:
        frac_share = df["pct_published"].dropna().between(0, 1).mean()
        if frac_share > 0.9:
            df["pct_published"] = df["pct_published"] * 100
        df["pct_published"] = df["pct_published"].clip(lower=0)

    if "n_records" in df.columns and "n_unique" in df.columns:
        df["uniq_ratio"] = (df["n_unique"] / df["n_records"]).replace([np.inf, -np.inf], np.nan)
    if "n_records" in df.columns and "n_is_version_of" in df.columns:
        df["version_share"] = (df["n_is_version_of"] / df["n_records"]).replace([np.inf, -np.inf], np.nan)
    if "count_2025" in df.columns and "count_2024" in df.columns:
        df["growth_2025_vs_2024"] = (df["count_2025"] - df["count_2024"]).astype("Int64")

    df["server_name"] = df["server_name"].astype(str).str.strip()
    return df

@st.cache_data
def clean_yearly(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "server_name" not in df.columns:
        for cand in ["server", "serverName", "Server Name"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "server_name"})
                break
    if "server_name" not in df.columns:
        raise ValueError("Yearly file must have a 'server_name' column.")
    df["server_name"] = df["server_name"].astype(str).str.strip()

    year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    if not year_cols:
        year_cols = [c for c in df.columns if str(c).isdigit()]
    long_df = df.melt(id_vars=["server_name"], value_vars=year_cols, var_name="year", value_name="count")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce").fillna(0).astype(int)
    long_df = long_df.dropna(subset=["year"])
    return long_df

@st.cache_data
def yearly_totals(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("server_name", as_index=False)["count"].sum().rename(columns={"count": "total_all_years"})

def download_csv(df: pd.DataFrame, label, fname):
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    st.download_button(label, buff.getvalue(), file_name=fname, mime="text/csv")

# URL helpers
qp = st.query_params  # mutable mapping

def _qp_bool(key: str, default: bool) -> bool:
    v = qp.get(key)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")

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

def _encode_list(vals):
    # Use "|" as a separator; encode individual items for safety
    return "|".join(quote(str(v), safe="") for v in vals)

def _decode_list(s):
    if not s:
        return []
    return [unquote(x) for x in str(s).split("|") if x != ""]

# ───────────────────────── data loading ─────────────────────────
sum_path = _find_file("summary")
yr_path  = _find_file("yearly")

st.sidebar.header("Data source")
if not sum_path or not yr_path:
    st.sidebar.error("Missing bundled files. Add `data/summary.csv` (or .xlsx) and `data/yearly.csv` (or .xlsx).")
    st.stop()

# Quick reset
if st.sidebar.button("↺ Reset filters"):
    st.session_state.clear()
    try:
        st.query_params.clear()
    except Exception:
        pass
    st.rerun()

with st.sidebar.expander("Optional: override with uploads"):
    up_sum = st.file_uploader("Summary (CSV/XLSX)", type=["csv","xlsx"], key="sum_up")
    up_yr  = st.file_uploader("Yearly (CSV/XLSX)", type=["csv","xlsx"], key="yr_up")

summary_raw = read_any(up_sum if up_sum else sum_path)
yearly_raw  = read_any(up_yr if up_yr else yr_path)

try:
    summary = clean_summary(summary_raw)
    yearly  = clean_yearly(yearly_raw)
except Exception as e:
    st.error(f"Error parsing files: {e}")
    st.stop()

# ─────────────── filters (source, min_count, year range) ───────────────
has_source = "source" in summary.columns
src_opts = ["All"] + (sorted(summary["source"].dropna().unique()) if has_source else [])

# read URL defaults
qp_source   = qp.get("source", "")
qp_min      = qp.get("min_count", "0")
qp_yr_from  = qp.get("yr_from")
qp_yr_to    = qp.get("yr_to")

src_index = 0
if has_source and qp_source and qp_source in src_opts:
    src_index = src_opts.index(qp_source)

src_filter = st.sidebar.selectbox("Source", src_opts, index=src_index)
min_count_default = int(qp_min) if qp_min.isdigit() else 0
min_count = st.sidebar.number_input("Min records (summary)", min_value=0, value=min_count_default, step=100)

totals_from_yearly = yearly_totals(yearly)
summary = summary.merge(totals_from_yearly, on="server_name", how="left")

show = summary.copy()
if src_filter != "All" and has_source:
    show = show[show["source"] == src_filter]
if min_count and "n_records" in show.columns:
    show = show[show["n_records"].fillna(0) >= min_count]

yr_min = int(yearly["year"].min()) if len(yearly) else 2000
yr_max = int(yearly["year"].max()) if len(yearly) else 2025

# Prefer 1990 as the default start, but clamp to dataset min if needed
PREFERRED_START_YEAR = 1990

if qp_yr_from and str(qp_yr_from).isdigit():
    yr_from_default = int(qp_yr_from)
else:
    yr_from_default = max(yr_min, PREFERRED_START_YEAR)

if qp_yr_to and str(qp_yr_to).isdigit():
    yr_to_default = int(qp_yr_to)
else:
    yr_to_default = yr_max

yr_from, yr_to = st.sidebar.slider(
    "Year range",
    yr_min,
    yr_max,
    (max(yr_min, yr_from_default), min(yr_max, yr_to_default)),
    step=1
)
yearly_rng = yearly[(yearly["year"] >= yr_from) & (yearly["year"] <= yr_to)]
yearly_rng = yearly_rng[yearly_rng["server_name"].isin(show["server_name"])]

# ── Global, consistent color map for all charts
servers_all = sorted(show["server_name"].unique().tolist())
base_palette = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.D3
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Alphabet
    + px.colors.qualitative.Safe
    + px.colors.qualitative.Vivid
)

seen, long_palette = set(), []
for c in base_palette:
    if c not in seen:
        seen.add(c)
        long_palette.append(c)

if len(long_palette) < len(servers_all):
    need = len(servers_all) - len(long_palette)
    long_palette += [f"hsl({int(360*i/max(1,need))},70%,50%)" for i in range(need)]

color_map = {name: long_palette[i] for i, name in enumerate(servers_all)}
color_map["Other"] = "#9e9e9e"  # fixed gray for aggregate

# ── Suggestion order helper ────────────────────────────────
def _sorted_servers_for_suggestions(show_df: pd.DataFrame, yearly_rng_df: pd.DataFrame, mode: str, restrict_names=None):
    """
    mode in {"alpha","count"}.
    Returns a list of server names sorted per mode.
      - 'alpha'  → A..Z by name
      - 'count'  → Descending by total count in the current year range, ties A..Z
                   (servers absent from yearly_rng_df get 0).
    If restrict_names is provided (set or list), only names in it are considered.
    """
    names = show_df["server_name"].dropna().astype(str).unique().tolist()
    if restrict_names is not None:
        rset = set(restrict_names)
        names = [n for n in names if n in rset]

    if mode == "count":
        tot = (yearly_rng_df.groupby("server_name", as_index=False)["count"]
               .sum()
               .rename(columns={"count": "total"}))
        tmp = pd.DataFrame({"server_name": names})
        tmp = tmp.merge(tot, on="server_name", how="left")
        tmp["total"] = tmp["total"].fillna(0).astype(int)
        tmp = tmp.sort_values(["total", "server_name"], ascending=[False, True])
        return tmp["server_name"].tolist()

    return sorted(names, key=str.lower)

# ─────────────── ranking mode toggle (URL-synced) ──────────
rank_key_in = qp.get("rank_mode", "range")  # 'summary' or 'range'
rank_index_default = 0 if rank_key_in == "summary" else 1
rank_mode = st.sidebar.radio(
    "Top sources ranking based on",
    ["Summary total (all-time)", "Year range (from yearly file)"],
    index=rank_index_default
)
rank_key_out = "summary" if rank_mode.startswith("Summary") else "range"

# Persist core filters to URL
_update_qp_if_changed(
    source=(src_filter if has_source and src_filter != "All" else ""),
    min_count=min_count,
    yr_from=yr_from,
    yr_to=yr_to,
    rank_mode=rank_key_out,
)

# ─────────────── section/tab routing via URL ───────────────
st.title("📄 Preprint Tracker")
if "collection_date" in summary.columns and summary["collection_date"].notna().any():
    last_dt = pd.to_datetime(summary["collection_date"], errors="coerce").max()
    if pd.notna(last_dt):
        st.caption(f"Last updated (collection_date): **{last_dt.date()}**")

section_map = {
    "overview": "📊 Overview",
    "explorer": "🔎 Source Explorer",
    "compare":  "⚖️ Compare",
    "data":     "🗂️ Data",
}
rev_section_map = {v: k for k, v in section_map.items()}

section_default_key = qp.get("section", "overview")
section_default_label = section_map.get(section_default_key, "📊 Overview")
section_label = st.radio(
    "Section",
    options=list(section_map.values()),
    index=list(section_map.values()).index(section_default_label),
    horizontal=True,
)
section_key = rev_section_map[section_label]
_update_qp_if_changed(section=section_key)

# ───────────────────────── sections ─────────────────────────

if section_key == "overview":
    c1, c2, c3 = st.columns(3)
    servers_in_range = yearly_rng.loc[yearly_rng["count"] > 0, "server_name"].nunique()
    total_preprints_range = int(yearly_rng["count"].sum())
    unique_all_time = int(pd.to_numeric(show.get("n_unique", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    c1.metric("Sources (active in selected range)", servers_in_range)
    c2.metric(f"Preprints in range {yr_from}–{yr_to}", f"{total_preprints_range:,}")
    c3.metric("Preprints (all-time, unique)", f"{unique_all_time:,}")

    st.markdown("---")
    st.write("**Top sources**")

    # Build ranking according to selected mode
    if rank_key_out == "summary":
        if "n_records" in show.columns and show["n_records"].notna().any():
            ranking = show[["server_name", "n_records"]].dropna().rename(columns={"n_records": "total"})
            rank_title_suffix = " (summary, all-time)"
        else:
            ranking = totals_from_yearly.rename(columns={"total_all_years": "total"})
            ranking = ranking[ranking["server_name"].isin(show["server_name"])]
            rank_title_suffix = " (yearly fallback, all-time)"
        ranking = ranking[ranking["total"] > 0]
    else:
        ranking = (
            yearly_rng.groupby("server_name", as_index=False)["count"]
            .sum()
            .rename(columns={"count": "total"})
        )
        ranking = ranking[ranking["server_name"].isin(show["server_name"])]
        ranking = ranking[ranking["total"] > 0]
        rank_title_suffix = f" ({yr_from}–{yr_to})"

    n_rows = int(ranking.shape[0])
    if n_rows < 1:
        st.info("No sources to show for the current filters or year range.")
        topN = None
    else:
        ranking_sorted = ranking.sort_values("total", ascending=False)
        qp_topN = qp.get("topN")
        default_n = min(15, n_rows) if n_rows > 1 else 1
        topN_default = int(qp_topN) if (qp_topN and qp_topN.isdigit()) else default_n
        topN_default = max(1, min(n_rows, topN_default))

        if n_rows == 1:
            topN = 1
            st.caption("Only one server matches the current filters.")
        else:
            topN = st.slider(
                "Show top N",
                min_value=1,
                max_value=n_rows,
                value=topN_default,
                step=1,
                key="topn_ranking"
            )

        top_df = ranking_sorted.head(topN)

        if len(top_df) == 0:
            st.info("No sources with counts greater than zero in this range.")
        else:
            fig_bar = px.bar(
                top_df,
                x="total",
                y="server_name",
                orientation="h",
                labels={"total": "Total preprints", "server_name": "Server"},
                title=f"Top {topN} sources{rank_title_suffix}",
                # color="server_name",
                color_discrete_map=color_map,
                category_orders={"server_name": top_df.sort_values("total", ascending=True)["server_name"].tolist()},
            )
            fig_bar.update_layout(
                showlegend=False,
                yaxis={"categoryorder": "total ascending"},
                height=min(200 + 28 * len(top_df), 900),
                margin=dict(l=140, r=20, t=60, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        _update_qp_if_changed(topN=(topN if topN is not None else ""))

    # Yearly trend — stacked area
    st.markdown("**Yearly trend — stacked area**")
    col_sa1, col_sa2, col_sa3 = st.columns([1.2, 1, 1])
    with col_sa1:
        n_rows_for_stack = int(ranking.shape[0]) if 'ranking' in locals() else 0
        if n_rows_for_stack <= 1:
            stackN = 1
            if n_rows_for_stack == 0:
                st.caption("No ranked sources — using top of year-range totals.")
            else:
                st.caption("Only one server available for stacked area.")
        elif n_rows_for_stack == 2:
            qp_saN = qp.get("saN")
            saN_default = int(qp_saN) if (qp_saN and qp_saN.isdigit()) else 2
            saN_default = max(1, min(2, saN_default))
            stackN = st.slider(
                "Top N for stacked area",
                min_value=1,
                max_value=2,
                value=saN_default,
                step=1,
                key="stackN",
                help="With only two sources available, you can show one or both."
            )
        else:
            default_stack_n = min(10, n_rows_for_stack)
            qp_saN = qp.get("saN")
            saN_default = int(qp_saN) if (qp_saN and qp_saN.isdigit()) else default_stack_n
            saN_default = max(2, min(n_rows_for_stack, saN_default))
            stackN = st.slider(
                "Top N for stacked area",
                min_value=2,
                max_value=n_rows_for_stack,
                value=saN_default,
                step=1,
                key="stackN"
            )
    with col_sa2:
        show_other_default = _qp_bool("saOther", True)
        show_other = st.checkbox("Show ‘Other’ band", value=show_other_default, key="stack_show_other")
    with col_sa3:
        percent_mode_default = _qp_bool("saPct", False)
        percent_mode = st.checkbox("Show % of total", value=percent_mode_default, key="stack_percent")

    if 'ranking' in locals() and n_rows_for_stack > 0:
        top_names = (
            ranking.sort_values("total", ascending=False)
                   .head(stackN)["server_name"]
                   .tolist()
        )
    else:
        tmp = (yearly_rng.groupby("server_name", as_index=False)["count"].sum()
               .sort_values("count", ascending=False))
        top_names = tmp[tmp["count"] > 0].head(stackN)["server_name"].tolist()

    df_top = yearly_rng[yearly_rng["server_name"].isin(top_names)].copy()

    if show_other:
        df_other = yearly_rng[~yearly_rng["server_name"].isin(top_names)]
        if not df_other.empty:
            df_other = df_other.groupby("year", as_index=False)["count"].sum()
            df_other["server_name"] = "Other"
            df_plot = pd.concat([df_top, df_other], ignore_index=True)
        else:
            df_plot = df_top
    else:
        df_plot = df_top

    if not df_plot.empty:
        df_plot = df_plot[df_plot["count"] > 0]
        if not df_plot.empty:
            year_totals = df_plot.groupby("year")["count"].sum()
            nonzero_years = year_totals[year_totals > 0].index
            df_plot = df_plot[df_plot["year"].isin(nonzero_years)]

    if df_plot.empty:
        st.info("No data to display for the current filters/year range.")
    else:
        order_df = df_plot.groupby("server_name", as_index=False)["count"].sum().sort_values("count", ascending=False)
        labels_order = [s for s in order_df["server_name"] if s != "Other"]
        if "Other" in order_df["server_name"].values:
            labels_order += ["Other"]
        category_order = {"server_name": labels_order}

        fig_stack = px.area(
            df_plot.sort_values(["year", "server_name"]),
            x="year",
            y="count",
            color="server_name",
            category_orders=category_order,
            color_discrete_map=color_map,
            groupnorm="fraction" if percent_mode else None,
            labels={"count": "Preprints" if not percent_mode else "Share", "year": "Year", "server_name": "Server"},
            title=f"Top {min(stackN, len(category_order['server_name']))} sources{f' + Other' if show_other else ''} • {yr_from}–{yr_to}"
        )

        others_exist = show_other and "Other" in df_plot["server_name"].unique()
        num_layers = len([s for s in df_plot["server_name"].unique() if s != "Other"]) + (1 if others_exist else 0)

        base_height = 320
        per_layer = 28
        max_height = 900
        adaptive_height = min(base_height + per_layer * max(num_layers, 1), max_height)
        if percent_mode:
            adaptive_height = int(adaptive_height * 0.9)

        legend_conf = {}
        if num_layers >= 12:
            legend_conf = {"legend_orientation": "h", "legend_y": -0.2}

        fig_stack.update_layout(
            legend_title_text="Server",
            height=adaptive_height,
            margin=dict(t=70, r=20, b=40, l=60),
            **legend_conf
        )

        if percent_mode:
            fig_stack.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Year=%{x}<br>Share=%{y:.1%}<extra></extra>")
        else:
            fig_stack.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Year=%{x}<br>Count=%{y:,}<extra></extra>")

        st.plotly_chart(fig_stack, use_container_width=True)

        with st.expander("Download stacked-area dataset"):
            download_csv(df_plot.sort_values(["server_name", "year"]),
                         "⬇️ Download (stacked data, CSV)", "stacked_area_data.csv")

    _update_qp_if_changed(saN=stackN, saOther=(1 if show_other else 0), saPct=(1 if percent_mode else 0))

elif section_key == "explorer":
    st.subheader("Source Explorer")

    # --- Suggestions order (local to Explorer) ---
    suggest_modes = {"By total in selected range": "count", "Alphabetical (A–Z)": "alpha"}
    qp_suggest = qp.get("suggest", "count")  # default = count
    expl_suggest_label = st.radio(
        "Suggestions order",
        list(suggest_modes.keys()),
        index=0 if qp_suggest not in ("alpha",) else 1,
        horizontal=True,
        help="Controls how the list below is sorted.",
        key="explorer_suggest_radio",
    )
    expl_suggest_mode = suggest_modes[expl_suggest_label]
    _update_qp_if_changed(suggest=expl_suggest_mode)

    # --- Use suggestion order for servers ---
    servers = _sorted_servers_for_suggestions(show, yearly_rng, expl_suggest_mode)
    if len(servers) == 0:
        st.info("No sources available with the current filters. Adjust filters in the sidebar.")
        st.stop()

    # URL-synced selected server
    qp_server = qp.get("server", "")
    sel_default = servers[0]
    if qp_server and qp_server in servers:
        sel_default = qp_server

    sel = st.selectbox("Choose a source", servers, index=servers.index(sel_default))
    _update_qp_if_changed(server=sel)

    left, right = st.columns([1, 2], gap="large")

    with left:
        row = summary.loc[summary["server_name"] == sel].head(1)
        st.markdown(f"### **{sel}**")

        sv_all = yearly[yearly["server_name"] == sel]
        sv_rng = yearly_rng[yearly_rng["server_name"] == sel]

        # Impactful key metrics
        st.metric(f"Preprints ({yr_from}–{yr_to})", f"{int(sv_rng['count'].sum()):,}")

        if not row.empty:
            # Extra context metrics if available
            if pd.notna(row.iloc[0].get("n_records", np.nan)):
                st.metric("Records (summary)", f"{int(row.iloc[0]['n_records']):,}")
            if pd.notna(row.iloc[0].get("n_unique", np.nan)):
                st.metric("Unique (summary)", f"{int(row.iloc[0]['n_unique']):,}")
            n_pub = row.iloc[0].get("n_published", np.nan)
            pct = row.iloc[0].get("pct_published", np.nan)
            if pd.notna(n_pub) and pd.notna(pct):
                # Show both count and percent in one metric
                st.metric("Published (summary)", f"{int(n_pub):,} ({float(pct):.2f}%)")
            elif pd.notna(n_pub):
                st.metric("Published (summary)", f"{int(n_pub):,}")
            elif pd.notna(pct):
                st.metric("% Published (summary)", f"{float(pct):.2f}%")
            if 'n_is_version_of' in row.columns and pd.notna(row.iloc[0].get('n_is_version_of')):
                st.metric("Version(s)", f"{int(row.iloc[0]['n_is_version_of']):,}")
            if pd.notna(row.iloc[0].get("count_2024", np.nan)) or pd.notna(row.iloc[0].get("count_2025", np.nan)):
                c24 = row.iloc[0].get("count_2024", np.nan)
                c25 = row.iloc[0].get("count_2025", np.nan)
                if pd.notna(c24): st.caption(f"Count 2024 (summary): **{int(c24):,}**")
                if pd.notna(c25): st.caption(f"Count 2025 (summary): **{int(c25):,}**")

    # Bigger, cleaner Summary row
    st.markdown("### Summary row")
    st.dataframe(
        row.reset_index(drop=True),
        use_container_width=False,
        hide_index=True,
        # height=260,  # larger, impactful view
    )

    with right:
        # Header + Download button on same row (clean + actionable)
        hdr_col, dl_col = st.columns([4, 1], gap="small")
        with hdr_col:
            st.caption("Yearly trend")

        sv_all = yearly[yearly["server_name"] == sel].sort_values("year")
        sv_nz = sv_all[sv_all["count"] > 0]

        # Prepare safe filename once
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sel)[:80]
        with dl_col:
            st.download_button(
                "⬇️ CSV",
                sv_all.to_csv(index=False),
                file_name=f"{safe_name}_yearly.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download this server’s yearly data",
            )

        if sv_nz.empty:
            st.info("No yearly preprints for this server in the dataset.")
        else:
            fig = px.line(
                sv_nz,
                x="year",
                y="count",
                markers=True,
                color="server_name",
                color_discrete_map=color_map,
                labels={"count": "Preprints", "year": "Year"},
                title=f"{sel} • yearly preprints",
            )
            fig.update_layout(showlegend=False, margin=dict(t=60, r=20, b=40, l=60))
            st.plotly_chart(fig, use_container_width=True)




elif section_key == "compare":
    st.subheader("Compare Sources")

    # Local Suggestions order (Compare) — default to "count"
    suggest_modes = {"By total in selected range": "count", "Alphabetical (A–Z)": "alpha"}
    qp_suggest = qp.get("suggest", "count")
    comp_suggest_label = st.radio(
        "Suggestions order",
        list(suggest_modes.keys()),
        index=0 if qp_suggest != "alpha" else 1,
        horizontal=True,
        help="Controls the suggestions list for the picker below.",
        key="compare_suggest_radio",
    )
    comp_suggest_mode = suggest_modes[comp_suggest_label]
    _update_qp_if_changed(suggest=comp_suggest_mode)

    active_names = set(yearly_rng.loc[yearly_rng["count"] > 0, "server_name"].unique().tolist())
    servers_active = _sorted_servers_for_suggestions(show, yearly_rng, comp_suggest_mode, restrict_names=active_names)

    qp_cmp = _decode_list(qp.get("cmp", ""))
    qp_cmp = [s for s in qp_cmp if s in servers_active]

    if len(qp_cmp) >= 2:
        default_pick = qp_cmp
    else:
        default_pick = servers_active[:3] if len(servers_active) >= 3 else servers_active

    pick = st.multiselect("Pick 2–10 sources", options=servers_active, default=default_pick, max_selections=10)
    _update_qp_if_changed(cmp=_encode_list(pick))

    cmp_percent_default = _qp_bool("cmpPct", False)
    cmp_percent = st.checkbox("Show % of total (stacked)", value=cmp_percent_default, key="cmp_percent")

    if len(pick) >= 2:
        cmp = yearly_rng[yearly_rng["server_name"].isin(pick)].copy()
        cmp = cmp[cmp["count"] > 0]
        if not cmp.empty:
            yr_tot = cmp.groupby("year")["count"].sum()
            cmp = cmp[cmp["year"].isin(yr_tot[yr_tot > 0].index)]

        if cmp.empty:
            st.info("No non-zero data for the chosen sources in this range.")
        else:
            if cmp_percent:
                fig_cmp = px.area(
                    cmp, x="year", y="count", color="server_name",
                    color_discrete_map=color_map,
                    groupnorm="fraction",
                    labels={"count": "Share", "year": "Year", "server_name": "Server"},
                    title=f"Comparison (share) • {yr_from}–{yr_to}"
                )
            else:
                fig_cmp = px.line(
                    cmp, x="year", y="count", color="server_name", markers=True,
                    color_discrete_map=color_map,
                    labels={"count": "Preprints", "year": "Year", "server_name": "Server"},
                    title=f"Comparison • {yr_from}–{yr_to}"
                )
            st.plotly_chart(fig_cmp, use_container_width=True)

            agg = (cmp.groupby("server_name", as_index=False)
                      .agg(total_in_range=("count", "sum"),
                           mean_per_year=("count", "mean")))
            agg["mean_per_year"] = agg["mean_per_year"].round(1)
            st.dataframe(agg.sort_values("total_in_range", ascending=False),
                         use_container_width=True, hide_index=True)
    else:
        st.info("Select at least two sources to compare.")

    _update_qp_if_changed(cmpPct=(1 if cmp_percent else 0))

elif section_key == "data":
    st.header("🗂️ Data")

    # Only FULL datasets view (Filtered view removed)
    st.subheader("Original & cleaned datasets")

    st.markdown(f"### Summary (original columns, wide) · {len(summary_raw):,} rows")
    show_all_sum = st.checkbox("Show all rows (summary)", value=False, key="show_all_sum")
    sum_view = summary_raw if show_all_sum else summary_raw.head(200)
    st.dataframe(sum_view, use_container_width=True, hide_index=True)
    download_csv(summary_raw, "⬇️ Download summary (original, CSV)", "summary_original.csv")

    st.divider()

    st.markdown(f"### Yearly (original columns, wide) · {len(yearly_raw):,} rows")
    show_all_yr = st.checkbox("Show all rows (yearly, wide)", value=False, key="show_all_yr")
    yr_view = yearly_raw if show_all_yr else yearly_raw.head(200)
    st.dataframe(yr_view, use_container_width=True, hide_index=True)
    download_csv(yearly_raw, "⬇️ Download yearly (original wide, CSV)", "yearly_original_wide.csv")

    st.markdown(f"### Yearly (cleaned, long) · {len(yearly):,} rows")
    show_all_long = st.checkbox("Show all rows (yearly, long)", value=False, key="show_all_long")
    yearly_long_view = yearly if show_all_long else yearly.head(500)
    st.dataframe(
        yearly_long_view.sort_values(["server_name", "year"]),
        use_container_width=True, hide_index=True
    )
    download_csv(yearly, "⬇️ Download yearly (cleaned long, CSV)", "yearly_cleaned_long.csv")

    
