import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Preprint Servers — Explorer", page_icon="📄", layout="wide")

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

    num_cols = ["n_records","n_is_version_of","n_unique","n_published","pct_published","count_2024","count_2025"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "collection_date" in df.columns:
        df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")

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
    long_df = df.melt(id_vars=["server_name"], value_vars=year_cols,
                      var_name="year", value_name="count")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce").fillna(0).astype(int)
    long_df = long_df.dropna(subset=["year"])
    return long_df

def download_csv(df: pd.DataFrame, label, fname):
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    st.download_button(label, buff.getvalue(), file_name=fname, mime="text/csv")

# ───────────────────────── data loading ─────────────────────────
sum_path = _find_file("summary")
yr_path  = _find_file("yearly")

st.sidebar.header("Data source")
if not sum_path or not yr_path:
    st.sidebar.error("Missing bundled files. Add `data/summary.csv` (or .xlsx) and `data/yearly.csv` (or .xlsx).")
    st.stop()

st.sidebar.success("Using bundled data in `data/`")

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

# Filters
src_opts = ["All"] + (sorted([s for s in summary["source"].dropna().unique()]) if "source" in summary.columns else [])
src_filter = st.sidebar.selectbox("Source", src_opts, index=0)
min_count = st.sidebar.number_input("Min records (summary)", min_value=0, value=0, step=100)

totals_from_yearly = (yearly.groupby("server_name", as_index=False)["count"]
                      .sum().rename(columns={"count":"total_all_years"}))
summary = summary.merge(totals_from_yearly, on="server_name", how="left")

show = summary.copy()
if src_filter != "All" and "source" in show.columns:
    show = show[show["source"] == src_filter]
if min_count and "n_records" in show.columns:
    show = show[show["n_records"].fillna(0) >= min_count]

yr_min = int(yearly["year"].min()) if len(yearly) else 2000
yr_max = int(yearly["year"].max()) if len(yearly) else 2025
yr_from, yr_to = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max), step=1)
yearly_rng = yearly[(yearly["year"] >= yr_from) & (yearly["year"] <= yr_to)]
yearly_rng = yearly_rng[yearly_rng["server_name"].isin(show["server_name"])]

# NEW: ranking mode toggle
rank_mode = st.sidebar.radio(
    "Top servers ranking based on",
    ["Summary total (all-time)", "Year range (from yearly file)"],
    index=1  # default to dynamic ranking that follows the slider
)

# Header + last update
st.title("📄 Preprint Servers — Explorer")
if "collection_date" in summary.columns and summary["collection_date"].notna().any():
    last_dt = pd.to_datetime(summary["collection_date"], errors="coerce").max()
    if pd.notna(last_dt):
        st.caption(f"Last updated (collection_date): **{last_dt.date()}**")

# Tabs
tab_overview, tab_explorer, tab_compare, tab_data = st.tabs(
    ["📊 Overview", "🔎 Server Explorer", "⚖️ Compare", "🗂️ Data"]
)

# Overview
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Servers", show["server_name"].nunique())
    c2.metric(f"Preprints in range {yr_from}–{yr_to}", f"{int(yearly_rng['count'].sum()):,}")
    c3.metric("Preprints (all-time, yearly)", f"{int(yearly['count'].sum()):,}")

    st.markdown("---")
    st.write("**Top servers**")

    # UPDATED: build ranking according to the chosen mode
    if rank_mode == "Summary total (all-time)":
        if "n_records" in show.columns and show["n_records"].notna().any():
            ranking = (show[["server_name","n_records"]]
                       .dropna()
                       .rename(columns={"n_records":"total"})
                      )
            rank_title_suffix = " (summary, all-time)"
        else:
            # fallback: if summary lacks totals, use all-time totals from yearly file
            ranking = (totals_from_yearly
                       .rename(columns={"total_all_years":"total"})
                      )
            ranking = ranking[ranking["server_name"].isin(show["server_name"])]
            rank_title_suffix = " (yearly fallback, all-time)"
    else:
        # Year-range mode: sum only within selected years
        ranking = (yearly_rng.groupby("server_name", as_index=False)["count"]
                   .sum()
                   .rename(columns={"count":"total"})
                  )
        ranking = ranking[ranking["server_name"].isin(show["server_name"])]
        rank_title_suffix = f" ({yr_from}–{yr_to})"

    topN = st.slider("Show top N", 5, 50, 15, 5)
    top_df = ranking.sort_values("total", ascending=False).head(topN)

    fig_bar = px.bar(
        top_df, x="total", y="server_name", orientation="h",
        labels={"total":"Total preprints","server_name":"Server"},
        title=f"Top {topN} servers{rank_title_suffix}"
    )
    fig_bar.update_layout(yaxis={"categoryorder":"total ascending"}, height=600)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("**Overall yearly trend (filtered)**")
    yearly_total = yearly_rng.groupby("year", as_index=False)["count"].sum()
    fig_line_total = px.line(yearly_total, x="year", y="count", markers=True,
                             labels={"count":"Preprints","year":"Year"},
                             title=f"All servers • {yr_from}–{yr_to}")
    st.plotly_chart(fig_line_total, use_container_width=True)

# Explorer
with tab_explorer:
    st.subheader("Server Explorer")
    servers = sorted(show["server_name"].unique().tolist())
    sel = st.selectbox("Choose a server", servers)

    left, right = st.columns([1,2], gap="large")
    with left:
        row = summary.loc[summary["server_name"] == sel].head(1)
        st.markdown(f"### **{sel}**")

        sv_all = yearly[yearly["server_name"] == sel]
        sv_rng = yearly_rng[yearly_rng["server_name"] == sel]
        st.metric(f"Preprints ({yr_from}–{yr_to})", f"{int(sv_rng['count'].sum()):,}")
        st.metric("Preprints (all-time, yearly)", f"{int(sv_all['count'].sum()):,}")

        if not row.empty:
            if pd.notna(row.iloc[0].get("n_records", np.nan)):
                st.metric("Records (summary)", f"{int(row.iloc[0]['n_records']):,}")
            if pd.notna(row.iloc[0].get("n_unique", np.nan)):
                st.metric("Unique (summary)", f"{int(row.iloc[0]['n_unique']):,}")
            if pd.notna(row.iloc[0].get("pct_published", np.nan)):
                raw_pct = float(row.iloc[0]["pct_published"])
                # If <=1 assume it's a fraction; else assume it's already a percent value
                pct_display = raw_pct * 100 if raw_pct <= 1 else raw_pct
                st.metric("% Published (summary)", f"{pct_display:.2f}%")
            if pd.notna(row.iloc[0].get("count_2024", np.nan)) or pd.notna(row.iloc[0].get("count_2025", np.nan)):
                c24 = row.iloc[0].get("count_2024", np.nan)
                c25 = row.iloc[0].get("count_2025", np.nan)
                if pd.notna(c24): st.caption(f"Count 2024 (summary): **{int(c24):,}**")
                if pd.notna(c25): st.caption(f"Count 2025 (summary): **{int(c25):,}**")

        if not row.empty:
            st.caption("Summary row")
            st.dataframe(row.reset_index(drop=True), use_container_width=True)

    with right:
        st.caption("Yearly trend")
        sv = sv_all.sort_values("year")
        fig = px.line(sv, x="year", y="count", markers=True,
                      labels={"count":"Preprints","year":"Year"},
                      title=f"{sel} • yearly preprints")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Data for this server (yearly)")
        st.dataframe(sv, use_container_width=True, hide_index=True)

# Compare
with tab_compare:
    st.subheader("Compare Servers")
    default_pick = servers[:3] if len(servers) >= 3 else servers
    pick = st.multiselect("Pick 2–10 servers", options=servers, default=default_pick)
    if len(pick) >= 2:
        cmp = yearly_rng[yearly_rng["server_name"].isin(pick)]
        fig_cmp = px.line(cmp, x="year", y="count", color="server_name", markers=True,
                          labels={"count":"Preprints","year":"Year","server_name":"Server"},
                          title=f"Comparison • {yr_from}–{yr_to}")
        st.plotly_chart(fig_cmp, use_container_width=True)

        agg = (cmp.groupby("server_name", as_index=False)
                  .agg(total_in_range=("count","sum"),
                       mean_per_year=("count","mean")))
        agg["mean_per_year"] = agg["mean_per_year"].round(1)
        st.dataframe(agg.sort_values("total_in_range", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Select at least two servers to compare.")


# Data (export)
with tab_data:
    sub1, sub2 = st.tabs(["🔎 Filtered (current view)", "📦 Full datasets"])

    # ── Filtered slice (uses current year range + optional picks)
    with sub1:
        st.subheader("Filtered data (based on year range and server selection)")
        export_df = yearly_rng if "pick" not in locals() or not pick else yearly_rng[yearly_rng["server_name"].isin(pick)]
        st.dataframe(export_df.sort_values(["server_name","year"]), use_container_width=True, hide_index=True)
        download_csv(export_df, "⬇️ Download filtered CSV", "preprints_filtered.csv")

    # ── Full datasets (original & cleaned)
    with sub2:
        st.subheader("Original & cleaned datasets")

        # (A) Summary file — ORIGINAL columns (exactly as loaded)
        st.markdown("### Summary (original columns, wide)")
        # To avoid rendering huge tables by default, show first 200 rows with a toggle
        show_all_sum = st.checkbox("Show all rows (summary)", value=False)
        sum_view = summary_raw if show_all_sum else summary_raw.head(200)
        st.dataframe(sum_view, use_container_width=True, hide_index=True)
        download_csv(summary_raw, "⬇️ Download summary (original, CSV)", "summary_original.csv")

        st.divider()

        # (B) Yearly file — ORIGINAL columns (wide, with one column per year)
        st.markdown("### Yearly (original columns, wide)")
        show_all_yr = st.checkbox("Show all rows (yearly, wide)", value=False)
        yr_view = yearly_raw if show_all_yr else yearly_raw.head(200)
        st.dataframe(yr_view, use_container_width=True, hide_index=True)
        download_csv(yearly_raw, "⬇️ Download yearly (original wide, CSV)", "yearly_original_wide.csv")

        # (C) Yearly file — CLEANED (long form: server_name, year, count)
        st.markdown("### Yearly (cleaned, long)")
        show_all_long = st.checkbox("Show all rows (yearly, long)", value=False)
        yearly_long_view = yearly if show_all_long else yearly.head(500)
        st.dataframe(yearly_long_view.sort_values(["server_name","year"]),
                     use_container_width=True, hide_index=True)
        download_csv(yearly, "⬇️ Download yearly (cleaned long, CSV)", "yearly_cleaned_long.csv")

        st.caption(
            "Notes: "
            "• ‘Original’ tables reflect exactly what was bundled in `data/`. "
            "• ‘Cleaned long’ is the normalized format used for charts (one row per server-year)."
        )
