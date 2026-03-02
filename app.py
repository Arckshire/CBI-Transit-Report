import io
import re
import math
import zipfile
from typing import Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz


# -----------------------------
# Helpers: fuzzy column finding
# -----------------------------
def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[\s\-_\/]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def best_match_column(columns: List[str], patterns: List[str], min_score: int = 70) -> Optional[str]:
    """
    Returns the best matching column using fuzzy matching over normalized strings.
    """
    cols_norm = {c: _normalize(c) for c in columns}

    best_col = None
    best_score = -1

    for col, col_norm in cols_norm.items():
        for pat in patterns:
            pat_norm = _normalize(pat)
            # token_set_ratio is robust to reordering words
            score = fuzz.token_set_ratio(col_norm, pat_norm)
            if score > best_score:
                best_score = score
                best_col = col

    if best_score >= min_score:
        return best_col
    return None


def to_bool_series(s: pd.Series) -> pd.Series:
    """
    Convert common true/false representations into booleans (True/False),
    with anything unknown -> False.
    """
    if s is None:
        return pd.Series([False] * 0)

    def conv(x):
        if pd.isna(x):
            return False
        if isinstance(x, bool):
            return x
        v = str(x).strip().lower()
        if v in {"true", "t", "1", "yes", "y"}:
            return True
        if v in {"false", "f", "0", "no", "n"}:
            return False
        return False

    return s.map(conv)


def parse_dt_utc(s: pd.Series) -> pd.Series:
    """
    Parse datetimes; treat as UTC. If source already has timezone, it will be normalized.
    """
    # utc=True will localize naive times to UTC (assumption matches "UTC timestamp raw")
    return pd.to_datetime(s, errors="coerce", utc=True)


def split_city_state(val) -> Tuple[str, str]:
    if pd.isna(val):
        return "", ""
    txt = str(val).strip()
    if not txt:
        return "", ""
    # Split on last hyphen to preserve city names that may contain hyphens
    if "-" in txt:
        left, right = txt.rsplit("-", 1)
        return left.strip(), right.strip()
    return txt, ""


def round_days_from_hours(hours: float) -> float:
    """
    Your rounding logic:
    - Convert to days_raw = hours / 24
    - If 0 < days_raw < 0.5 => 0.5
    - Else round to nearest integer with .5 rounding UP
    - No decimals except the special 0.5 case
    """
    if hours is None or (isinstance(hours, float) and math.isnan(hours)):
        return 0
    if hours <= 0:
        return 0
    days_raw = hours / 24.0
    if 0 < days_raw < 0.5:
        return 0.5
    # half-up rounding:
    return float(math.floor(days_raw + 0.5))


def fmt_hours(x: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return float(round(x, 2))


def safe_unique_join(values: List[str], limit: int = 25) -> str:
    """
    Join unique items into a single string; if too many, truncate.
    """
    uniq = []
    seen = set()
    for v in values:
        if v is None:
            continue
        t = str(v).strip()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if len(uniq) <= limit:
        return "; ".join(uniq)
    return "; ".join(uniq[:limit]) + f"; (+{len(uniq)-limit} more)"


# -----------------------------
# Report building
# -----------------------------
def build_report(
    df_raw: pd.DataFrame,
    colmap: Dict[str, Optional[str]],
    lane_arrow: str = " → "
) -> Dict[str, pd.DataFrame]:
    """
    Returns dataframes for:
      - summary_detail_df
      - carrier_perf_df
      - lane_perf_df (grouped layout)
    Also returns counts for summary table via dict keys.
    """
    df = df_raw.copy()

    # Pull mapped columns
    c_tracked = colmap.get("tracked")
    c_pickup_dt = colmap.get("pickup_departure")
    c_dropoff_dt = colmap.get("dropoff_arrival")

    # Shipment-level
    c_bol = colmap.get("bill_of_lading")
    c_carrier = colmap.get("carrier_name")
    c_tenant = colmap.get("tenant_name")
    c_scac = colmap.get("scac")

    c_pu_name = colmap.get("pickup_name")
    c_pu_citystate = colmap.get("pickup_city_state")
    c_pu_country = colmap.get("pickup_country")

    c_do_name = colmap.get("dropoff_name")
    c_do_citystate = colmap.get("dropoff_city_state")
    c_do_country = colmap.get("dropoff_country")

    # Core: tracked + timestamps
    tracked = to_bool_series(df[c_tracked]) if c_tracked else pd.Series([False] * len(df))
    pu_dt = parse_dt_utc(df[c_pickup_dt]) if c_pickup_dt else pd.Series([pd.NaT] * len(df))
    do_dt = parse_dt_utc(df[c_dropoff_dt]) if c_dropoff_dt else pd.Series([pd.NaT] * len(df))

    transit_hours = (do_dt - pu_dt).dt.total_seconds() / 3600.0
    # Missed milestone logic (tracked only)
    missed_milestone = tracked & (pu_dt.isna() | do_dt.isna() | (transit_hours < 0))
    valid_tracked = tracked & (~missed_milestone)

    # Summary counts
    tracked_count = int(tracked.sum())
    untracked_count = int((~tracked).sum())
    missed_count = int(missed_milestone.sum())
    total_count = int(len(df))

    # Build shipment detail table (Summary sheet detail)
    def get_col(series_col, default=""):
        if series_col and series_col in df.columns:
            return df[series_col]
        return pd.Series([default] * len(df))

    pu_city, pu_state = zip(*get_col(c_pu_citystate, "").map(split_city_state).tolist()) if c_pu_citystate else ([""]*len(df), [""]*len(df))
    do_city, do_state = zip(*get_col(c_do_citystate, "").map(split_city_state).tolist()) if c_do_citystate else ([""]*len(df), [""]*len(df))

    detail = pd.DataFrame({
        "Bill of lading": get_col(c_bol, ""),
        "Carrier name": get_col(c_carrier, ""),
        "Pickup name": get_col(c_pu_name, ""),
        "Pickup city": list(pu_city),
        "Pickup state": list(pu_state),
        "Pickup country": get_col(c_pu_country, ""),
        "Drop-off name": get_col(c_do_name, ""),
        "Drop-off city": list(do_city),
        "Drop-off state": list(do_state),
        "Drop-off country": get_col(c_do_country, ""),
        "Transit time (hours)": transit_hours.map(fmt_hours),
        "Transit time (days)": transit_hours.map(round_days_from_hours),
    })

    # Filter to valid tracked shipments only
    detail_valid = detail[valid_tracked].reset_index(drop=True)

    # Build lane string (city only)
    lane_series = pd.Series(list(pu_city)) + lane_arrow + pd.Series(list(do_city))
    lane_series = lane_series.fillna("").map(lambda x: x.strip())

    # For performance sheets, use valid shipments only
    df_perf = df.copy()
    df_perf["_tracked"] = tracked
    df_perf["_missed"] = missed_milestone
    df_perf["_valid"] = valid_tracked
    df_perf["_transit_hours"] = transit_hours
    df_perf["_transit_hours"] = df_perf["_transit_hours"].where(df_perf["_valid"], other=pd.NA)
    df_perf["_lane"] = lane_series.where(df_perf["_valid"], other=pd.NA)

    # Tenant / carrier / scac fields
    df_perf["_tenant"] = get_col(c_tenant, "")
    df_perf["_carrier"] = get_col(c_carrier, "")
    df_perf["_scac"] = get_col(c_scac, "")

    df_valid = df_perf[df_perf["_valid"]].copy()

    # -----------------------------
    # Carrier Performance
    # -----------------------------
    if len(df_valid) == 0:
        carrier_perf = pd.DataFrame(columns=[
            "Tenant name", "Carrier name", "Carrier SCAC", "Shipment volume", "Lanes",
            "Total transit time (hours)", "Total transit time (days)",
            "Average transit time (hours)", "Average transit time (days)",
            "Median transit time (hours)", "Median transit time (days)",
            "Maximum transit time (hours)", "Maximum transit time (days)",
        ])
    else:
        grp = df_valid.groupby(["_tenant", "_carrier"], dropna=False)

        rows = []
        for (tenant, carrier), g in grp:
            hours_list = g["_transit_hours"].dropna().astype(float).tolist()
            if not hours_list:
                continue

            shipment_volume = int(len(hours_list))
            total_hours = float(sum(hours_list))
            avg_hours = float(pd.Series(hours_list).mean())
            med_hours = float(pd.Series(hours_list).median())
            max_hours = float(max(hours_list))

            # SCAC choice: most frequent non-empty
            scac_vals = g["_scac"].dropna().astype(str).map(str.strip)
            scac_vals = scac_vals[scac_vals != ""]
            scac = scac_vals.value_counts().index[0] if len(scac_vals) else ""

            lanes = g["_lane"].dropna().astype(str).tolist()
            lanes_txt = safe_unique_join(lanes, limit=25)

            rows.append({
                "Tenant name": tenant,
                "Carrier name": carrier,
                "Carrier SCAC": scac,
                "Shipment volume": shipment_volume,
                "Lanes": lanes_txt,
                "Total transit time (hours)": fmt_hours(total_hours),
                "Total transit time (days)": round_days_from_hours(total_hours),
                "Average transit time (hours)": fmt_hours(avg_hours),
                "Average transit time (days)": round_days_from_hours(avg_hours),
                "Median transit time (hours)": fmt_hours(med_hours),
                "Median transit time (days)": round_days_from_hours(med_hours),
                "Maximum transit time (hours)": fmt_hours(max_hours),
                "Maximum transit time (days)": round_days_from_hours(max_hours),
            })

        carrier_perf = pd.DataFrame(rows).sort_values(
            ["Tenant name", "Shipment volume", "Carrier name"],
            ascending=[True, False, True],
            kind="mergesort"
        ).reset_index(drop=True)

    # -----------------------------
    # Lane Performance (grouped)
    # -----------------------------
    lane_cols = [
        "Tenant name", "Lane", "Carrier name", "Carrier SCAC", "Shipment volume",
        "Total transit time (hours)", "Total transit time (days)",
        "Average transit time (hours)", "Average transit time (days)",
        "Median transit time (hours)", "Median transit time (days)",
        "Maximum transit time (hours)", "Maximum transit time (days)",
    ]

    if len(df_valid) == 0:
        lane_perf = pd.DataFrame(columns=lane_cols)
    else:
        lane_rows = []
        # Unique lanes per tenant
        grp_lane = df_valid.groupby(["_tenant", "_lane"], dropna=False)

        # Sort lanes by volume desc
        lane_summary = grp_lane.size().reset_index(name="_vol").sort_values(
            ["_tenant", "_vol", "_lane"], ascending=[True, False, True], kind="mergesort"
        )

        for _, r in lane_summary.iterrows():
            tenant = r["_tenant"]
            lane = r["_lane"]
            g_lane = df_valid[(df_valid["_tenant"] == tenant) & (df_valid["_lane"] == lane)]

            # Lane header row (carrier/metrics blank)
            lane_rows.append({
                "Tenant name": tenant,
                "Lane": lane,
                "Carrier name": "",
                "Carrier SCAC": "",
                "Shipment volume": "",
                "Total transit time (hours)": "",
                "Total transit time (days)": "",
                "Average transit time (hours)": "",
                "Average transit time (days)": "",
                "Median transit time (hours)": "",
                "Median transit time (days)": "",
                "Maximum transit time (hours)": "",
                "Maximum transit time (days)": "",
            })

            # Carrier rows within the lane
            grp_c = g_lane.groupby(["_carrier"], dropna=False)
            # Sort carriers by volume desc
            carrier_order = grp_c.size().reset_index(name="_volc").sort_values(
                ["_volc", "_carrier"], ascending=[False, True], kind="mergesort"
            )["_carrier"].tolist()

            for carrier in carrier_order:
                gc = g_lane[g_lane["_carrier"] == carrier]
                hours_list = gc["_transit_hours"].dropna().astype(float).tolist()
                if not hours_list:
                    continue

                shipment_volume = int(len(hours_list))
                total_hours = float(sum(hours_list))
                avg_hours = float(pd.Series(hours_list).mean())
                med_hours = float(pd.Series(hours_list).median())
                max_hours = float(max(hours_list))

                scac_vals = gc["_scac"].dropna().astype(str).map(str.strip)
                scac_vals = scac_vals[scac_vals != ""]
                scac = scac_vals.value_counts().index[0] if len(scac_vals) else ""

                lane_rows.append({
                    "Tenant name": "",  # keep blank for visual grouping
                    "Lane": "",
                    "Carrier name": carrier,
                    "Carrier SCAC": scac,
                    "Shipment volume": shipment_volume,
                    "Total transit time (hours)": fmt_hours(total_hours),
                    "Total transit time (days)": round_days_from_hours(total_hours),
                    "Average transit time (hours)": fmt_hours(avg_hours),
                    "Average transit time (days)": round_days_from_hours(avg_hours),
                    "Median transit time (hours)": fmt_hours(med_hours),
                    "Median transit time (days)": round_days_from_hours(med_hours),
                    "Maximum transit time (hours)": fmt_hours(max_hours),
                    "Maximum transit time (days)": round_days_from_hours(max_hours),
                })

        lane_perf = pd.DataFrame(lane_rows, columns=lane_cols)

    return {
        "tracked_count": pd.DataFrame({"value": [tracked_count]}),
        "untracked_count": pd.DataFrame({"value": [untracked_count]}),
        "missed_count": pd.DataFrame({"value": [missed_count]}),
        "total_count": pd.DataFrame({"value": [total_count]}),
        "summary_detail": detail_valid,
        "carrier_performance": carrier_perf,
        "lane_performance": lane_perf,
    }


def build_excel_bytes(
    df_raw: pd.DataFrame,
    summary_counts: Dict[str, int],
    summary_detail: pd.DataFrame,
    carrier_perf: pd.DataFrame,
    lane_perf: pd.DataFrame
) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Sheet 1: Raw Data
        df_raw.to_excel(writer, sheet_name="Raw Data", index=False)

        # Sheet 2: Summary
        workbook = writer.book
        ws = workbook.add_worksheet("Summary")
        writer.sheets["Summary"] = ws

        # Formats
        fmt_header = workbook.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
        fmt_label = workbook.add_format({"bold": True, "border": 1})
        fmt_cell = workbook.add_format({"border": 1})
        fmt_title = workbook.add_format({"bold": True})
        fmt_num = workbook.add_format({"border": 1, "num_format": "0"})
        fmt_hours = workbook.add_format({"border": 1, "num_format": "0.00"})

        # Summary table A1:B5
        ws.write(0, 0, "Label", fmt_header)
        ws.write(0, 1, "Shipment count", fmt_header)

        rows = [
            ("Tracked", summary_counts["tracked"]),
            ("Missed milestone", summary_counts["missed"]),
            ("Untracked", summary_counts["untracked"]),
            ("Grand total", summary_counts["total"]),
        ]
        for i, (lab, val) in enumerate(rows, start=1):
            ws.write(i, 0, lab, fmt_label)
            ws.write(i, 1, val, fmt_num)

        ws.set_column(0, 0, 20)
        ws.set_column(1, 1, 16)

        # Shipment detail header at row 7 (index 6)
        startrow = 6
        ws.write(startrow - 1, 0, "", None)  # just spacing row (row 6 visually)

        # Write a title (optional)
        ws.write(startrow - 1, 0, "Shipment-level detail (tracked & valid)", fmt_title)

        # Write detail dataframe starting row 7 (index 6)
        summary_detail.to_excel(writer, sheet_name="Summary", index=False, startrow=startrow)

        # Apply formatting to the detail header row
        for col_idx, col_name in enumerate(summary_detail.columns):
            ws.write(startrow, col_idx, col_name, fmt_header)

        # Set column widths
        widths = {
            "Bill of lading": 18,
            "Carrier name": 22,
            "Pickup name": 22,
            "Pickup city": 16,
            "Pickup state": 12,
            "Pickup country": 14,
            "Drop-off name": 22,
            "Drop-off city": 16,
            "Drop-off state": 12,
            "Drop-off country": 14,
            "Transit time (hours)": 18,
            "Transit time (days)": 18,
        }
        for idx, col in enumerate(summary_detail.columns):
            ws.set_column(idx, idx, widths.get(col, 16))

        # Format transit columns if present
        if "Transit time (hours)" in summary_detail.columns:
            h_idx = list(summary_detail.columns).index("Transit time (hours)")
            ws.set_column(h_idx, h_idx, widths.get("Transit time (hours)", 18), fmt_hours)
        if "Transit time (days)" in summary_detail.columns:
            d_idx = list(summary_detail.columns).index("Transit time (days)")
            ws.set_column(d_idx, d_idx, widths.get("Transit time (days)", 18), fmt_cell)

        # Sheet 3: Carrier Performance
        carrier_perf.to_excel(writer, sheet_name="Carrier Performance", index=False)
        ws_cp = writer.sheets["Carrier Performance"]
        # header format
        for c, name in enumerate(carrier_perf.columns):
            ws_cp.write(0, c, name, fmt_header)
        ws_cp.set_column(0, 0, 18)
        ws_cp.set_column(1, 1, 24)
        ws_cp.set_column(2, 2, 14)
        ws_cp.set_column(3, 3, 16)
        ws_cp.set_column(4, 4, 55)
        ws_cp.set_column(5, len(carrier_perf.columns)-1, 22)

        # Sheet 4: Lane Performance
        lane_perf.to_excel(writer, sheet_name="Lane Performance", index=False)
        ws_lp = writer.sheets["Lane Performance"]
        for c, name in enumerate(lane_perf.columns):
            ws_lp.write(0, c, name, fmt_header)
        ws_lp.set_column(0, 0, 18)
        ws_lp.set_column(1, 1, 30)
        ws_lp.set_column(2, 2, 24)
        ws_lp.set_column(3, 3, 14)
        ws_lp.set_column(4, len(lane_perf.columns)-1, 22)

    return output.getvalue()


def build_csv_zip_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """
    Creates a ZIP containing CSV files for Summary Detail, Carrier Performance, Lane Performance.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in sheets.items():
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            safe_name = name.replace(" ", "_").lower()
            z.writestr(f"{safe_name}.csv", csv_bytes)
    return buf.getvalue()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Transit Time Report Builder", layout="wide")
st.title("Transit Time Report Builder (CSV/XLSX → Excel report)")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded:
    # Read input
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    if df_raw.empty:
        st.warning("Uploaded file is empty.")
        st.stop()

    st.subheader("Preview")
    st.dataframe(df_raw.head(25), use_container_width=True)

    cols = list(df_raw.columns)

    # Auto-detect mappings
    auto = {
        "tenant_name": best_match_column(cols, ["tenant name", "tenant"], 70),
        "carrier_name": best_match_column(cols, ["carrier name", "carrier"], 70),
        "scac": best_match_column(cols, ["scac"], 85),
        "tracked": best_match_column(cols, ["tracked", "is tracked", "tracking"], 70),

        "pickup_departure": best_match_column(cols, [
            "pickup departure utc timestamp raw",
            "pickup departure timestamp",
            "pickup departure",
            "pickup departed",
            "departure pickup",
        ], 60),

        "dropoff_arrival": best_match_column(cols, [
            "drop off arrival utc timestamp raw",
            "drop-off arrival utc timestamp raw",
            "dropoff arrival timestamp",
            "drop off arrival",
            "dropoff arrival",
            "delivery arrival",
            "arrival dropoff",
        ], 60),

        "bill_of_lading": best_match_column(cols, [
            "bill of lading", "bol", "b/l", "bill lading"
        ], 60),

        "pickup_name": best_match_column(cols, ["pickup name", "origin name", "shipper name"], 60),
        "pickup_city_state": best_match_column(cols, ["pickup city state", "origin city state", "pickup city-state"], 55),
        "pickup_country": best_match_column(cols, ["pickup country", "origin country"], 60),

        "dropoff_name": best_match_column(cols, ["drop-off name", "drop off name", "destination name", "consignee name"], 55),
        "dropoff_city_state": best_match_column(cols, ["drop-off city state", "drop off city state", "destination city state"], 55),
        "dropoff_country": best_match_column(cols, ["drop-off country", "drop off country", "destination country"], 60),
    }

    st.subheader("Column Mapping (auto-detected, but editable)")
    st.caption("If the file headers change, adjust these dropdowns. The app will still work.")

    with st.expander("Open/adjust column mapping", expanded=True):
        options = ["(None)"] + cols

        def pick(label, key):
            default = auto.get(key)
            default_idx = options.index(default) if default in options else 0
            sel = st.selectbox(label, options=options, index=default_idx, key=f"map_{key}")
            return None if sel == "(None)" else sel

        colmap = {
            "tenant_name": pick("Tenant name column", "tenant_name"),
            "carrier_name": pick("Carrier name column", "carrier_name"),
            "scac": pick("SCAC column", "scac"),
            "tracked": pick("Tracked (true/false) column", "tracked"),
            "pickup_departure": pick("Pickup departure timestamp column", "pickup_departure"),
            "dropoff_arrival": pick("Drop-off arrival timestamp column", "dropoff_arrival"),
            "bill_of_lading": pick("Bill of lading column", "bill_of_lading"),
            "pickup_name": pick("Pickup name column", "pickup_name"),
            "pickup_city_state": pick("Pickup city-state column (e.g., Baytown-TX)", "pickup_city_state"),
            "pickup_country": pick("Pickup country column", "pickup_country"),
            "dropoff_name": pick("Drop-off name column", "dropoff_name"),
            "dropoff_city_state": pick("Drop-off city-state column (e.g., Richmond-CA)", "dropoff_city_state"),
            "dropoff_country": pick("Drop-off country column", "dropoff_country"),
        }

    # Validate minimum required fields
    missing_required = []
    for req in ["tracked", "pickup_departure", "dropoff_arrival", "carrier_name", "tenant_name", "scac",
                "bill_of_lading", "pickup_name", "pickup_city_state", "pickup_country",
                "dropoff_name", "dropoff_city_state", "dropoff_country"]:
        if not colmap.get(req):
            missing_required.append(req)

    if missing_required:
        st.warning(
            "Please map these required fields before generating the report:\n\n- "
            + "\n- ".join(missing_required)
        )
        st.stop()

    if st.button("Generate report"):
        try:
            report = build_report(df_raw, colmap)

            tracked_count = int(report["tracked_count"]["value"].iloc[0])
            untracked_count = int(report["untracked_count"]["value"].iloc[0])
            missed_count = int(report["missed_count"]["value"].iloc[0])
            total_count = int(report["total_count"]["value"].iloc[0])

            st.success("Report built!")

            # Show quick summary
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tracked", tracked_count)
            c2.metric("Missed milestone", missed_count)
            c3.metric("Untracked", untracked_count)
            c4.metric("Grand total", total_count)

            summary_detail = report["summary_detail"]
            carrier_perf = report["carrier_performance"]
            lane_perf = report["lane_performance"]

            with st.expander("Preview: Shipment detail (valid tracked)", expanded=False):
                st.dataframe(summary_detail.head(50), use_container_width=True)

            with st.expander("Preview: Carrier Performance", expanded=False):
                st.dataframe(carrier_perf.head(50), use_container_width=True)

            with st.expander("Preview: Lane Performance", expanded=False):
                st.dataframe(lane_perf.head(100), use_container_width=True)

            # Build excel bytes
            excel_bytes = build_excel_bytes(
                df_raw=df_raw,
                summary_counts={
                    "tracked": tracked_count,
                    "missed": missed_count,
                    "untracked": untracked_count,
                    "total": total_count,
                },
                summary_detail=summary_detail,
                carrier_perf=carrier_perf,
                lane_perf=lane_perf
            )

            st.download_button(
                label="Download Excel report (.xlsx)",
                data=excel_bytes,
                file_name="transit_time_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # CSV ZIP
            zip_bytes = build_csv_zip_bytes({
                "summary_detail": summary_detail,
                "carrier_performance": carrier_perf,
                "lane_performance": lane_perf,
            })
            st.download_button(
                label="Download CSVs (ZIP)",
                data=zip_bytes,
                file_name="transit_time_report_csvs.zip",
                mime="application/zip",
            )

        except Exception as e:
            st.error(f"Error generating report: {e}")
            st.stop()
else:
    st.info("Upload a CSV/XLSX to begin.")
