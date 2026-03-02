import io
import re
import math
import zipfile
from typing import Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Helpers: column normalization + matching (NO rapidfuzz)
# -----------------------------
def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[\s\-_\/]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def best_match_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    """
    Streamlit-Cloud-safe matching without external deps.
    Priority:
      1) exact normalized match
      2) pattern contained in column name
      3) all words of pattern appear in column name
      4) any word overlap (fallback)
    """
    cols_norm = {c: _normalize(c) for c in columns}

    # 1) exact normalized match
    pat_norms = [_normalize(p) for p in patterns]
    for p in pat_norms:
        for col, cn in cols_norm.items():
            if cn == p:
                return col

    # 2) contains match
    for p in pat_norms:
        for col, cn in cols_norm.items():
            if p and p in cn:
                return col

    # 3) all words in pattern appear in column
    for p in pat_norms:
        words = [w for w in p.split() if w]
        if not words:
            continue
        for col, cn in cols_norm.items():
            if all(w in cn for w in words):
                return col

    # 4) any word overlap fallback
    for p in pat_norms:
        words = [w for w in p.split() if w]
        if not words:
            continue
        for col, cn in cols_norm.items():
            if any(w in cn for w in words):
                return col

    return None


def to_bool_series(s: pd.Series) -> pd.Series:
    """Convert common true/false representations into booleans."""
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
    Parse datetimes and treat as UTC.
    If values are naive, we assume they are UTC.
    """
    return pd.to_datetime(s, errors="coerce", utc=True)


def split_city_state(val) -> Tuple[str, str]:
    if pd.isna(val):
        return "", ""
    txt = str(val).strip()
    if not txt:
        return "", ""
    # Split on last hyphen to keep cities with hyphens intact
    if "-" in txt:
        left, right = txt.rsplit("-", 1)
        return left.strip(), right.strip()
    return txt, ""


def round_days_from_hours(hours: float) -> float:
    """
    Rounding logic agreed:
    - days_raw = hours / 24
    - if 0 < days_raw < 0.5 => 0.5
    - else round half-up to nearest integer
    - no decimals except special 0.5
    """
    if hours is None or (isinstance(hours, float) and math.isnan(hours)):
        return 0
    if hours <= 0:
        return 0
    days_raw = hours / 24.0
    if 0 < days_raw < 0.5:
        return 0.5
    return float(math.floor(days_raw + 0.5))


def fmt_hours(x: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return float(round(float(x), 2))


def safe_unique_join(values: List[str], limit: int = 25) -> str:
    """Join unique strings, truncating if too many."""
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
# Core report computation
# -----------------------------
def build_report(df_raw: pd.DataFrame, colmap: Dict[str, str], lane_arrow: str = " → ") -> Dict[str, object]:
    df = df_raw.copy()

    # Mapped columns
    c_tracked = colmap["tracked"]
    c_pickup_dt = colmap["pickup_departure"]
    c_dropoff_dt = colmap["dropoff_arrival"]

    c_bol = colmap["bill_of_lading"]
    c_carrier = colmap["carrier_name"]
    c_tenant = colmap["tenant_name"]
    c_scac = colmap["scac"]

    c_pu_name = colmap["pickup_name"]
    c_pu_citystate = colmap["pickup_city_state"]
    c_pu_country = colmap["pickup_country"]

    c_do_name = colmap["dropoff_name"]
    c_do_citystate = colmap["dropoff_city_state"]
    c_do_country = colmap["dropoff_country"]

    # Tracked + timestamps
    tracked = to_bool_series(df[c_tracked])
    pu_dt = parse_dt_utc(df[c_pickup_dt])
    do_dt = parse_dt_utc(df[c_dropoff_dt])

    transit_hours = (do_dt - pu_dt).dt.total_seconds() / 3600.0

    # Missed milestone: only for tracked shipments
    missed_milestone = tracked & (pu_dt.isna() | do_dt.isna() | (transit_hours < 0))
    valid_tracked = tracked & (~missed_milestone)

    tracked_count = int(tracked.sum())
    untracked_count = int((~tracked).sum())
    missed_count = int(missed_milestone.sum())
    total_count = int(len(df))

    # Split pickup/dropoff city-state
    pu_city, pu_state = zip(*df[c_pu_citystate].map(split_city_state).tolist())
    do_city, do_state = zip(*df[c_do_citystate].map(split_city_state).tolist())

    # Shipment detail table (valid tracked only)
    detail = pd.DataFrame({
        "Bill of lading": df[c_bol],
        "Carrier name": df[c_carrier],
        "Pickup name": df[c_pu_name],
        "Pickup city": list(pu_city),
        "Pickup state": list(pu_state),
        "Pickup country": df[c_pu_country],
        "Drop-off name": df[c_do_name],
        "Drop-off city": list(do_city),
        "Drop-off state": list(do_state),
        "Drop-off country": df[c_do_country],
        "Transit time (hours)": transit_hours.map(fmt_hours),
        "Transit time (days)": transit_hours.map(round_days_from_hours),
    })
    summary_detail = detail[valid_tracked].reset_index(drop=True)

    # Lane series (for performance sheets) - city only
    lane_series = (pd.Series(list(pu_city)).fillna("") + lane_arrow + pd.Series(list(do_city)).fillna("")).map(str.strip)

    # Build a working df for performance calculations (valid tracked only)
    df_valid = pd.DataFrame({
        "_tenant": df[c_tenant].fillna(""),
        "_carrier": df[c_carrier].fillna(""),
        "_scac": df[c_scac].fillna(""),
        "_lane": lane_series,
        "_transit_hours": transit_hours,
    })[valid_tracked].copy()

    # -----------------------------
    # Carrier Performance
    # -----------------------------
    carrier_cols = [
        "Tenant name", "Carrier name", "Carrier SCAC", "Shipment volume", "Lanes",
        "Total transit time (hours)", "Total transit time (days)",
        "Average transit time (hours)", "Average transit time (days)",
        "Median transit time (hours)", "Median transit time (days)",
        "Maximum transit time (hours)", "Maximum transit time (days)",
    ]

    if df_valid.empty:
        carrier_perf = pd.DataFrame(columns=carrier_cols)
    else:
        rows = []
        grp = df_valid.groupby(["_tenant", "_carrier"], dropna=False)
        for (tenant, carrier), g in grp:
            hrs = g["_transit_hours"].dropna().astype(float).tolist()
            if not hrs:
                continue

            shipment_volume = int(len(hrs))
            total_hours = float(sum(hrs))
            avg_hours = float(pd.Series(hrs).mean())
            med_hours = float(pd.Series(hrs).median())
            max_hours = float(max(hrs))

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

        carrier_perf = pd.DataFrame(rows, columns=carrier_cols).sort_values(
            ["Tenant name", "Shipment volume", "Carrier name"],
            ascending=[True, False, True],
            kind="mergesort"
        ).reset_index(drop=True)

    # -----------------------------
    # Lane Performance (grouped visual)
    # -----------------------------
    lane_cols = [
        "Tenant name", "Lane", "Carrier name", "Carrier SCAC", "Shipment volume",
        "Total transit time (hours)", "Total transit time (days)",
        "Average transit time (hours)", "Average transit time (days)",
        "Median transit time (hours)", "Median transit time (days)",
        "Maximum transit time (hours)", "Maximum transit time (days)",
    ]

    if df_valid.empty:
        lane_perf = pd.DataFrame(columns=lane_cols)
    else:
        lane_rows = []

        # lane summary ordering
        lane_summary = df_valid.groupby(["_tenant", "_lane"], dropna=False).size().reset_index(name="_vol")
        lane_summary = lane_summary.sort_values(["_tenant", "_vol", "_lane"], ascending=[True, False, True], kind="mergesort")

        for _, r in lane_summary.iterrows():
            tenant = r["_tenant"]
            lane = r["_lane"]
            g_lane = df_valid[(df_valid["_tenant"] == tenant) & (df_valid["_lane"] == lane)].copy()

            # Header row for lane
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

            # Carrier rows ordered by volume desc
            carr_order = g_lane.groupby("_carrier").size().reset_index(name="_v").sort_values(
                ["_v", "_carrier"], ascending=[False, True], kind="mergesort"
            )["_carrier"].tolist()

            for carrier in carr_order:
                gc = g_lane[g_lane["_carrier"] == carrier]
                hrs = gc["_transit_hours"].dropna().astype(float).tolist()
                if not hrs:
                    continue

                shipment_volume = int(len(hrs))
                total_hours = float(sum(hrs))
                avg_hours = float(pd.Series(hrs).mean())
                med_hours = float(pd.Series(hrs).median())
                max_hours = float(max(hrs))

                scac_vals = gc["_scac"].dropna().astype(str).map(str.strip)
                scac_vals = scac_vals[scac_vals != ""]
                scac = scac_vals.value_counts().index[0] if len(scac_vals) else ""

                lane_rows.append({
                    "Tenant name": "",
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
        "counts": {
            "tracked": tracked_count,
            "missed": missed_count,
            "untracked": untracked_count,
            "total": total_count,
        },
        "summary_detail": summary_detail,
        "carrier_performance": carrier_perf,
        "lane_performance": lane_perf,
    }


def build_excel_bytes(
    df_raw: pd.DataFrame,
    counts: Dict[str, int],
    summary_detail: pd.DataFrame,
    carrier_perf: pd.DataFrame,
    lane_perf: pd.DataFrame
) -> bytes:
    """
    Build formatted XLSX with:
      - Raw Data
      - Summary (A1:B5 + detail starting row 7)
      - Carrier Performance
      - Lane Performance
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Raw Data
        df_raw.to_excel(writer, sheet_name="Raw Data", index=False)

        workbook = writer.book

        # Summary sheet with custom layout
        ws = workbook.add_worksheet("Summary")
        writer.sheets["Summary"] = ws

        fmt_header = workbook.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
        fmt_label = workbook.add_format({"bold": True, "border": 1})
        fmt_num = workbook.add_format({"border": 1, "num_format": "0"})
        fmt_title = workbook.add_format({"bold": True})
        fmt_hours = workbook.add_format({"border": 1, "num_format": "0.00"})
        fmt_cell = workbook.add_format({"border": 1})

        # Summary table A1:B5
        ws.write(0, 0, "Label", fmt_header)
        ws.write(0, 1, "Shipment count", fmt_header)

        rows = [
            ("Tracked", counts["tracked"]),
            ("Missed milestone", counts["missed"]),
            ("Untracked", counts["untracked"]),
            ("Grand total", counts["total"]),
        ]
        for i, (lab, val) in enumerate(rows, start=1):
            ws.write(i, 0, lab, fmt_label)
            ws.write(i, 1, val, fmt_num)

        ws.set_column(0, 0, 22)
        ws.set_column(1, 1, 16)

        # Detail title row (Row 6 visually), then header at Row 7
        startrow = 6
        ws.write(startrow - 1, 0, "Shipment-level detail (tracked & valid)", fmt_title)

        # Write detail DF
        summary_detail.to_excel(writer, sheet_name="Summary", index=False, startrow=startrow)

        # Format detail header row
        for col_idx, col_name in enumerate(summary_detail.columns):
            ws.write(startrow, col_idx, col_name, fmt_header)

        # Column widths
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

        # Number formats
        if "Transit time (hours)" in summary_detail.columns:
            h_idx = list(summary_detail.columns).index("Transit time (hours)")
            ws.set_column(h_idx, h_idx, widths.get("Transit time (hours)", 18), fmt_hours)
        if "Transit time (days)" in summary_detail.columns:
            d_idx = list(summary_detail.columns).index("Transit time (days)")
            ws.set_column(d_idx, d_idx, widths.get("Transit time (days)", 18), fmt_cell)

        # Carrier Performance
        carrier_perf.to_excel(writer, sheet_name="Carrier Performance", index=False)
        ws_cp = writer.sheets["Carrier Performance"]
        for c, name in enumerate(carrier_perf.columns):
            ws_cp.write(0, c, name, fmt_header)
        ws_cp.set_column(0, 0, 18)
        ws_cp.set_column(1, 1, 24)
        ws_cp.set_column(2, 2, 14)
        ws_cp.set_column(3, 3, 16)
        ws_cp.set_column(4, 4, 55)
        ws_cp.set_column(5, max(5, len(carrier_perf.columns) - 1), 22)

        # Lane Performance
        lane_perf.to_excel(writer, sheet_name="Lane Performance", index=False)
        ws_lp = writer.sheets["Lane Performance"]
        for c, name in enumerate(lane_perf.columns):
            ws_lp.write(0, c, name, fmt_header)
        ws_lp.set_column(0, 0, 18)
        ws_lp.set_column(1, 1, 30)
        ws_lp.set_column(2, 2, 24)
        ws_lp.set_column(3, 3, 14)
        ws_lp.set_column(4, max(4, len(lane_perf.columns) - 1), 22)

    return output.getvalue()


def build_csv_zip_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in sheets.items():
            z.writestr(f"{name}.csv", df.to_csv(index=False).encode("utf-8"))
    return buf.getvalue()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Transit Time Report Builder", layout="wide")
st.title("Transit Time Report Builder (CSV/XLSX → Excel report)")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if not uploaded:
    st.info("Upload a CSV/XLSX to begin.")
    st.stop()

# Read file
try:
    if uploaded.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded, engine="openpyxl")
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if df_raw is None or df_raw.empty:
    st.warning("Uploaded file is empty.")
    st.stop()

st.subheader("Preview (first 25 rows)")
st.dataframe(df_raw.head(25), use_container_width=True)

cols = list(df_raw.columns)

# Auto-detect column mapping
auto = {
    "tenant_name": best_match_column(cols, ["tenant name", "tenant"]),
    "carrier_name": best_match_column(cols, ["carrier name", "carrier"]),
    "scac": best_match_column(cols, ["scac"]),
    "tracked": best_match_column(cols, ["tracked", "is tracked", "tracking"]),

    "pickup_departure": best_match_column(cols, [
        "pickup departure utc timestamp raw",
        "pickup departure timestamp",
        "pickup departure",
        "pickup departed",
        "origin departure",
    ]),

    "dropoff_arrival": best_match_column(cols, [
        "drop off arrival utc timestamp raw",
        "drop-off arrival utc timestamp raw",
        "dropoff arrival timestamp",
        "drop off arrival",
        "dropoff arrival",
        "destination arrival",
        "delivery arrival",
    ]),

    "bill_of_lading": best_match_column(cols, ["bill of lading", "bol", "b/l", "bill lading"]),

    "pickup_name": best_match_column(cols, ["pickup name", "origin name", "shipper name"]),
    "pickup_city_state": best_match_column(cols, ["pickup city state", "origin city state", "pickup city-state"]),
    "pickup_country": best_match_column(cols, ["pickup country", "origin country"]),

    "dropoff_name": best_match_column(cols, ["drop-off name", "drop off name", "destination name", "consignee name"]),
    "dropoff_city_state": best_match_column(cols, ["drop-off city state", "drop off city state", "destination city state"]),
    "dropoff_country": best_match_column(cols, ["drop-off country", "drop off country", "destination country"]),
}

st.subheader("Column Mapping (auto-detected, editable)")
st.caption("If headers change, adjust these dropdowns. Required fields must be mapped to generate output.")

options = ["(None)"] + cols

def pick(label: str, key: str) -> Optional[str]:
    default = auto.get(key)
    default_idx = options.index(default) if default in options else 0
    sel = st.selectbox(label, options=options, index=default_idx, key=f"map_{key}")
    return None if sel == "(None)" else sel

with st.expander("Open/adjust column mapping", expanded=True):
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

required_keys = [
    "tenant_name", "carrier_name", "scac", "tracked",
    "pickup_departure", "dropoff_arrival",
    "bill_of_lading",
    "pickup_name", "pickup_city_state", "pickup_country",
    "dropoff_name", "dropoff_city_state", "dropoff_country",
]
missing = [k for k in required_keys if not colmap.get(k)]

if missing:
    st.warning("Map these required fields to continue:\n\n- " + "\n- ".join(missing))
    st.stop()

lane_arrow = st.text_input("Lane arrow symbol", value=" → ")

# Generate
if st.button("Generate report"):
    try:
        report = build_report(df_raw, colmap, lane_arrow=lane_arrow)
        counts = report["counts"]
        summary_detail = report["summary_detail"]
        carrier_perf = report["carrier_performance"]
        lane_perf = report["lane_performance"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tracked", counts["tracked"])
        c2.metric("Missed milestone", counts["missed"])
        c3.metric("Untracked", counts["untracked"])
        c4.metric("Grand total", counts["total"])

        with st.expander("Preview: Shipment detail (valid tracked)", expanded=False):
            st.dataframe(summary_detail.head(50), use_container_width=True)

        with st.expander("Preview: Carrier Performance", expanded=False):
            st.dataframe(carrier_perf.head(50), use_container_width=True)

        with st.expander("Preview: Lane Performance", expanded=False):
            st.dataframe(lane_perf.head(120), use_container_width=True)

        excel_bytes = build_excel_bytes(
            df_raw=df_raw,
            counts=counts,
            summary_detail=summary_detail,
            carrier_perf=carrier_perf,
            lane_perf=lane_perf,
        )

        st.download_button(
            "Download Excel report (.xlsx)",
            data=excel_bytes,
            file_name="transit_time_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        zip_bytes = build_csv_zip_bytes({
            "summary_detail": summary_detail,
            "carrier_performance": carrier_perf,
            "lane_performance": lane_perf,
        })
        st.download_button(
            "Download CSVs (ZIP)",
            data=zip_bytes,
            file_name="transit_time_report_csvs.zip",
            mime="application/zip",
        )

        st.success("Done.")

    except Exception as e:
        st.error(f"Error generating report: {e}")
