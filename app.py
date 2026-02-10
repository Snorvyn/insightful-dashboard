import os
import requests
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta

# -----------------------------
# Config
# -----------------------------
load_dotenv()

BASE = os.getenv("INSIGHTFUL_BASE_URL", "https://app.insightful.io/api/v1").rstrip("/")
TOKEN = os.getenv("INSIGHTFUL_TOKEN", "")

PROJECT_ID = "wl4jbkqdtqt_dzl"  # AI Data Engineers
TZ_NAME = "Asia/Yerevan"
TZ = ZoneInfo(TZ_NAME)

headers = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}

# LIGHT MODE
st.set_page_config(page_title="AI Data Engineers — Insightful", layout="wide")


# -----------------------------
# Simple password protection
# -----------------------------
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("### 🔒 Private dashboard")

    password = st.text_input(
        "Enter password",
        type="password",
    )

    if password:
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")

    return False


if not check_password():
    st.stop()




# -----------------------------
# Helpers
# -----------------------------
def local_start_of_day(d: date) -> datetime:
    return datetime.combine(d, time(0, 0, 0), tzinfo=TZ)

def local_end_of_day(d: date) -> datetime:
    return datetime.combine(d, time(23, 59, 59, 999000), tzinfo=TZ)

def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def now_local() -> datetime:
    return datetime.now(TZ)

def ms_to_hhmm(ms: int) -> str:
    total_minutes = int(round(ms / 60000))
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{h}h {m:02d}m"

def week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday = 0

# -----------------------------
# Presets
# -----------------------------
PRESETS = [
    "This week",
    "Today",
    "Yesterday",
    "Last 7 days",
    "Previous week",
    "This month",
    "Previous Month",
    "Last 3 Month",
    "Last 6 months",
    "This Year",
    "last 12 months",
    "Custom range",
]

def compute_range(preset: str):
    today = now_local().date()

    if preset == "Today":
        return local_start_of_day(today), local_end_of_day(today)

    if preset == "Yesterday":
        y = today - timedelta(days=1)
        return local_start_of_day(y), local_end_of_day(y)

    if preset == "This week":
        mon = week_monday(today)
        return local_start_of_day(mon), local_end_of_day(today)

    if preset == "Previous week":
        this_mon = week_monday(today)
        prev_mon = this_mon - timedelta(days=7)
        prev_sun = prev_mon + timedelta(days=6)
        return local_start_of_day(prev_mon), local_end_of_day(prev_sun)

    if preset == "Last 7 days":
        start_day = today - timedelta(days=6)
        return local_start_of_day(start_day), local_end_of_day(today)

    if preset == "This month":
        first = today.replace(day=1)
        return local_start_of_day(first), local_end_of_day(today)

    if preset == "Previous Month":
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        return local_start_of_day(first_prev), local_end_of_day(last_prev)

    if preset == "Last 3 Month":
        start_day = today + relativedelta(months=-3)
        return local_start_of_day(start_day), local_end_of_day(today)

    if preset == "Last 6 months":
        start_day = today + relativedelta(months=-6)
        return local_start_of_day(start_day), local_end_of_day(today)

    if preset == "This Year":
        first = date(today.year, 1, 1)
        return local_start_of_day(first), local_end_of_day(today)

    if preset == "last 12 months":
        start_day = today + relativedelta(months=-12)
        return local_start_of_day(start_day), local_end_of_day(today)

    raise ValueError(f"Unknown preset: {preset}")

# -----------------------------
# API (cached)
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_tasks(project_id: str):
    r = requests.get(f"{BASE}/task", headers=headers, params={"projectId": project_id}, timeout=30)
    r.raise_for_status()
    tasks = r.json()
    task_name = {t["id"]: (t.get("name", t["id"]) or t["id"]) for t in tasks}
    task_ids = set(task_name.keys())
    task_order = sorted(task_name.values(), key=lambda s: s.lower())
    return task_name, task_ids, task_order

@st.cache_data(ttl=3600)
def fetch_employees():
    r = requests.get(f"{BASE}/employee", headers=headers, timeout=30)
    r.raise_for_status()
    emps = r.json()
    emp_name = {e["id"]: (e.get("name", e["id"]) or e["id"]) for e in emps}
    return emp_name

@st.cache_data(ttl=300, show_spinner=False)
def fetch_windows_cached(start_ms: int, end_ms: int):
    r = requests.get(
        f"{BASE}/analytics/window",
        headers=headers,
        params={"start": start_ms, "end": end_ms, "projectId": PROJECT_ID, "timezone": TZ_NAME},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_manual_entries_cached(start_ms: int, end_ms: int):
    r = requests.get(
        f"{BASE}/analytics/manual-entry",
        headers=headers,
        params={"start": start_ms, "end": end_ms, "timezone": TZ_NAME},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    return data if isinstance(data, list) else []


# -----------------------------
# Data builder (Total only)
# -----------------------------
def normalize_employee_name(s: str) -> str:
    # fixes hidden “different strings” causing grouped bars
    return " ".join((s or "").split()).strip()

@st.cache_data(ttl=300, show_spinner=False)
def build_df(start_dt: datetime, end_dt: datetime):
    start_ms = dt_to_ms(start_dt)
    end_ms = dt_to_ms(end_dt)

    task_name, task_ids, task_order = fetch_tasks(PROJECT_ID)
    emp_name = fetch_employees()

    windows = fetch_windows_cached(start_ms, end_ms)
    manual_rows = fetch_manual_entries_cached(start_ms, end_ms)


    total_ms = defaultdict(int)  # (employeeName, taskName) -> ms

    for row in windows:
        emp_id = row.get("employeeId")
        task_id = row.get("taskId")
        dur = row.get("duration")
        if not emp_id or not task_id or task_id not in task_ids:
            continue
        if not isinstance(dur, (int, float)) or dur <= 0:
            continue

        e = normalize_employee_name(str(emp_name.get(emp_id, emp_id)))
        t = str(task_name.get(task_id, task_id))
        total_ms[(e, t)] += int(dur)

    for row in manual_rows:
        if row.get("projectId") != PROJECT_ID:
            continue
        emp_id = row.get("employeeId")
        task_id = row.get("taskId")
        dur = row.get("duration")
        if not emp_id or not task_id or task_id not in task_ids:
            continue
        if not isinstance(dur, (int, float)) or dur <= 0:
            continue

        e = normalize_employee_name(str(emp_name.get(emp_id, emp_id)))
        t = str(task_name.get(task_id, task_id))
        total_ms[(e, t)] += int(dur)

    rows = []
    for (e, t), ms in total_ms.items():
        if ms < 60_000:
            continue
        rows.append({"employee": e, "task": t, "ms": int(ms)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df, task_order

    df["hours"] = df["ms"] / (1000 * 60 * 60)
    df["hhmm"] = df["ms"].apply(ms_to_hhmm)

    # keep only tasks that appear
    present_tasks = sorted(df["task"].unique(), key=lambda s: s.lower())
    # prefer task_order order if matches
    ordered = [t for t in task_order if t in set(present_tasks)]
    ordered += [t for t in present_tasks if t not in set(ordered)]

    return df, ordered

# -----------------------------
# Sidebar (stable radio — FIXED)
# -----------------------------
with st.sidebar:
    st.header("Time range")

    # Initialize once
    if "preset" not in st.session_state:
        st.session_state.preset = "This week"

    # IMPORTANT:
    # - use key="preset"
    # - DO NOT assign st.session_state.preset manually
    st.radio(
        "Preset",
        PRESETS,
        key="preset",
    )

    # spacing
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    preset = st.session_state.preset

    if preset == "Custom range":
        today = now_local().date()
        default_start = week_monday(today)
        default_end = today

        dr = st.date_input(
            "Pick dates",
            value=(default_start, default_end),
        )

        if isinstance(dr, tuple) and len(dr) == 2:
            d_start, d_end = dr
        else:
            d_start, d_end = default_start, default_end

        start_dt = local_start_of_day(d_start)
        end_dt = local_end_of_day(d_end)
    else:
        start_dt, end_dt = compute_range(preset)

    st.caption(f"Timezone: {TZ_NAME}")
    st.caption(f"{start_dt.date().isoformat()} - {end_dt.date().isoformat()}")


# -----------------------------
# Build
# -----------------------------
with st.spinner("Fetching Insightful data..."):
    df, task_order = build_df(start_dt, end_dt)

if df.empty:
    st.warning("No data returned for this project/time range.")
    st.stop()

# ---- Header row: title left, download right
left, right = st.columns([0.78, 0.22], vertical_alignment="center")

with left:
    st.markdown("### AI Data Engineers — Time report")

with right:
    download_df = (
        df[["employee", "task", "hours"]]
        .copy()
        .rename(columns={"employee": "user", "hours": "time_hours"})
    )
    download_df["time_hours"] = download_df["time_hours"].round(1)

    csv_bytes = download_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥  Download CSV",
        data=csv_bytes,
        file_name=f"ai-data-engineers_{start_dt.date()}_{end_dt.date()}.csv",
        mime="text/csv",
        use_container_width=True,
    )



# order employees by total desc
employee_order = (
    df.groupby("employee")["hours"].sum().sort_values(ascending=False).index.tolist()
)

# -----------------------------
# Plotly STACKED (graph_objects, guaranteed)
# -----------------------------
# Pivot to ensure one x per employee, one series per task
pivot = (
    df.pivot_table(index="employee", columns="task", values="hours", aggfunc="sum", fill_value=0.0)
      .reindex(employee_order)
)

# Matching pivot for hh:mm tooltips (store ms too)
pivot_ms = (
    df.pivot_table(index="employee", columns="task", values="ms", aggfunc="sum", fill_value=0)
      .reindex(employee_order)
)

def ms_to_hhmm_safe(ms):
    ms = int(ms or 0)
    total_minutes = int(round(ms / 60000))
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{h}h {m:02d}m"


fig = go.Figure()

for task in task_order:
    if task not in pivot.columns:
        continue

    y_hours = pivot[task].tolist()

    # per-point hh:mm based on ms
    if task in pivot_ms.columns:
        hhmm_list = [ms_to_hhmm_safe(v) for v in pivot_ms[task].tolist()]
    else:
        hhmm_list = ["0:00"] * len(employee_order)

    fig.add_trace(
        go.Bar(
            name=task,
            x=employee_order,
            y=y_hours,
            customdata=hhmm_list,  # one hh:mm per bar segment
            hovertemplate=f"<b>{task}</b><br>%{{x}}<br>%{{customdata}}<extra></extra>",
        )
    )


fig.update_layout(
    template="plotly_dark",
    # colorway=[
    #     "#4CC9F0",  # cyan
    #     "#4361EE",  # blue
    #     "#F72585",  # pink
    #     "#7209B7",  # purple
    #     "#3A86FF",  # light blue
    # ],
    barmode="stack",
    title=dict(
        text="Total time by employee (stacked by task)",
        x=0.0,
        xanchor="left",
        font=dict(size=18),
    ),
    xaxis_title="Employee",
    yaxis_title="Hours",
    legend=dict(
        title="Tasks",
        orientation="h",
        yanchor="top",
        y=-0.18,
        xanchor="left",
        x=0,
        itemclick="toggle",
        itemdoubleclick="toggleothers",
    ),
    margin=dict(b=140),
)

# Download CSV: one row per (user, task)
download_df = (
    df[["employee", "task", "hours"]]
    .copy()
    .rename(columns={"employee": "user", "hours": "time_hours"})
)
download_df["time_hours"] = download_df["time_hours"].round(1)

csv_bytes = download_df.to_csv(index=False).encode("utf-8")




st.plotly_chart(fig, width="stretch")

# -----------------------------
# Task totals strip (smaller header, no hh:mm text)
# -----------------------------
st.markdown("#### Task totals")

task_totals = (
    df.groupby("task", as_index=False)["ms"]
      .sum()
      .sort_values("ms", ascending=False)
)
task_totals["Total"] = task_totals["ms"].apply(ms_to_hhmm)

# compact row blocks
i = 0
while i < len(task_totals):
    chunk = task_totals.iloc[i:i+6]
    cols = st.columns(len(chunk))
    for c, (_, r) in zip(cols, chunk.iterrows()):
        with c:
            st.markdown(f"**{r['task']}**")
            st.markdown(r["Total"])
    i += 6
