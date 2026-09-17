"""
Data loading + light preprocessing for the PHC Case Management dashboard.

IMPORTANT (Streamlit Cloud): build the .cache.parquet file locally with
build_cache.py and commit it to the repo BEFORE deploying/pushing. Every
deploy is a fresh container, so if no parquet is committed, the deployed
app is forced to parse the full raw .xlsx with openpyxl on every cold
start - which is memory-heavy enough on a 765K-row file to get the app
OOM-killed before it ever finishes loading. With a committed parquet
present, load_data() skips the raw-file parse entirely.

MEMORY NOTE: this file used to also build four full-length Python-list
columns (Judges_List, Petitioner_Advocate_List, Respondent_Advocate_List,
All_Advocates_List) and store them permanently on the 765K-row DataFrame.
A Python list object (plus a str object per name inside it) carries heavy
per-object overhead - across 4 columns x 765K rows that was very likely
400-500MB all on its own, which is what kept OOM-killing the app on
Streamlit Cloud's ~1GB cap even after the parquet-cache and cache_resource
fixes (those fixed load *speed*, not the standing memory footprint of the
loaded DataFrame). Those list columns are gone now. Name lookups/filters
are computed on demand instead - see build_name_indexes(), judge_mask(),
advocate_mask() below.
"""
import os
import re
import pandas as pd
import streamlit as st

MONTH_NUM = {
    'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4, 'MAY': 5, 'JUNE': 6,
    'JULY': 7, 'AUGUST': 8, 'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12,
}


def _split_names(raw, sep_pattern=r'&|,|/| and '):
    """Split a 'A & B & C' style string into a clean list of individual names."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = re.split(sep_pattern, raw, flags=re.IGNORECASE)
    return [p.strip(' .,') for p in parts if p.strip(' .,')]


@st.cache_resource(show_spinner="Loading case data (first run only - this is cached after)...")
def load_data(source_path: str) -> pd.DataFrame:
    parquet_path = os.path.splitext(source_path)[0] + ".cache.parquet"

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        ext = os.path.splitext(source_path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(source_path, dtype=str, encoding="utf-8-sig")
        else:
            df = pd.read_excel(source_path, dtype=str)
        df = _prepare(df)
        try:
            df.to_parquet(parquet_path)
        except Exception:
            pass  # caching is a nice-to-have, not required

    return df


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].fillna("")

    year_num = pd.to_numeric(df["Year"].str.strip(), errors="coerce")
    month_num = df["Month"].str.strip().str.upper().map(MONTH_NUM)
    day_num = pd.to_numeric(df["Date"].str.strip(), errors="coerce")

    primary_date = pd.to_datetime(
        pd.DataFrame({"year": year_num, "month": month_num, "day": day_num}),
        errors="coerce",
    )

    src_extract = df["Source_File"].astype(str).str.extract(r'(\d{4})_(\d{2})_(\d{2})_')
    fallback_date = pd.to_datetime(
        pd.DataFrame({
            "year": pd.to_numeric(src_extract[0], errors="coerce"),
            "month": pd.to_numeric(src_extract[1], errors="coerce"),
            "day": pd.to_numeric(src_extract[2], errors="coerce"),
        }),
        errors="coerce",
    )

    df["Hearing_Date"] = primary_date.fillna(fallback_date)

    # Date, raw Month, Source_File, System_ID, and Page are either only
    # needed to build Hearing_Date above (Date, Month, Source_File) or
    # aren't referenced anywhere in app.py at all (System_ID, Page).
    # Confirmed via grep across app.py before removing - none of the five
    # are read again after this point. Dropping them here keeps ~62MB of
    # columns (measured on the real 765K-row dataset) off the DataFrame for
    # the rest of its cached lifetime, since it lives in memory for as long
    # as the Streamlit process does.
    df = df.drop(columns=[c for c in
                           ["Date", "Month", "Source_File", "System_ID", "Page"]
                           if c in df.columns])

    # NOTE: Judges_List / Petitioner_Advocate_List / Respondent_Advocate_List /
    # All_Advocates_List used to be built and stored here as full-length
    # Python-list columns. They're gone - see the module docstring. The raw
    # Judges / Petitioner_Advocate / Respondent_Advocates string columns are
    # kept as-is; use build_name_indexes()/judge_mask()/advocate_mask()
    # below wherever the app used to reach for a _List column.

    DAY_NORM = {
        "MON": "MONDAY", "MONDAY": "MONDAY",
        "TUE": "TUESDAY", "TUESDAY": "TUESDAY",
        "WED": "WEDNESDAY", "WEDNESDAY": "WEDNESDAY",
        "THU": "THURSDAY", "THURSDAY": "THURSDAY",
        "FRI": "FRIDAY", "FRIDAY": "FRIDAY",
        "SAT": "SATURDAY", "SATURDAY": "SATURDAY",
        "SUN": "SUNDAY", "SUNDAY": "SUNDAY",
    }
    df["Day_Norm"] = df["Day"].str.upper().str.strip().map(DAY_NORM).fillna("")

    # Downcast the handful of low-cardinality, highly repetitive text
    # columns to `category` dtype - a meaningful memory cut on a dataset
    # this size, with no behavior change for equality/groupby/value_counts.
    for col in ["Court_No", "Case_Category", "Day_Norm", "Year", "Needs_Review", "Section"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Judges / advocate / case-title text is a different situation: it's
    # only worth converting to category if values genuinely repeat a lot
    # across the 765K rows (e.g. the same judge or advocate appearing on
    # many cases). If a column is closer to unique-per-row, category dtype
    # adds overhead (a categories array PLUS an integer code per row) on
    # top of what object dtype already costs, instead of saving anything.
    # Rather than guess, measure the actual repeat ratio at load time and
    # only convert columns that are genuinely repetitive (<50% unique).
    for col in ["Judges", "Petitioner_Advocate", "Respondent_Advocates", "Case_Title"]:
        if col in df.columns and len(df) > 0 and df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category")

    return df


@st.cache_resource(show_spinner=False)
def build_name_indexes(_df: pd.DataFrame) -> dict:
    """
    Precompute small, cheap-to-hold lookups derived from the raw Judges /
    Petitioner_Advocate / Respondent_Advocates string columns, once, without
    ever materializing a full per-row Python-list column across all 765K
    rows (see module docstring for why that mattered).

    The leading underscore on `_df` tells Streamlit not to hash the
    DataFrame itself when deciding whether to reuse the cache.

    Returns a dict with:
      - judge_names:     sorted list of unique judge names
      - judge_counts:    pd.Series, judge name -> case count (whole dataset)
      - advocate_names:  sorted list of unique advocate names (both sides)
      - advocate_counts: pd.Series, advocate name -> case count (whole dataset)
    """
    judge_counter: dict = {}
    advocate_counter: dict = {}

    for judges_raw, pet_raw, resp_raw in zip(
        _df["Judges"], _df["Petitioner_Advocate"], _df["Respondent_Advocates"]
    ):
        for j in _split_names(judges_raw, sep_pattern=r'&'):
            judge_counter[j] = judge_counter.get(j, 0) + 1
        for a in _split_names(pet_raw) + _split_names(resp_raw):
            advocate_counter[a] = advocate_counter.get(a, 0) + 1

    judge_counts = pd.Series(judge_counter, dtype="int64").sort_values(ascending=False)
    advocate_counts = pd.Series(advocate_counter, dtype="int64").sort_values(ascending=False)

    return {
        "judge_names": sorted(judge_counter.keys()),
        "judge_counts": judge_counts,
        "advocate_names": sorted(advocate_counter.keys()),
        "advocate_counts": advocate_counts,
    }


def judge_mask(judges_series: pd.Series, name: str) -> pd.Series:
    """Boolean mask over a Judges string column: True where `name` is one
    of the judges on the bench for that row. Computed on demand (no stored
    list column) - equivalent to the old `Judges_List.apply(lambda lst:
    name in lst)`, just without ever holding the split-out lists in memory
    beyond this one call."""
    return judges_series.apply(lambda x: name in _split_names(x, sep_pattern=r'&'))


@st.cache_data(show_spinner=False, max_entries=30, ttl=1800)
def filter_by_judge(_df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Cached wrapper around judge_mask() for the full 765K-row df.

    Streamlit reruns the ENTIRE script top-to-bottom on every widget
    interaction anywhere in the app - st.tabs() does not lazily skip
    hidden tabs' code, so every tab's body runs on every rerun regardless
    of which tab is visible. Without caching, this row-wise regex-split
    .apply() over the full dataset was re-running on every single
    interaction anywhere in the app (not just when the judge picker
    itself changed), repeatedly allocating fresh temporary Python
    objects - the likely cause of the app getting OOM-killed a few
    minutes into a session. Cached on `name` so it only recomputes when
    the actual judge selection changes. Leading underscore on `_df`
    tells Streamlit not to hash the DataFrame itself (it's already
    stable/cached via load_data's cache_resource).

    max_entries=30 / ttl=1800 (30 min): st.cache_data has NO size limit or
    expiry by default, and this cache is process-wide - shared across every
    viewer's session, not per-user. With ~150+ distinct judges in the data,
    a session (or several concurrent viewers) clicking through many
    different judges would otherwise pile up that many full filtered
    DataFrame slices in memory with no eviction, which on its own is
    enough to exceed Streamlit Cloud's ~1GB cap after enough distinct
    judges get selected - this was the most likely cause of the app
    crashing after some minutes of active use rather than on cold start.
    max_entries makes Streamlit evict the least-recently-used entry once
    the 31st distinct judge is selected; ttl also expires any entry after
    30 minutes regardless, so idle/stale slices don't linger forever
    either. Tune max_entries up/down to trade off memory vs. how many
    distinct judges stay "hot" (instant, no recompute) at once."""
    return _df[judge_mask(_df["Judges"], name)]


def advocate_mask(pet_series: pd.Series, resp_series: pd.Series, name: str) -> pd.Series:
    """Boolean mask: True where `name` appears as a petitioner or
    respondent advocate on that row. Equivalent to the old
    `All_Advocates_List.apply(lambda lst: name in lst)`."""
    return pd.Series(
        [name in (_split_names(p) + _split_names(r)) for p, r in zip(pet_series, resp_series)],
        index=pet_series.index,
    )


def top_names(series_of_raw, sep_pattern=r'&|,|/| and ', top_n=10) -> pd.Series:
    """Value-counts of individual names extracted from a (typically already
    filtered, so small) Series of raw 'A & B & C' style strings. Intended
    for use on a filtered `fdf`, not the full 765K-row `df` - splitting on
    the fly is cheap at the size of a single court's or judge's cases."""
    counter: dict = {}
    for raw in series_of_raw:
        for name in _split_names(raw, sep_pattern=sep_pattern):
            counter[name] = counter.get(name, 0) + 1
    if not counter:
        return pd.Series(dtype="int64")
    return pd.Series(counter, dtype="int64").sort_values(ascending=False).head(top_n)


def unique_names(series_of_raw, sep_pattern=r'&|,|/| and ') -> set:
    """Flatten a (typically filtered/small) Series of raw 'A & B & C' style
    strings into a set of unique individual names."""
    out = set()
    for raw in series_of_raw:
        out.update(_split_names(raw, sep_pattern=sep_pattern))
    return out


def unique_flat(series_of_lists) -> set:
    """Flatten a pandas Series of lists into one set of unique values.
    Kept for backward compatibility with any lingering callers that still
    pass pre-split list columns."""
    out = set()
    for lst in series_of_lists:
        out.update(lst)
    return out
