import pandas as pd
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import io
import os
import re
import json
from dotenv import load_dotenv
# High-contrast widgets everywhere (main area + sidebar)
# Light-blue selectboxes (main content + sidebar), readable text, no chip changes
# Light-blue selectboxes (main content + sidebar), readable text, no chip changes
st.markdown("""
<style>
/* ===== MAIN PANE SELECTBOXES ===== */
[data-testid="stAppViewContainer"] .stSelectbox div[data-baseweb="select"] > div,
[data-testid="stAppViewContainer"] .stSelectbox div[role="combobox"]{
  background: #E6F0FF !important;         /* light blue */
  color: #0E141B !important;               /* readable text */
  border: 2px solid #5B8CFF !important;    /* blue border */
  border-radius: 10px !important;
  min-height: 46px !important;
  padding: 8px 12px !important;
}

/* Make selected value clearly visible */
[data-testid="stAppViewContainer"] .stSelectbox div[role="combobox"] > div{
  color: #0E141B !important;
  font-weight: 600 !important;
}

/* Focus glow */
[data-testid="stAppViewContainer"] .stSelectbox div[role="combobox"]:focus-within{
  box-shadow: 0 0 0 3px rgba(91,140,255,.35) !important;
  outline: none !important;
}

/* ===== SIDEBAR SELECTBOXES ===== */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
[data-testid="stSidebar"] .stSelectbox div[role="combobox"]{
  background: #E6F0FF !important;
  color: #0E141B !important;
  border: 2px solid #5B8CFF !important;
  border-radius: 10px !important;
  min-height: 46px !important;
  padding: 8px 12px !important;
}
[data-testid="stSidebar"] .stSelectbox div[role="combobox"] > div{
  color: #0E141B !important;
  font-weight: 600 !important;
}
[data-testid="stSidebar"] .stSelectbox div[role="combobox"]:focus-within{
  box-shadow: 0 0 0 3px rgba(91,140,255,.35) !important;
  outline: none !important;
}

/* ===== DO NOT TOUCH HEATMAP FILTER CHIPS ===== */
.stMultiSelect [data-baseweb="tag"]{
  background: initial !important;
  color: initial !important;
  border: initial !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* --- Fix text clipping inside all Streamlit selectboxes (main + sidebar) --- */
/* Target the BaseWeb <Select> control's visible wrapper */
.stSelectbox div[data-baseweb="select"] > div{
  min-height: 56px !important;
  height: 56px !important;
  padding: 12px 16px !important;     /* space above/below text */
  display: flex !important;
  align-items: center !important;     /* vertical center */
  box-sizing: border-box !important;
}

/* The inner value container (where text lives) */
.stSelectbox div[data-baseweb="select"] > div > div{
  display: flex !important;
  align-items: center !important;
}

/* Make sure actual text nodes have enough line-height */
.stSelectbox div[data-baseweb="select"] span{
  line-height: 1.35 !important;
  padding-top: 2px !important;        /* tiny nudge if needed */
}

/* BaseWeb also uses a hidden/visible input for single value */
.stSelectbox div[data-baseweb="select"] input{
  height: 24px !important;
  line-height: 24px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* =============== Sidebar text input + textarea =============== */

/* Outer control (BaseWeb wrappers) */
section[data-testid="stSidebar"] .stTextInput div[data-baseweb="input"] > div,
section[data-testid="stSidebar"] .stTextArea  div[data-baseweb="textarea"] > div{
  background: #EAF2FF !important;               /* light blue */
  border: 2px solid #7AA6FF !important;          /* blue border */
  border-radius: 12px !important;
  box-shadow: 0 0 0 3px rgba(122,166,255,.25) inset !important;
  transition: border-color .2s ease, box-shadow .2s ease;
}

/* Actual input/textarea fields */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea  textarea{
  background: transparent !important;
  color: #0B132A !important;
 
  padding: 12px 14px !important;
  border: 0 !important;
  box-shadow: none !important;
  line-height: 24px !important;
}

/* Consistent heights */
section[data-testid="stSidebar"] .stTextInput input{
  height: 48px !important;
}
section[data-testid="stSidebar"] .stTextArea textarea{
  min-height: 120px !important;   /* adjust if you want a taller box */
}

/* Focus state */
section[data-testid="stSidebar"] .stTextInput div[data-baseweb="input"]:focus-within > div,
section[data-testid="stSidebar"] .stTextArea  div[data-baseweb="textarea"]:focus-within > div{
  border-color: #3B82F6 !important;                  /* brighter blue */
  box-shadow: 0 0 0 3px rgba(59,130,246,.35) inset !important;
}

/* (Optional) labels above the fields */
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stTextArea  label{
  margin-bottom: 6px !important;
}

/* Dark theme text color tweak */
html[data-theme="dark"] section[data-testid="stSidebar"] .stTextInput input,
html[data-theme="dark"] section[data-testid="stSidebar"] .stTextArea  textarea{
  color: #F2F6FF !important;
}
</style>
""", unsafe_allow_html=True)



# -----------------------------------------------------------
# Env & LLM setup
# -----------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.2) if api_key else None

# -----------------------------------------------------------
# Prompts (GenAI)
# -----------------------------------------------------------
feature_prompt = PromptTemplate(
    input_variables=["background", "columns", "target_column"],
    template=(
        "Given this background about the dataset: {background}, available columns: {columns}, "
        "and the target column to create: {target_column}, generate Python code as plain text to perform "
        "feature engineering. Create the target column '{target_column}' in the DataFrame 'df' using the "
        "appropriate columns based on the background. Use only the columns provided in the 'columns' list "
        "and ensure the code is valid Python syntax (e.g., df['{target_column}'] = df['col1'] - df['col2']). "
        "Return only the code without explanations or formatting."
    ),
)

forecast_prompt = PromptTemplate(
    input_variables=["task", "data", "context"],
    template=(
        "Given this task: {task}, data: {data}, and context: {context}, generate the appropriate code or insight as "
        "plain text without markdown, backticks, or additional formatting. For Prophet code, use "
        "'from prophet import Prophet', define 'model' as the Prophet instance, and 'forecast' as the prediction output, "
        "ensuring the DataFrame 'df' has 'ds' for dates and 'y' for the target column specified. For insights, provide "
        "a detailed analysis of trends, peaks, or dips in the forecast, with actionable business recommendations in a "
        "concise paragraph (3-5 sentences), avoiding code or technical jargon, and leveraging the context to tailor the insights."
    ),
)

feature_chain = RunnableSequence(feature_prompt | llm) if llm else None
forecast_chain = RunnableSequence(forecast_prompt | llm) if llm else None

# -----------------------------------------------------------
# Data loading & utilities
# -----------------------------------------------------------
def load_data(file=None, date_column='ds', filename='time_series_data.csv'):
    """Load CSV either from uploaded file-like object or from a path."""
    try:
        if file is not None:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(filename)

        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found.")

        # normalize to Prophet's 'ds'
        df = df.rename(columns={date_column: 'ds'})
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        if df['ds'].isna().any():
            raise ValueError("Some dates could not be parsed.")

        return df, df.columns.tolist()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, []


def infer_granularity(df):
    """Detect the tightest frequency: returns allowed options among D/W/M."""
    try:
        df_sorted = df[['ds']].sort_values('ds').drop_duplicates()
        time_diffs = df_sorted['ds'].diff().dropna()
        min_diff = time_diffs.min()
        min_diff_seconds = min_diff.total_seconds()
        if min_diff_seconds <= 86400:      # ~daily
            return ['D', 'W', 'M']
        elif min_diff_seconds <= 604800:   # ~weekly
            return ['W', 'M']
        else:
            return ['M']
    except Exception as e:
        st.warning(f"Could not infer granularity: {e}. Defaulting to Weekly.")
        return ['W']


def aggregate_data(df, target_column, frequency='W', group_columns=None):
    """Aggregate to D/W/M and optionally by group columns."""
    try:
        df_copy = df.copy()
        if frequency == 'D':
            df_copy['ds'] = df_copy['ds'].dt.floor('D')
        elif frequency == 'W':
            df_copy['ds'] = df_copy['ds'].dt.to_period('W').dt.to_timestamp()
        elif frequency == 'M':
            df_copy['ds'] = df_copy['ds'].dt.to_period('M').dt.to_timestamp()
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        if group_columns and group_columns[0]:
            groupby_cols = ['ds'] + group_columns
            agg_df = df_copy.groupby(groupby_cols, as_index=False)[target_column].sum()
        else:
            agg_df = df_copy.groupby('ds', as_index=False)[target_column].sum()
        return agg_df
    except Exception as e:
        st.error(f"Error aggregating data: {e}")
        return df


def engineer_features(df, target_column, background, columns):
    """Use GenAI to create a missing target column if needed (safe no-op if exists)."""
    try:
        if target_column in df.columns or not feature_chain:
            if target_column in df.columns:
                st.info(f"Target column '{target_column}' already exists.")
            return df

        feature_code = feature_chain.invoke({
            "background": background,
            "columns": ", ".join(columns),
            "target_column": target_column
        }).content

        local_vars = {'df': df.copy()}
        exec(feature_code, globals(), local_vars)
        df = local_vars['df']
        st.success(f"Created '{target_column}'.")
        return df
    except Exception as e:
        st.error(f"Error in feature engineering: {e}")
        return df


def run_forecast(df, target_column, periods, frequency, data_color, forecast_color):
    """Train simple Prophet model and return forecast + figures for one series."""
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found.")
        return None, None, None, None

    df_prophet = df[['ds', target_column]].rename(columns={target_column: 'y'})
    if len(df_prophet.dropna()) < 2:
        st.warning("Not enough data for forecasting (less than 2 non-NaN rows).")
        return None, None, None, None

    try:
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=periods, freq=frequency)
        forecast = model.predict(future)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        last_hist = df_prophet['ds'].max()
        hist = df_prophet[df_prophet['ds'] <= last_hist]
        fut = forecast[forecast['ds'] > last_hist]

        ax1.plot(hist['ds'], hist['y'], '-', color=data_color, label='Historical Data')
        ax1.plot(fut['ds'], fut['yhat'], '-', color=forecast_color, label='Forecast')
        ax1.fill_between(fut['ds'], fut['yhat_lower'], fut['yhat_upper'], color=forecast_color, alpha=0.2)
        ax1.legend()
        ax1.set_title(f'{target_column} Forecast')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(target_column)
        ax1.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        fig2 = model.plot_components(forecast, figsize=(10, 8))
        return model, forecast, fig1, fig2
    except Exception as e:
        st.error(f"Error in Prophet model: {e}")
        return None, None, None, None


def get_insights(forecast, target_column, context):
    """Summarize the last few forecast points with GenAI (safe fallback if no key)."""
    if not forecast_chain:
        return "Insights: (OpenAI key not set) Review trends, peaks and dips to adjust inventory and promotions."

    try:
        insights = forecast_chain.invoke({
            "task": "Provide a detailed business insights",
            "data": f"forecast for {target_column}: {forecast[['ds', 'yhat']].tail().to_string()}",
            "context": context
        }).content
        return insights
    except Exception as e:
        return f"Error generating insights: {e}"
# -------------------- Upgraded GenAI Insights (sectioned Markdown) --------------------
def get_genai_insights(
    forecast_df: pd.DataFrame,
    target_col: str,
    context: str,
    style: str = "detailed",
    hist_df: pd.DataFrame | None = None,   # NEW: pass history for deeper stats
) -> str:
    """
    Returns Markdown with multi-section insights using GenAI.
    If llm is unavailable, returns a numeric fallback.
    """
    try:
        if forecast_df is None or "ds" not in forecast_df or "yhat" not in forecast_df:
            return "_Insights unavailable: need forecast with columns 'ds' and 'yhat'._"

        f = forecast_df.copy()
        f["ds"] = pd.to_datetime(f["ds"])
        f = f.sort_values("ds")
        n = len(f)
        if n == 0:
            return "_Insights unavailable: empty forecast._"

        # ---------- Forecast-horizon stats ----------
        peak_row = f.loc[f["yhat"].idxmax()]
        trough_row = f.loc[f["yhat"].idxmin()]
        mean_val = float(f["yhat"].mean())
        std_val = float(f["yhat"].std(ddof=0)) if n > 1 else 0.0
        volatility = float(std_val / mean_val) if mean_val else 0.0

        x = np.arange(n, dtype=float)
        y = f["yhat"].values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0]) if n >= 2 else 0.0

        # Top upcoming 3 weeks by forecast
        top_upcoming = (
            f[["ds", "yhat"]]
            .nlargest(3, "yhat")
            .assign(ds=lambda d: d["ds"].dt.strftime("%Y-%m-%d"),
                    yhat=lambda d: d["yhat"].round(2))
            .to_dict("records")
        )

        # Seasonality proxy
        month_means = (
            f.assign(month=f["ds"].dt.month)
             .groupby("month")["yhat"].mean()
             .sort_values(ascending=False)
             .to_dict()
        )
        top_months = list(month_means.keys())[:3]
        amplitude = ((float(peak_row["yhat"]) - float(trough_row["yhat"])) / mean_val) if mean_val else 0.0

        # Confidence interval summary
        if {"yhat_lower", "yhat_upper"}.issubset(f.columns):
            f["ci_width"] = f["yhat_upper"] - f["yhat_lower"]
            avg_ci = float(f["ci_width"].mean())
            wide_ci_weeks = int((f["ci_width"] > f["ci_width"].quantile(0.75)).sum())
        else:
            avg_ci, wide_ci_weeks = None, None

        # ---------- History-based stats (if provided) ----------
        hist_stats = {}
        if hist_df is not None and target_col in hist_df.columns:
            h = hist_df.copy()
            if "ds" in h.columns:
                h["ds"] = pd.to_datetime(h["ds"])
                h = h.sort_values("ds")

            # Momentum: next 4 forecast weeks vs last 4 observed weeks
            last4_hist = float(h[target_col].tail(4).sum()) if len(h) >= 4 else None
            next4_fc = float(f["yhat"].head(4).sum()) if len(f) >= 4 else None
            momentum = None
            if last4_hist and next4_fc:
                momentum = round(((next4_fc / last4_hist) - 1) * 100.0, 2)

            # Price correlation / elasticity (if PRICE column exists)
            price_col = None
            for c in ["PRICE", "price", "Base_Price", "BASE_PRICE"]:
                if c in h.columns:
                    price_col = c
                    break

            price_corr = None
            elasticity = None
            if price_col:
                sample = h[[price_col, target_col]].dropna()
                if len(sample) >= 10 and sample[target_col].gt(0).all():
                    price_corr = float(sample.corr().iloc[0, 1])
                    # simple log-log slope
                    X = np.log(sample[price_col].values)
                    Y = np.log(sample[target_col].values)
                    if np.isfinite(X).all() and np.isfinite(Y).all():
                        elasticity = float(np.polyfit(X, Y, 1)[0])

            # Promo lifts (if binary flags exist)
            promo_cols = [c for c in ["FEATURE", "DISPLAY", "TPR_ONLY"] if c in h.columns]
            lifts = {}
            for pc in promo_cols:
                on = h.loc[h[pc] == 1, target_col].dropna()
                off = h.loc[h[pc] == 0, target_col].dropna()
                if len(on) >= 5 and len(off) >= 5 and off.mean() > 0:
                    lifts[pc] = round(((on.mean() / off.mean()) - 1) * 100.0, 2)

            hist_stats = {
                "momentum_pct_next4_vs_last4": momentum,
                "price_corr": price_corr,
                "price_elasticity_loglog": elasticity,
                "promo_lifts_pct": lifts,
            }

        # Combined stats payload
        stats = {
            "horizon_periods": n,
            "target": target_col,
            "mean": round(mean_val, 2),
            "volatility": round(volatility, 4),
            "trend_slope_per_period": round(slope, 2),
            "peak": {"date": str(peak_row["ds"].date()), "value": round(float(peak_row["yhat"]), 2)},
            "trough": {"date": str(trough_row["ds"].date()), "value": round(float(trough_row["yhat"]), 2)},
            "amplitude_peak_trough_pct_of_mean": round(amplitude * 100.0, 2) if mean_val else None,
            "top_months_by_avg": top_months,
            "top_upcoming_weeks": top_upcoming,
            "avg_ci_width": None if avg_ci is None else round(avg_ci, 2),
            "count_wide_ci_weeks": wide_ci_weeks,
            "history": hist_stats,
        }

        # ----------- Fallback (no LLM) -----------
        if llm is None:
            lines = [
                "### GenAI Insights (fallback)",
                f"- Horizon: **{stats['horizon_periods']}** periods",
                f"- Trend slope: **{stats['trend_slope_per_period']}**",
                f"- Volatility (σ/μ): **{stats['volatility']}**",
                f"- Peak: **{stats['peak']['value']}** on **{stats['peak']['date']}**",
                f"- Trough: **{stats['trough']['value']}** on **{stats['trough']['date']}**",
                f"- Top upcoming: {stats['top_upcoming_weeks']}",
            ]
            if stats["history"].get("momentum_pct_next4_vs_last4") is not None:
                lines.append(f"- Momentum (next4 vs last4): **{stats['history']['momentum_pct_next4_vs_last4']}%**")
            if stats["history"].get("price_corr") is not None:
                lines.append(f"- Price correlation: **{round(stats['history']['price_corr'],3)}**")
            if stats["history"].get("price_elasticity_loglog") is not None:
                lines.append(f"- Price elasticity (log-log): **{round(stats['history']['price_elasticity_loglog'],3)}**")
            if stats["history"].get("promo_lifts_pct"):
                lines.append(f"- Promo lifts (%): **{stats['history']['promo_lifts_pct']}**")
            return "\n".join(lines)

        style_map = {
            "brief": "tight bullet points",
            "detailed": "a short report with sections and bullet points",
            "executive": "an executive summary with 4-6 bullets",
        }
        style_words = style_map.get(style.lower(), style_map["detailed"])

        prompt = f"""
You are a demand-planning analyst for **manufacturing sales**.
Write {style_words} in **Markdown** using ONLY the JSON stats & business context below.
Be specific and numeric. Avoid generic advice. Reference at least 6 numbers and 3 dates.
If price/promo effects exist, include a section for them with concrete lifts/correlations.
List the **next 3 highest weeks** explicitly with dates and values.

Context:
{context}

JSON_STATS:
{json.dumps(stats, indent=2)}

Sections to produce:
1. **Overview** – one or two sentences with quantified trend & scale.
2. **Seasonality & Peaks** – bullets including peak/trough, amplitude(% of mean), and top upcoming weeks.
3. **Price / Promo Effects** – only if present (elasticity, correlation, lifts %).
4. **Risk & Uncertainty** – CI width, any weeks with wide CI.
5. **Actions for Ops** – 3-5 bullets (production, inventory cover, promo timing) tied to dates/numbers above.
6. **One-slide takeaway** – one sentence, quantified.
"""
        resp = llm.invoke(prompt)
        return getattr(resp, "content", resp)

    except Exception as e:
        return f"_Insights error: {e}_"

# -------------------- end upgraded insights --------------------


# --------------- Robust selection parsing (fixes your IndexError) ---------------
def parse_group_selection(selection: str, group_columns: list[str]):
    """
    Accept labels like:
      - 'UPC=1111009497' or 'Store=367 & Dept=12'
      - or just '1111009497' (no '=')
    Returns dict like {'UPC': '1111009497'}.
    """
    filters = {}
    if not selection:
        return filters
    sel = str(selection).strip()

    if '=' in sel:
        for token in sel.replace(';', '&').split('&'):
            token = token.strip()
            if '=' in token:
                k, v = token.split('=', 1)
                filters[k.strip()] = v.strip()
    else:
        if group_columns:
            filters[group_columns[0]] = sel
    return filters


def filter_by_selection(df, selection: str, group_columns: list[str]):
    out = df.copy()
    filters = parse_group_selection(selection, group_columns)
    for col, val in filters.items():
        if col in out.columns:
            out = out[out[col].astype(str) == str(val)]
    return out
# -------------------------------------------------------------------------------


def run_multi_group_forecast(
    df, group_columns, target_column, periods, frequency, context,
    data_color, forecast_color, top_n=10, filter_group=None, selected_group=None
):
    """Train & compare per-group Prophet models; persist heatmap-ready data in session."""
    group_title = filter_group if filter_group else " & ".join(group_columns)

    if filter_group:
        st.subheader(f"Analyzing {target_column} for top {top_n} {filter_group}s ({context})")
    else:
        st.subheader(f"Forecasting {target_column} across top {top_n} groups ({' & '.join(group_columns)}) from {context}")

    agg_df = aggregate_data(df, target_column, frequency, group_columns if not filter_group else [filter_group])

    if len(group_columns if not filter_group else [filter_group]) == 1:
        col = group_columns[0] if not filter_group else filter_group
        vals = agg_df[col].value_counts().nlargest(top_n).index.tolist()
        combined_groups = [(v,) for v in vals]
    else:
        sums = agg_df.groupby(group_columns)[target_column].sum().nlargest(top_n)
        combined_groups = list(sums.index)

    fig_compare, ax = plt.subplots(figsize=(12, 6))
    forecasts_dict, agg_df_dict = {}, {}

    for i, combo in enumerate(combined_groups):
        group_color = plt.cm.tab10(i % 10)
        gd = agg_df.copy()

        if len(group_columns if not filter_group else [filter_group]) == 1:
            col = group_columns[0] if not filter_group else filter_group
            gd = gd[gd[col] == combo[0]]
            group_label = str(combo[0])
        else:
            for c, v in zip(group_columns, combo):
                gd = gd[gd[c] == v]
            group_label = " & ".join([f"{c}={v}" for c, v in zip(group_columns, combo)])

        if not gd.empty and len(gd.dropna()) >= 2:
            try:
                prop = gd[['ds', target_column]].rename(columns={target_column: 'y'})
                model = Prophet()
                model.fit(prop)
                future = pd.DataFrame({
                    'ds': pd.date_range(start=prop['ds'].max(), periods=periods + 1, freq=frequency)[1:]
                })
                forecast = model.predict(future)

                last_date = prop['ds'].max()
                hist = prop[prop['ds'] <= last_date]
                fut = forecast[forecast['ds'] > last_date]

                if selected_group is None or group_label == selected_group:
                    ax.plot(hist['ds'], hist['y'], '-', color=group_color, alpha=0.5, label=f"{group_label} (Historical)")
                    ax.plot(fut['ds'], fut['yhat'], '-', color=group_color, label=f"{group_label} (Forecast)")

                forecasts_dict[group_label] = forecast
                agg_df_dict[group_label] = gd
            except Exception as e:
                st.warning(f"Could not forecast for {group_label}: {e}")
        else:
            st.warning(f"Skipping {group_label}: not enough data.")

    ax.set_title(f"{target_column} Forecast Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_column)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()

    if forecasts_dict:
        labels = list(forecasts_dict.keys())
        st.session_state[f"heatmap_data_{group_title}"] = (forecasts_dict, agg_df_dict, labels, target_column, group_title)
    else:
        st.error("No valid forecasts generated for any group.")

    return fig_compare, forecasts_dict, agg_df


def create_forecast_heatmap(forecasts_dict=None, group_labels=None, target_column=None, group_title=None, agg_df_dict=None):
    """Heatmap of per-group forecast values for the future periods."""
    if not all([forecasts_dict, group_labels, target_column, group_title]):
        st.warning("Missing data for heatmap generation.")
        return None

    st.subheader(f"Forecast Heatmap by {group_title}")

    selected_groups = st.multiselect(
        "Filter Groups for Heatmap",
        options=group_labels,
        default=group_labels[:min(10, len(group_labels))],
        key=f"heatmap_filter_{group_title}"
    )
    if not selected_groups:
        st.warning("Please select at least one group to display the heatmap.")
        return None

    all_forecasts = pd.DataFrame()
    for label in selected_groups:
        if label in forecasts_dict:
            forecast = forecasts_dict[label]
            last_hist = agg_df_dict[label]['ds'].max() if (agg_df_dict and label in agg_df_dict) else forecast['ds'][forecast['yhat_upper'].isna()].max()
            fut = forecast[forecast['ds'] > last_hist].copy()
            if not fut.empty:
                fut['group'] = label
                all_forecasts = pd.concat([all_forecasts, fut[['ds', 'yhat', 'group']]])
            else:
                st.warning(f"No future data for {label} after {last_hist}")

    if all_forecasts.empty:
        st.warning("No future forecast data available for the selected groups.")
        return None

    pivot_df = all_forecasts.pivot(index='group', columns='ds', values='yhat')
    pivot_df.columns = pivot_df.columns.strftime('%Y-%m-%d')

    fig, ax = plt.subplots(figsize=(14, len(selected_groups) * 0.5 + 2))
    sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=.5, ax=ax)
    ax.set_title(f"{target_column} Forecast Heatmap by {group_title}")
    ax.set_ylabel("Group")
    ax.set_xlabel("Date")
    plt.tight_layout()

    st.pyplot(fig)
    st.download_button(
        label="Download Heatmap Data",
        data=pivot_df.reset_index().to_csv(index=False),
        file_name=f"forecast_heatmap_{target_column}_by_{group_title}.csv",
        mime="text/csv",
    )
    return fig


def convert_df_to_csv(df):
    return df.to_csv(index=False)


# -----------------------------------------------------------
# Streamlit app
# -----------------------------------------------------------
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = {}

st.set_page_config(page_title="Time Series Forecasting AI", layout="wide")
st.title("Time Series Forecasting - Generative AI")
st.markdown("Upload your dataset, specify the target column and its calculation in the context, and customize the forecast.")

with st.sidebar:
    st.header("Options")

    # Defaults tailored to your demo CSV
    context = st.text_area(
        "Dataset context (e.g., 'Retail sales data, Weekly_Sales is the target')",
        value=("Manufacturing weekly sales demo. Date column = WEEK_END_DATE. "
               "Target column = UNITS (units sold). Primary group = UPC (product code). "
               "Optional secondary group = STORE_NUM (store). "
               "Other available columns: PRICE, BASE_PRICE, FEATURE, DISPLAY, TPR_ONLY, VISITS, HHS, SPEND.")
    )
    date_column = st.text_input("Date column name", value="WEEK_END_DATE")

    # Upload or use demo CSV from repo
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    use_demo = st.checkbox("Use demo dataset (DemoSalesData.csv)", value=False)

    df, columns = (None, [])
    if uploaded_file is not None:
        df, columns = load_data(file=uploaded_file, date_column=date_column)
    elif use_demo:
        demo_path = "DemoSalesData.csv"
        if not os.path.exists(demo_path):
            alt_path = os.path.join("data", "DemoSalesData.csv")
            demo_path = alt_path if os.path.exists(alt_path) else demo_path
        df, columns = load_data(file=None, date_column=date_column, filename=demo_path)
        st.caption(f"Using demo dataset: {demo_path}")

    if df is not None:
        st.write("Columns:", ", ".join(columns))
        st.write(f"Raw data date range: {df['ds'].min()} to {df['ds'].max()}")

        target_column = st.text_input("Target column to forecast", value="UNITS")

        enable_groupby = st.checkbox("Enable Group By", value=True)
        selected_group_columns = []

        if enable_groupby:
            non_date_columns = [c for c in df.columns if c != 'ds']

            group_col1 = st.selectbox(
                "Primary Group Column",
                options=[""] + non_date_columns,
                index=(non_date_columns.index('UPC') + 1) if 'UPC' in non_date_columns else 0,
                key="group1",
            )
            if group_col1:
                selected_group_columns.append(group_col1)

            group_col2 = st.selectbox(
                "Secondary Group Column (optional)",
                options=[""] + [c for c in non_date_columns if c != group_col1],
                index=0,
                key="group2",
            )
            if group_col2:
                selected_group_columns.append(group_col2)

            if selected_group_columns:
                top_n = st.slider("Number of Top Groups to Compare", min_value=2, max_value=20, value=10)
        else:
            selected_group_columns = []
            top_n = 10

        # Frequency & periods (D/W/M)
        granularity_options = infer_granularity(df)
        frequency_map = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}
        frequency_label = st.selectbox(
            "Forecast Frequency",
            options=[frequency_map[f] for f in granularity_options],
            index=1 if 'W' in granularity_options else 0,
        )
        frequency = [k for k, v in frequency_map.items() if v == frequency_label][0]

        period_unit = 'days' if frequency == 'D' else 'weeks' if frequency == 'W' else 'months'
        max_periods = 365 if frequency == 'D' else 52 if frequency == 'W' else 12
        periods = st.slider(f"Forecast Periods ({period_unit})", min_value=1, max_value=max_periods, value=12)

        data_color = st.color_picker("Historical Data Color", value="#000000")
        forecast_color = st.color_picker("Forecast Color", value="#FF0000")
    else:
        # sensible fallbacks so code paths compile
        target_column = "UNITS"
        enable_groupby = False
        selected_group_columns = []
        top_n = 10
        frequency = 'W'
        periods = 12
        data_color = '#000000'
        forecast_color = '#FF0000'

    run_button = st.button("Generate Forecast", disabled=(df is None))

# -----------------------------------------------------------
# Run forecasting
# -----------------------------------------------------------
if run_button and df is not None:
    with st.spinner("Generating forecasts..."):
        df = engineer_features(df, target_column, context, columns)
        st.write("Columns after feature engineering:", ", ".join(df.columns))

        st.session_state.forecast_results = {
            'df': df,
            'target_column': target_column,
            'periods': periods,
            'frequency': frequency,
            'data_color': data_color,
            'forecast_color': forecast_color,
            'selected_group_columns': selected_group_columns,
            'top_n': top_n,
            'context': context,
        }

        # All data combined
        all_agg_df = aggregate_data(df, target_column, frequency)
        all_model, all_forecast, all_fig1, all_fig2 = run_forecast(
            all_agg_df, target_column, periods, frequency, data_color, forecast_color
        )
        st.session_state.forecast_results['all'] = (all_model, all_forecast, all_fig1, all_fig2, all_agg_df)

        # Grouped results
        if enable_groupby and selected_group_columns:
            combined_fig, combined_forecasts, combined_agg_df = run_multi_group_forecast(
                df, selected_group_columns, target_column, periods, frequency, context, data_color, forecast_color, top_n=top_n
            )
            st.session_state.forecast_results['combined'] = (combined_fig, combined_forecasts, combined_agg_df)

            if len(selected_group_columns) >= 1:
                primary_fig, primary_forecasts, primary_agg_df = run_multi_group_forecast(
                    df, selected_group_columns, target_column, periods, frequency, context, data_color, forecast_color, top_n=top_n, filter_group=selected_group_columns[0]
                )
                st.session_state.forecast_results['primary'] = (primary_fig, primary_forecasts, primary_agg_df)

            if len(selected_group_columns) >= 2:
                secondary_fig, secondary_forecasts, secondary_agg_df = run_multi_group_forecast(
                    df, selected_group_columns, target_column, periods, frequency, context, data_color, forecast_color, top_n=top_n, filter_group=selected_group_columns[1]
                )
                st.session_state.forecast_results['secondary'] = (secondary_fig, secondary_forecasts, secondary_agg_df)

# -----------------------------------------------------------
# Display & drilldown
# -----------------------------------------------------------
if st.session_state.get('forecast_results'):
    res = st.session_state.forecast_results
    df = res['df']
    target_column = res['target_column']
    periods = res['periods']
    frequency = res['frequency']
    data_color = res['data_color']
    forecast_color = res['forecast_color']
    selected_group_columns = res['selected_group_columns']
    top_n = res['top_n']
    context = res['context']

    filter_options = ["All Data Combined"]
    if selected_group_columns:
        filter_options.append(f"Combined ({' & '.join(selected_group_columns)})")
        filter_options.append(selected_group_columns[0])
        if len(selected_group_columns) >= 2:
            filter_options.append(selected_group_columns[1])

    st.subheader("Filter Forecast Results")

    # Default the dropdown to "Combined (UPC)" if it's available
    default_idx = 0
    if selected_group_columns:
        combined_label = f"Combined ({' & '.join(selected_group_columns)})"
        if combined_label in filter_options:
            default_idx = filter_options.index(combined_label)

    selected_filter = st.selectbox(
        "Choose a filter to view results:",
        filter_options,
        index=default_idx,
        key="filter_select",
)


    if selected_filter == "All Data Combined" and 'all' in res:
        all_model, all_forecast, all_fig1, all_fig2, all_agg_df = res['all']
        if all_model is not None:
            st.subheader("Forecast Results (All Data Combined)")
            st.write(all_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            st.markdown("### GenAI Insights")
            ins_md = get_genai_insights(all_forecast, target_column, context, style="detailed", hist_df=all_agg_df)
            st.markdown(ins_md)
            st.download_button(
                label="Download Insights (.md)",
                data=ins_md,
                file_name="insights_all.md",
                mime="text/markdown",
            )


            if all_fig1:
                st.pyplot(all_fig1)
            if all_fig2:
                st.pyplot(all_fig2)
            st.download_button(
                label="Download Full Dataset",
                data=convert_df_to_csv(df),
                file_name="dataset_with_features.csv",
                mime="text/csv",
            )

    elif selected_group_columns:
        # Combined (all selected columns)
        if selected_filter == f"Combined ({' & '.join(selected_group_columns)})" and 'combined' in res:
            combined_fig, combined_forecasts, combined_agg_df = res['combined']
            st.subheader(f"Forecast Results by {' & '.join(selected_group_columns)}")
            st.write(f"Aggregated data ({frequency} frequency):")
            st.dataframe(combined_agg_df.head())
            if combined_fig:
                st.pyplot(combined_fig)

            if combined_forecasts:
                title = " & ".join(selected_group_columns)
                if f"heatmap_data_{title}" in st.session_state:
                    f_dict, agg_dict, labels, tcol, gtitle = st.session_state[f"heatmap_data_{title}"]
                    create_forecast_heatmap(f_dict, labels, tcol, gtitle, agg_dict)

                all_fc = pd.concat([fc[['ds', 'yhat']].assign(group=g) for g, fc in combined_forecasts.items()])
                st.download_button(
                    label="Download Combined Forecasts",
                    data=convert_df_to_csv(all_fc),
                    file_name=f"combined_forecasts_{target_column}.csv",
                    mime="text/csv",
                )

                detailed_group = st.selectbox("Select a group for detailed view", options=list(combined_forecasts.keys()), key="combined_detail")
                if detailed_group:
                    st.subheader(f"Detailed Forecast for {detailed_group}")
                    forecast = combined_forecasts[detailed_group]
                     # Robust filtering (fixes previous IndexError)
                    group_data = filter_by_selection(combined_agg_df, detailed_group, selected_group_columns)
                    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                    st.markdown("### GenAI Insights")
                    ins_md = get_genai_insights(forecast, target_column, context, style="detailed", hist_df=group_data)
                    st.markdown(ins_md)
                    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(detailed_group))
                    st.download_button(
                        label="Download Insights (.md)",
                        data=ins_md,
                        file_name=f"insights_{safe_name}.md",
                        mime="text/markdown",
                    )



                   

                    st.write(f"Detailed view for {detailed_group} has {len(group_data)} rows")
                    st.dataframe(group_data.head())
                    model, _, fig1, fig2 = run_forecast(group_data, target_column, periods, frequency, data_color, forecast_color)
                    if fig1: st.pyplot(fig1)
                    if fig2: st.pyplot(fig2)

        # Primary only
        elif selected_filter == selected_group_columns[0] and 'primary' in res:
            primary_fig, primary_forecasts, primary_agg_df = res['primary']
            st.subheader(f"Forecast Results by {selected_group_columns[0]}")
            st.write(f"Aggregated data ({frequency} frequency):")
            st.dataframe(primary_agg_df.head())
            if primary_fig:
                st.pyplot(primary_fig)

            if primary_forecasts:
                title = selected_group_columns[0]
                labels = list(primary_forecasts.keys())
                selected_group = st.selectbox(f"Select {title} to view", options=["All"] + labels, key="primary_group_select")

                if f"heatmap_data_{title}" in st.session_state:
                    f_dict, agg_dict, all_labels, tcol, gtitle = st.session_state[f"heatmap_data_{title}"]
                    create_forecast_heatmap(f_dict, all_labels, tcol, gtitle, agg_dict)

                all_fc = pd.concat([fc[['ds', 'yhat']].assign(group=g) for g, fc in primary_forecasts.items()])
                st.download_button(
                    label=f"Download {title} Forecasts",
                    data=convert_df_to_csv(all_fc),
                    file_name=f"{title}_forecasts_{target_column}.csv",
                    mime="text/csv",
                )

                if selected_group != "All":
                    st.subheader(f"Detailed Forecast for {selected_group}")
                    forecast = primary_forecasts[selected_group]
                    group_data = filter_by_selection(primary_agg_df, selected_group, selected_group_columns)
                    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                    st.markdown("### GenAI Insights")
                    ins_md = get_genai_insights(forecast, target_column, context, style="detailed", hist_df=group_data)
                    st.markdown(ins_md)
                    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(selected_group))
                    st.download_button(
                        label="Download Insights (.md)",
                        data=ins_md,
                        file_name=f"insights_{safe_name}.md",
                        mime="text/markdown",
                    )




                
                    st.write(f"Detailed view for {selected_group} has {len(group_data)} rows")
                    st.dataframe(group_data.head())
                    model, _, fig1, fig2 = run_forecast(group_data, target_column, periods, frequency, data_color, forecast_color)
                    if fig1: st.pyplot(fig1)
                    if fig2: st.pyplot(fig2)

        # Secondary only
        elif len(selected_group_columns) >= 2 and selected_filter == selected_group_columns[1] and 'secondary' in res:
            secondary_fig, secondary_forecasts, secondary_agg_df = res['secondary']
            st.subheader(f"Forecast Results by {selected_group_columns[1]}")
            st.write(f"Aggregated data ({frequency} frequency):")
            st.dataframe(secondary_agg_df.head())
            if secondary_fig:
                st.pyplot(secondary_fig)

            if secondary_forecasts:
                title = selected_group_columns[1]
                labels = list(secondary_forecasts.keys())
                selected_group = st.selectbox(f"Select {title} to view", options=["All"] + labels, key="secondary_group_select")

                if f"heatmap_data_{title}" in st.session_state:
                    f_dict, agg_dict, all_labels, tcol, gtitle = st.session_state[f"heatmap_data_{title}"]
                    create_forecast_heatmap(f_dict, all_labels, tcol, gtitle, agg_dict)

                all_fc = pd.concat([fc[['ds', 'yhat']].assign(group=g) for g, fc in secondary_forecasts.items()])
                st.download_button(
                    label=f"Download {title} Forecasts",
                    data=convert_df_to_csv(all_fc),
                    file_name=f"{title}_forecasts_{target_column}.csv",
                    mime="text/csv",
                )

                if selected_group != "All":
                    st.subheader(f"Detailed Forecast for {selected_group}")
                    forecast = secondary_forecasts[selected_group]
                    group_data = filter_by_selection(secondary_agg_df, selected_group, selected_group_columns)
                    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                    st.markdown("### GenAI Insights")
                    ins_md = get_genai_insights(forecast, target_column, context, style="detailed", hist_df=group_data)
                    st.markdown(ins_md)
                    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(selected_group))
                    st.download_button(
                        label="Download Insights (.md)",
                        data=ins_md,
                        file_name=f"insights_{safe_name}.md",
                        mime="text/markdown",
                    )



                    
                    st.write(f"Detailed view for {selected_group} has {len(group_data)} rows")
                    st.dataframe(group_data.head())
                    model, _, fig1, fig2 = run_forecast(group_data, target_column, periods, frequency, data_color, forecast_color)
                    if fig1: st.pyplot(fig1)
                    if fig2: st.pyplot(fig2)
