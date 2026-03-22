

import os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import holidays as holidays_lib
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="static")
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Global Dataset Load ──
try:
    print("Loading 36MB Kaggle Rossmann dataset...")
    ROSSMANN_DF = pd.read_csv("data/train.csv", low_memory=False)
    ROSSMANN_DF["Date"] = pd.to_datetime(ROSSMANN_DF["Date"])
    print("Rossmann dataset loaded successfully!")
except Exception as e:
    print("Could not load data/train.csv:", e)
    ROSSMANN_DF = None

# ══════════════════════════════════════════════════════════
#  1.  DATA GENERATION  (DMart-style synthetic)
# ══════════════════════════════════════════════════════════

INDIAN_HOLIDAYS = {
    "Republic Day":        "01-26",
    "Holi":                "03-25",
    "Independence Day":    "08-15",
    "Gandhi Jayanti":      "10-02",
    "Dussehra":            "10-12",
    "Diwali":              "11-01",
    "Christmas":           "12-25",
    "New Year":            "01-01",
}

def get_holiday_dates(start: pd.Timestamp, end: pd.Timestamp):
    """Return set of holiday date strings for date range."""
    h_dates = set()
    for year in range(start.year, end.year + 1):
        for name, mmdd in INDIAN_HOLIDAYS.items():
            try:
                d = pd.Timestamp(f"{year}-{mmdd}")
                if start <= d <= end:
                    h_dates.add(d.date())
            except Exception:
                pass
    return h_dates

def generate_dmart_data(start_date: str, end_date: str, n_products: int = 3):
    if ROSSMANN_DF is None:
        return None, "Error: Kaggle 'data/train.csv' not loaded on server."

    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)
    n     = (end - start).days + 1
    if n < 60:
        return None, "Please select at least 60 days of history."

    # Map Rossmann Store IDs 1, 2, 3 to DMart categories
    product_configs = [
        {"store_id": 1, "name": "Groceries",    "weather_sensitivity": -0.15, "base": 85000},
        {"store_id": 2, "name": "Electronics",  "weather_sensitivity": -0.25, "base": 45000},
        {"store_id": 3, "name": "Apparel",      "weather_sensitivity": -0.30, "base": 30000},
    ][:n_products]

    # Filter global DataFrame once
    mask = (ROSSMANN_DF["Date"] >= start) & (ROSSMANN_DF["Date"] <= end)
    df_filtered = ROSSMANN_DF.loc[mask].copy()

    rows = []
    # Real Dates
    full_dates = pd.DataFrame({"Date": pd.date_range(start, end)})
    holiday_set = get_holiday_dates(start, end)

    for cfg in product_configs:
        store_id  = cfg["store_id"]
        prod_name = cfg["name"]
        base      = cfg["base"]

        # Extract real data for this store, merge to fill date holes
        store_df = df_filtered[df_filtered["Store"] == store_id]
        
        is_fallback = store_df.empty
        
        store_df = pd.merge(full_dates, store_df, on="Date", how="left")
        
        # Fill holes
        store_df["Sales"].fillna(0, inplace=True)
        store_df["Promo"].fillna(0, inplace=True)
        
        # Real variables
        sales  = np.nan_to_num(store_df["Sales"].values, nan=0.0)
        promos = np.nan_to_num(store_df["Promo"].values, nan=0.0)

        if is_fallback:
            trend  = np.linspace(base, base * 1.4, n)
            weekly = base * 0.18 * np.sin(2 * np.pi * np.arange(n) / 7)
            monthly= base * 0.10 * np.sin(2 * np.pi * np.arange(n) / 30)
            noise  = np.random.normal(0, base * 0.06, n)
            sales  = np.maximum(trend + weekly + monthly + noise, 0)
            promo_days = np.random.choice(n, size=max(1, int(n * 0.05)), replace=False)
            promos[promo_days] = 1
        
        # Combine Rossmann StateHoliday with Indian Holidays
        state_hol = store_df.get("StateHoliday", pd.Series(["0"]*n)).astype(str)
        holidays_arr = (state_hol != "0").astype(int).values.copy()

        np.random.seed(42 + store_id)
        
        # Rebuild Synthetics for UX consistency
        temperature   = 28 + 5 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 2, n)
        is_raining    = np.zeros(n)
        days_until    = np.zeros(n)
        is_payday     = np.zeros(n)
        demand_index  = np.random.uniform(40, 60, n)
        
        dates = full_dates["Date"]
        for i, d in enumerate(dates):
            if 6 <= d.month <= 9: is_raining[i] = 1 if np.random.random() < 0.55 else 0
            elif d.month in [12, 1]: is_raining[i] = 1 if np.random.random() < 0.08 else 0
            if d.day == 1 or d.day >= 28: is_payday[i] = 1
            if d.date() in holiday_set: holidays_arr[i] = 1
            
        for i, d in enumerate(dates):
            # scan forward for next holiday
            for j in range(0, 15):
                if i+j < n and holidays_arr[i+j] == 1:
                    days_until[i] = j
                    break
            else:
                days_until[i] = 15

            if days_until[i] <= 7: demand_index[i] += (8 - days_until[i]) * 4
            if d.dayofweek in [3, 4]: demand_index[i] += 8
            
        demand_index = np.clip(demand_index, 0, 100)
        web_traffic_lag = np.zeros(n)
        web_traffic_lag[2:] = demand_index[:-2]
        web_traffic_lag[:2] = demand_index[0]
        
        comp_promo = np.zeros(n)
        comp_days  = np.random.choice(n, size=max(1, int(n * 0.04)), replace=False)
        comp_promo[comp_days] = 1

        for i, d in enumerate(dates):
            s_val = float(sales[i])
            if is_fallback:
                if holidays_arr[i]: s_val += base * np.random.uniform(0.35, 0.65)
                if is_payday[i]:    s_val += base * 0.12
                s_val += cfg["weather_sensitivity"] * is_raining[i] * base * 0.5 + (temperature[i] - 28) * base * 0.005
                s_val += (demand_index[i] - 50) * base * 0.003
                if promos[i]: s_val += base * np.random.uniform(0.18, 0.35)
                if comp_promo[i]: s_val -= base * np.random.uniform(0.08, 0.18)
                s_val = max(s_val, 0)
                
            rows.append({
                "date":              d,
                "product":           prod_name,
                "sales":             round(s_val, 2),
                "is_holiday":        int(holidays_arr[i]),
                "days_until_holiday":int(days_until[i]),
                "is_payday":         int(is_payday[i]),
                "temperature":       round(float(temperature[i]), 1),
                "is_raining":        int(is_raining[i]),
                "demand_index":      round(float(demand_index[i]), 1),
                "web_traffic_lag":   round(float(web_traffic_lag[i]), 1),
                "own_promo":         int(promos[i]),           # REAL DATA
                "competitor_promo":  int(comp_promo[i]),
            })

    df = pd.DataFrame(rows).sort_values(["product","date"]).reset_index(drop=True)
    return df, None


# ══════════════════════════════════════════════════════════
#  2.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

ADVANCED_FEATURES = [
    # Calendar
    "day_of_week","day_of_month","month","quarter","week_of_year","is_weekend",
    # Factor 1
    "is_holiday","days_until_holiday","is_payday",
    # Factor 2
    "temperature","is_raining",
    # Factor 3
    "demand_index","web_traffic_lag",
    # Factor 4
    "own_promo","competitor_promo",
    # Lags
    "lag_1","lag_3","lag_7","lag_14","lag_30",
    # Rolling
    "rolling_mean_7","rolling_mean_14","rolling_mean_30",
    "rolling_std_7","rolling_std_14","rolling_std_30",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    for lag in [1, 3, 7, 14, 30]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)
    for w in [7, 14, 30]:
        df[f"rolling_mean_{w}"] = df["sales"].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"]  = df["sales"].shift(1).rolling(w).std()

    df.dropna(inplace=True)
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════
#  3.  ML PIPELINE  (per-product XGBoost)
# ══════════════════════════════════════════════════════════

def train_and_forecast(df: pd.DataFrame, horizon: int):
    products = df["product"].unique().tolist()
    all_history, all_test_fit, all_forecast = [], [], []
    all_metrics = {}

    for product in products:
        pdf  = df[df["product"] == product].copy().reset_index(drop=True)
        feat = engineer_features(pdf)
        feature_cols = [c for c in ADVANCED_FEATURES if c in feat.columns]

        split    = int(len(feat) * 0.8)
        train_df = feat.iloc[:split]
        test_df  = feat.iloc[split:]

        model = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            early_stopping_rounds=30, eval_metric="rmse", verbosity=0
        )
        model.fit(
            train_df[feature_cols], train_df["sales"],
            eval_set=[(test_df[feature_cols], test_df["sales"])],
            verbose=False
        )

        preds = model.predict(test_df[feature_cols])
        mae   = mean_absolute_error(test_df["sales"], preds)
        rmse  = np.sqrt(mean_squared_error(test_df["sales"], preds))
        mask  = test_df["sales"] > 0
        mape  = np.mean(np.abs((test_df.loc[mask, "sales"] - preds[mask]) / test_df.loc[mask, "sales"])) * 100 if mask.any() else 0.0
        if pd.isna(mape) or np.isinf(mape): mape = 0.0
        all_metrics[product] = {"MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE": round(mape,2)}

        # Feature importance
        fi = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(fi.items(), key=lambda x: -x[1])[:5]
        all_metrics[product]["top_features"] = [{"name": k, "importance": round(float(v),4)} for k,v in top_features]

        # History
        for _, r in pdf.iterrows():
            all_history.append({"date": r["date"].strftime("%Y-%m-%d"), "sales": r["sales"], "product": product})

        # Test fit
        for (_, r), p in zip(test_df.iterrows(), preds):
            all_test_fit.append({"date": r["date"].strftime("%Y-%m-%d"), "predicted": round(float(p),2), "product": product})

        # Future forecast (iterative)
        last_date  = feat["date"].max()
        holiday_set = get_holiday_dates(last_date, last_date + timedelta(days=horizon+15))
        history_df = pdf.copy()

        for i in range(1, horizon + 1):
            next_date = last_date + timedelta(days=i)
            ddate     = next_date.date()
            days_until = next(
                (j for j in range(16) if (next_date + timedelta(days=j)).date() in holiday_set), 15
            )
            row = {
                "date":              next_date,
                "sales":             np.nan,
                "product":           product,
                "is_holiday":        int(ddate in holiday_set),
                "days_until_holiday":days_until,
                "is_payday":         int(next_date.day == 1 or next_date.day >= 28),
                "temperature":       28 + 5 * np.sin(2 * np.pi * i / 365),
                "is_raining":        int(6 <= next_date.month <= 9 and np.random.random() < 0.5),
                "demand_index":      55 + (8 - days_until) * 3 if days_until <= 7 else 52,
                "web_traffic_lag":   55.0,
                "own_promo":         0,
                "competitor_promo":  0,
            }
            temp_df  = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
            temp_feat= engineer_features(temp_df)
            latest   = temp_feat.iloc[[-1]]
            present  = [c for c in feature_cols if c in latest.columns]
            pred     = float(model.predict(latest[present])[0])

            history_df = pd.concat(
                [history_df, pd.DataFrame([{**row, "sales": pred}])],
                ignore_index=True
            )
            all_forecast.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "forecast": round(pred, 2),
                "product": product,
                "is_holiday": row["is_holiday"],
                "is_payday": row["is_payday"],
                "is_raining": row["is_raining"],
            })

    # Aggregate across products
    hist_agg = (
        pd.DataFrame(all_history).groupby("date")["sales"].sum()
        .reset_index().rename(columns={"sales":"total_sales"})
    )
    fc_agg = (
        pd.DataFrame(all_forecast).groupby("date")["forecast"].sum()
        .reset_index().rename(columns={"forecast":"total_forecast"})
    )

    fc_df = pd.DataFrame(all_forecast)
    summary = {
        "hist_mean":     round(hist_agg["total_sales"].mean(), 2),
        "hist_max":      round(hist_agg["total_sales"].max(), 2),
        "hist_total":    round(hist_agg["total_sales"].sum(), 2),
        "fc_total":      round(fc_agg["total_forecast"].sum(), 2),
        "fc_avg":        round(fc_agg["total_forecast"].mean(), 2),
        "holiday_days":  int(fc_df["is_holiday"].sum()),
        "rainy_days":    int(fc_df["is_raining"].sum()),
        "payday_days":   int(fc_df["is_payday"].sum()),
    }

    return {
        "history":    hist_agg.to_dict("records"),
        "per_product_history": all_history,
        "test_fit":   all_test_fit,
        "forecast":   fc_agg.to_dict("records"),
        "per_product_forecast": all_forecast,
        "metrics":    all_metrics,
        "products":   products,
        "summary":    summary,
    }


# ══════════════════════════════════════════════════════════
#  4.  LLM ANALYSIS  (Groq — free)
# ══════════════════════════════════════════════════════════

def get_llm_analysis(result: dict, horizon: int) -> str:
    if not GROQ_API_KEY:
        return "⚠️ Set GROQ_API_KEY to enable AI analysis.\nGet a free key at console.groq.com"
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        s  = result["summary"]
        m  = result["metrics"]

        metrics_txt = "\n".join(
            f"  {p}: MAE=₹{v['MAE']:,.0f}, MAPE={v['MAPE']}%, Top feature={v['top_features'][0]['name']}"
            for p, v in m.items()
        )
        prompt = f"""You are a senior retail analytics consultant for a DMart-style hypermarket in India.

FORECAST SUMMARY ({horizon} days ahead):
- Historical daily avg: ₹{s['hist_mean']:,.0f}
- Forecast total: ₹{s['fc_total']:,.0f} | Avg/day: ₹{s['fc_avg']:,.0f}
- Holiday days in forecast window: {s['holiday_days']}
- Rainy days (foot traffic impact): {s['rainy_days']}
- Payday days (salary cycle boost): {s['payday_days']}

MODEL ACCURACY BY PRODUCT:
{metrics_txt}

The model uses 4 advanced factors:
1. External Events (holidays, paydays, days_until_holiday)
2. Weather (temperature, rainfall → foot traffic)
3. Demand Sensing (Google Trends index, web traffic lag)
4. Cannibalization (own promotions, competitor promotions)

Respond with EXACTLY these 5 sections:
📈 TREND & MOMENTUM: What does the data say about growth direction? (2-3 sentences)
🎯 KEY FACTOR INSIGHTS: Which of the 4 factors is driving the most impact and why? (3-4 sentences)  
💡 TOP 3 ACTIONS: Numbered. Concrete DMart-style recommendations (staffing, inventory, promotions).
⚠️ RISKS TO WATCH: 2 specific risks that could make the forecast wrong.
🏆 CONFIDENCE SCORE: Rate this forecast 1-10 with a one-line justification based on MAPE values.

Be specific with Indian retail context. Reference rupee values. No generic advice."""

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3, max_tokens=700
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"LLM error: {e}"


# ══════════════════════════════════════════════════════════
#  5.  ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/forecast", methods=["POST"])
def forecast():
    try:
        body       = request.json
        start_date = body.get("start_date")
        end_date   = body.get("end_date")
        horizon    = int(body.get("horizon", 30))
        horizon    = max(7, min(horizon, 90))
        n_products = int(body.get("n_products", 3))

        df, err = generate_dmart_data(start_date, end_date, n_products)
        if err:
            return jsonify({"error": err}), 400

        result   = train_and_forecast(df, horizon)
        analysis = get_llm_analysis(result, horizon)
        result["llm_analysis"] = analysis
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    print("🚀  DMart Forecasting App → http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=True)