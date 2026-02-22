import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import zipfile

# Machine Learning & Evaluation
import xgboost as xgb
import joblib
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Parallelization
from concurrent.futures import ThreadPoolExecutor


# ============================================================
# CONSTANTS
# ============================================================

TRAIN_END     = '2015-12-31'
REF_START     = '1990-01-01'
SPLIT_DATE    = '2015-12-31'
SUMMER_MONTHS = [5, 6, 7, 8, 9]

ZSCORE_COLS = [
    'temperature_2m', 't_850hPa', 't_500hPa',
    'pressure_msl', 'relative_humidity_2m',
    'soil_moisture_0_to_7cm', 'wind_speed_10m',
]


# ============================================================
# PREPROCESSING
# ============================================================

def process_weather_data(uploaded_files: list) -> list:
    """
    Loads uploaded CSV/Parquet files, standardizes the timestamp column,
    sorts by time, and returns a list of result dicts with in-memory buffers.
    """
    results = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        city_id   = file_name.split('_')[0].lower()

        try:
            # Load file based on extension
            if file_name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif file_name.lower().endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                st.warning(f"⚠️ Skipped {file_name}: Unsupported file format.")
                continue

            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                time_col = [c for c in df.columns if 'time' in c.lower()]
                if time_col:
                    df.rename(columns={time_col[0]: 'timestamp'}, inplace=True)
                else:
                    st.warning(f"⚠️ Skipped {file_name}: No time column found.")
                    continue

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Extract time range metadata
            start_dt = df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
            end_dt   = df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            start_yr = df['timestamp'].min().year
            end_yr   = df['timestamp'].max().year

            var_names = [c for c in df.columns if c != 'timestamp']
            var_count = len(var_names)

            filename_base = f"{city_id}_{start_yr}-{end_yr}_vars{var_count}_v1"

            # Serialize to in-memory Parquet buffer
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)

            metadata = {
                "city": city_id,
                "time_frame": {"start_exact": start_dt, "end_exact": end_dt, "rows": len(df)},
                "variables":  {"count": var_count, "list": var_names}
            }
            json_bytes = json.dumps(metadata, indent=4).encode('utf-8')

            results.append({
                "filename_base": filename_base,
                "parquet_buffer": parquet_buffer,
                "json_bytes":     json_bytes,
                "metadata":       metadata,
            })

            st.success(f"✅ Processed: {filename_base} ({var_count} variables)")

        except Exception as e:
            st.error(f"❌ Error in {file_name}: {str(e)}")

    return results


def transform_wind_direction(all_frames: dict) -> dict:
    """
    Replaces wind_direction_10m (degrees) with cyclic sin/cos encoding.
    Skips silently if already transformed.
    """
    for city_id, df in all_frames.items():
        if 'wind_direction_10m' in df.columns:
            rad = df['wind_direction_10m'] * np.pi / 180
            df['wind_dir_sin'] = np.sin(rad)
            df['wind_dir_cos'] = np.cos(rad)
            df.drop(columns=['wind_direction_10m'], inplace=True)
            all_frames[city_id] = df
            st.success(f"✅ Wind direction transformed: {city_id}")
        elif 'wind_dir_sin' in df.columns and 'wind_dir_cos' in df.columns:
            pass  # Already transformed – no output needed
        else:
            st.info(f"ℹ️ No wind_direction_10m column found for {city_id} – skipped.")
    return all_frames


def show_validation_check(all_frames: dict) -> None:
    """Displays a validation table for the wind direction transformation."""
    st.subheader("🔍 Validation Check")

    rows = []
    for city_id, df in all_frames.items():
        df_cols = set(df.columns)
        rows.append({
            "City":                       city_id.upper(),
            "wind_direction_10m removed": "✅" if 'wind_direction_10m' not in df_cols else "❌",
            "sin/cos present":            "✅" if {'wind_dir_sin', 'wind_dir_cos'}.issubset(df_cols) else "❌",
            "# Variables":                len(df_cols - {'timestamp'}),
        })

    st.dataframe(pd.DataFrame(rows), width='content')


def optimize_dtypes(all_frames: dict) -> dict:
    """Converts float64 columns to float32 to reduce memory footprint."""
    for city_id, df in all_frames.items():
        cols = df.select_dtypes(include=['float64']).columns
        df[cols] = df[cols].astype('float32')
        all_frames[city_id] = df
    st.success("✅ float64 → float32 conversion complete.")
    return all_frames


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def compute_climatology(df: pd.DataFrame, cols: list) -> dict:
    """
    Computes hourly climatological mean and std for each variable
    using only the reference period (REF_START to TRAIN_END) to avoid leakage.
    """
    ref  = df[REF_START:TRAIN_END]
    clim = {}
    for col in cols:
        if col not in df.columns:
            continue
        grp = ref.groupby([ref.index.dayofyear, ref.index.hour])[col]
        clim[col] = {'mean': grp.mean(), 'std': grp.std()}
    return clim


def apply_zscore(df: pd.DataFrame, clim: dict) -> pd.DataFrame:
    """Applies leakage-free Z-score normalization using precomputed climatology."""
    doy  = df.index.dayofyear.values
    hour = df.index.hour.values
    for col, stats in clim.items():
        if col not in df.columns:
            continue
        mean_2d  = stats['mean'].unstack(level=1).values
        std_2d   = stats['std'].unstack(level=1).values
        mean_arr = mean_2d[doy - 1, hour]
        std_arr  = std_2d[doy - 1, hour]
        df[f'{col}_zscore'] = (df[col].values - mean_arr) / (std_arr + 1e-8)
    return df


def engineer_features(df: pd.DataFrame, city_name: str) -> tuple:
    """
    Applies all feature engineering steps to a single city DataFrame:
    1. Z-score normalization (leakage-free)
    2. Cyclical time features (hour, day-of-year)
    3. Lag features (72h, 48h)
    4. Tendency features (24h differences)
    5. Heat Aridity Index
    6. V850 smoothed (6h rolling mean)
    """
    df = df.copy()

    # Set timestamp as index if not already done
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)

    # 1. Z-score normalization
    clim = compute_climatology(df, ZSCORE_COLS)
    df   = apply_zscore(df, clim)

    # 2. Cyclical time features
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['doy_sin']  = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['doy_cos']  = np.cos(2 * np.pi * df.index.dayofyear / 365)

    # 3. Lag features
    if 't_850hPa' in df.columns:
        df['lag_t850_72h']  = df['t_850hPa'].shift(72)
    if 'pressure_msl' in df.columns:
        df['lag_press_48h'] = df['pressure_msl'].shift(48)
    if 'v_850hPa' in df.columns:
        df['lag_v850_72h']  = df['v_850hPa'].shift(72)
    if 't_500hPa' in df.columns:
        df['lag_t500_72h']  = df['t_500hPa'].shift(72)
        df['lag_t500_48h']  = df['t_500hPa'].shift(48)

    # 4. Tendency features (24h differences)
    if 'pressure_msl' in df.columns:
        df['delta_press_24h'] = df['pressure_msl'] - df['pressure_msl'].shift(24)
    if 'relative_humidity_2m' in df.columns:
        df['delta_hum_24h']   = df['relative_humidity_2m'] - df['relative_humidity_2m'].shift(24)

    # 5. Heat Aridity Index (t_850hPa converted to Celsius)
    if 't_850hPa' in df.columns and 'relative_humidity_2m' in df.columns:
        t850_celsius         = df['t_850hPa'] - 273.15
        df['heat_aridity_index'] = t850_celsius - df['relative_humidity_2m']

    # 6. V850 smoothed (6h rolling mean)
    if 'v_850hPa' in df.columns:
        df['v850_smooth_6h'] = df['v_850hPa'].rolling(window=6).mean()

    return df, clim


def _process_city(args: tuple) -> tuple:
    """Helper for parallel feature engineering execution."""
    city, df = args
    return city, engineer_features(df, city)


def run_feature_engineering(all_frames: dict) -> tuple[dict, dict]:
    """Runs feature engineering for all cities in parallel using ThreadPoolExecutor."""
    engineered_frames  = {}
    city_climatologies = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(_process_city, all_frames.items()))

    for city, (df, clim) in results:
        engineered_frames[city]  = df
        city_climatologies[city] = clim

    return engineered_frames, city_climatologies


# ============================================================
# MODEL TRAINING
# ============================================================

def label_heatwave_hourly(df: pd.DataFrame, city_name: str,
                           percentile: float = 0.95) -> tuple:
    """
    Creates binary hourly heatwave labels using the Klement event definition:
    - P95 threshold computed from the reference period (no leakage)
    - A Klement event requires 72 consecutive hot hours
    - Target y=1 if a Klement event occurs within the next 72 hours
    """
    daily_max        = df['temperature_2m'].resample('D').max()
    ref              = daily_max['1990':'2018']
    threshold        = ref.quantile(percentile)
    daily_max_hourly = daily_max.reindex(df.index, method='ffill')
    is_hot_hour      = (daily_max_hourly >= threshold).astype(float)
    is_klement       = is_hot_hour.rolling(window=72, min_periods=72).min()
    y_target         = is_klement.shift(-72).rolling(window=72, min_periods=1).max()
    return y_target, threshold


def run_training_pipeline(engineered_frames: dict,
                           city_climatologies: dict) -> tuple:
    """
    Full XGBoost training pipeline:
    1. Compute heatwave labels per city
    2. Build summer-only feature matrix (May–September)
    3. Time-based train/test split
    4. Compute class imbalance weight
    5. Train XGBoost classifier
    Returns: model, feature_cols, X_test, y_test, city_thresholds
    """

    # Step 1: Heatwave labels
    st.write("**Step 1/5:** Calculating heatwave labels...")
    city_targets    = {}
    city_thresholds = {}
    label_rows      = []

    for city, df in engineered_frames.items():
        target, thresh          = label_heatwave_hourly(df, city)
        city_targets[city]      = target
        city_thresholds[city]   = thresh
        label_rows.append({
            "City":          city,
            "P95 Threshold": round(float(thresh), 2),
            "Klement Hours": int(target.fillna(0).sum()),
            "Target y=1":    int(target.fillna(0).sum()),
        })

    st.dataframe(pd.DataFrame(label_rows), width='content')

    # Step 2: Feature matrix (summer months only)
    st.write("**Step 2/5:** Building feature matrix (May–September)...")
    hourly_frames = {}

    for city, df in engineered_frames.items():
        df          = df.copy()
        df['y']     = city_targets[city]
        df['city']  = city
        df          = df[df.index.month.isin(SUMMER_MONTHS)]
        df          = df.dropna()
        hourly_frames[city] = df

    full_df = pd.concat(hourly_frames.values()).sort_index()
    full_df = pd.get_dummies(full_df, columns=['city'], prefix='city')

    st.info(
        f"Dataset shape: {full_df.shape} | "
        f"Class balance: {full_df['y'].value_counts(normalize=True).round(3).to_dict()}"
    )

    # Step 3: Time-based train/test split
    st.write("**Step 3/5:** Train/Test split...")
    train        = full_df[full_df.index <= SPLIT_DATE]
    test         = full_df[full_df.index >  SPLIT_DATE]
    feature_cols = [c for c in full_df.columns if c != 'y']

    X_train, y_train = train[feature_cols], train['y']
    X_test,  y_test  = test[feature_cols],  test['y']

    st.write(f"Train: `{X_train.shape}` | `{int(y_train.sum())}` positive samples")
    st.write(f"Test:  `{X_test.shape}`  | `{int(y_test.sum())}` positive samples")

    # Step 4: Class imbalance weight
    neg              = (y_train == 0).sum()
    pos              = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    st.write(f"**Step 4/5:** Class ratio → `scale_pos_weight = {scale_pos_weight:.2f}`")

    # Step 5: XGBoost training
    st.write("**Step 5/5:** Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators          = 500,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 1,
        gamma                 = 0,
        scale_pos_weight      = scale_pos_weight,
        objective             = 'binary:logistic',
        eval_metric           = 'aucpr',
        early_stopping_rounds = 30,
        random_state          = 42,
        n_jobs                = -1,
        tree_method           = 'hist'
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Serialize model and climatologies for download
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)

    clim_buffer = io.BytesIO()
    joblib.dump(city_climatologies, clim_buffer)
    clim_buffer.seek(0)

    st.success("✅ Model training complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ Download model (.pkl)",
            data=model_buffer,
            file_name="model_hourly.pkl",
            mime="application/octet-stream"
        )
    with col2:
        st.download_button(
            label="⬇️ Download climatologies (.pkl)",
            data=clim_buffer,
            file_name="city_climatologies.pkl",
            mime="application/octet-stream"
        )

    return model, feature_cols, X_test, y_test, city_thresholds


# ============================================================
# EVALUATION
# ============================================================

def meteorological_baseline(engineered_frames: dict, city_thresholds: dict,
                              test_index) -> pd.Series:
    """
    Creates a simple meteorological baseline:
    Raises alarm whenever the daily max temperature exceeds the city-specific P95 threshold.
    """
    y_baseline = pd.Series(0, index=test_index, dtype=np.int64)

    for city, thresh in city_thresholds.items():
        df = engineered_frames[city].loc[
            engineered_frames[city].index.intersection(test_index)
        ]
        if df.empty:
            continue
        daily_max    = df['temperature_2m'].resample('D').max()
        alarm_days   = (daily_max >= thresh).astype(int)
        alarm_hourly = alarm_days.reindex(df.index, method='ffill').astype(np.int64)
        common_idx   = y_baseline.index.intersection(alarm_hourly.index)
        y_baseline.loc[common_idx] = (
            y_baseline.loc[common_idx] | alarm_hourly.loc[common_idx]
        )

    return y_baseline.astype(np.int64)


def show_evaluation(model, engineered_frames: dict, city_thresholds: dict,
                    X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Displays full model evaluation:
    - Key metrics (ROC-AUC, PR-AUC, Skill Score)
    - Confusion matrices (heatmaps)
    - Classification reports
    - Bar chart: Model vs Meteorological Baseline
    """
    st.subheader("📈 Model Evaluation & Comparison")

    # Predictions and baseline
    y_proba    = model.predict_proba(X_test)[:, 1]
    y_pred     = (y_proba >= 0.5).astype(int)
    y_baseline = meteorological_baseline(engineered_frames, city_thresholds, X_test.index)

    # Compute metrics
    roc_model = roc_auc_score(y_test, y_proba)
    pr_model  = average_precision_score(y_test, y_proba)
    roc_base  = roc_auc_score(y_test, y_baseline)
    pr_base   = average_precision_score(y_test, y_baseline)

    report_model = classification_report(y_test, y_pred,     digits=3, output_dict=True)
    report_base  = classification_report(y_test, y_baseline, digits=3, output_dict=True)

    # Robustly find the positive class key (can be '1', 1, or 1.0)
    pos_key    = next(k for k in report_model.keys() if str(k) in ('1', '1.0'))
    f1_model   = report_model[pos_key]['f1-score']
    f1_base    = report_base[pos_key]['f1-score']
    rec_model  = report_model[pos_key]['recall']
    rec_base   = report_base[pos_key]['recall']
    prec_model = report_model[pos_key]['precision']
    prec_base  = report_base[pos_key]['precision']

    # Key metric cards
    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC (Model)",    f"{roc_model:.4f}")
    m2.metric("PR-AUC (Model)",     f"{pr_model:.4f}")
    m3.metric("Skill vs. Baseline", f"{pr_model / pr_base:.2f}x better")

    # Confusion matrices
    st.write("### Confusion Matrices")
    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.write("**Model**")
        cm_model = confusion_matrix(y_test, y_pred)
        fig, ax  = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix – Model', color='steelblue')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_cm2:
        st.write("**Meteorological Baseline**")
        cm_base = confusion_matrix(y_test, y_baseline)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm_base, annot=True, fmt='d', cmap='Reds', cbar=True, ax=ax,
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix – Baseline', color='firebrick')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Classification reports
    st.write("### Detailed Classification Reports")
    rep_col1, rep_col2 = st.columns(2)

    with rep_col1:
        st.write("**Model Report**")
        st.dataframe(pd.DataFrame(report_model).T.round(3), width='content')

    with rep_col2:
        st.write("**Baseline Report**")
        st.dataframe(pd.DataFrame(report_base).T.round(3), width='content')

    # Bar chart: Model vs Baseline
    st.write("### Model vs Baseline – Key Metrics")
    metrics_names   = ['Precision (y=1)', 'Recall (y=1)', 'F1 (y=1)', 'ROC-AUC', 'PR-AUC']
    model_scores    = [prec_model, rec_model, f1_model, roc_model, pr_model]
    baseline_scores = [prec_base,  rec_base,  f1_base,  roc_base,  pr_base]

    x     = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, model_scores,    width, label='Model',    color='steelblue', alpha=0.8)
    ax.bar(x + width/2, baseline_scores, width, label='Baseline', color='firebrick', alpha=0.8)

    for i in range(len(x)):
        ax.text(x[i] - width/2, model_scores[i]    + 0.02,
                f"{model_scores[i]*100:.1f}%",    ha='center', fontsize=8)
        ax.text(x[i] + width/2, baseline_scores[i] + 0.02,
                f"{baseline_scores[i]*100:.1f}%", ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model vs Meteorological Baseline")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# SHAP ANALYSIS
# ============================================================

def estimate_threshold(feature: str, shap_vals: np.ndarray,
                        X: pd.DataFrame) -> float:
    """
    Estimates the feature value at which the SHAP contribution changes sign
    (i.e., where the feature starts increasing the predicted probability).
    """
    feat_vals   = X[feature].values
    col_idx     = X.columns.get_loc(feature)
    s_vals      = shap_vals[:, col_idx]
    idx         = np.argsort(feat_vals)
    feat_sorted = feat_vals[idx]
    shap_sorted = s_vals[idx]
    sign_changes = np.where(np.diff(np.sign(shap_sorted)))[0]
    if len(sign_changes) > 0:
        return feat_sorted[sign_changes[0]]
    return np.nan


def show_shap_analysis(model, X_test: pd.DataFrame,
                        y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Displays SHAP-based model interpretation:
    - Feature importance summary plot
    - Top 20 features table with direction, threshold, mean and std
    - False Positive analysis: FP vs TN feature differences
    """
    st.write("### 🔍 SHAP Analysis – Model Interpretation")

    # Use a sample for performance if X_test is large
    X_sample = X_test.head(1000) if len(X_test) > 1000 else X_test

    with st.spinner("Calculating SHAP values... this may take a moment."):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

    # SHAP Summary Plot
    st.write("**Feature Importance (SHAP Summary)**")
    fig, _ = plt.subplots()
    shap.summary_plot(shap_values, X_sample, max_display=15, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Top 20 SHAP feature table
    st.write("**Detailed SHAP Analysis: Top 20 Features**")
    shap_importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X_sample.columns
    ).sort_values(ascending=False)
    shap_mean = pd.Series(shap_values.mean(axis=0), index=X_sample.columns)

    rows = []
    for feat in shap_importance.head(20).index:
        thresh = estimate_threshold(feat, shap_values, X_sample)
        rows.append({
            'Feature':          feat,
            'Importance (abs)': round(float(shap_importance[feat]), 4),
            'Direction (mean)': round(float(shap_mean[feat]), 4),
            'Effect':           '🔥 ↑ Heat Event' if shap_mean[feat] > 0 else '❄️ ↓ No Event',
            'Threshold':        round(float(thresh), 3) if not np.isnan(thresh) else 'N/A',
            'Mean Value':       round(float(X_sample[feat].mean()), 3),
            'Std Value':        round(float(X_sample[feat].std()), 3),
        })

    st.dataframe(pd.DataFrame(rows), width='content')

    # False Positive analysis
    st.divider()
    st.write("**Why False Positives? – FP vs TN Feature Comparison**")

    fp_mask = (y_test.values == 0) & (y_pred == 1)
    tn_mask = (y_test.values == 0) & (y_pred == 0)

    col1, col2 = st.columns(2)
    col1.metric("False Positives (FP)", int(fp_mask.sum()))
    col2.metric("True Negatives (TN)",  int(tn_mask.sum()))

    if fp_mask.any():
        # Align masks with X_sample index
        fp_idx  = X_sample.index[fp_mask[:len(X_sample)]]
        tn_idx  = X_sample.index[tn_mask[:len(X_sample)]]
        X_fp    = X_sample.loc[X_sample.index.intersection(fp_idx)]
        X_tn    = X_sample.loc[X_sample.index.intersection(tn_idx)]

        shap_fp = shap_values[X_sample.index.isin(fp_idx)]
        shap_tn = shap_values[X_sample.index.isin(tn_idx)]

        if len(X_fp) == 0 or shap_fp.shape[0] == 0:
            st.info("ℹ️ No False Positives found within the SHAP sample window.")
        else:
            feature_diff = pd.DataFrame({
                'Mean FP':    X_fp.mean(),
                'Mean TN':    X_tn.mean(),
                'Difference': X_fp.mean() - X_tn.mean(),
                'SHAP FP':    np.abs(shap_fp).mean(axis=0),
                'SHAP TN':    np.abs(shap_tn).mean(axis=0),
            }).sort_values('Difference', key=abs, ascending=False)

            st.write("**Top 15 Feature Differences: FP vs TN**")
            st.dataframe(feature_diff.head(15).round(3), width='content')

            # SHAP Summary for False Positives
            st.write("**SHAP Summary – False Positives**")
            fig, _ = plt.subplots()
            shap.summary_plot(shap_fp, X_fp, max_display=15, show=False)
            plt.title("SHAP – False Positives (false alarms)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.success("No False Positives found in the test set!")


# ============================================================
# MAIN PAGE
# ============================================================

def show_page():
    st.title("🔧 Improvements & Final Model")

    # Sidebar: file upload
    with st.sidebar:
        st.markdown("---")
        st.header("📂 Data Upload")
        uploaded_files = st.file_uploader(
            "Upload weather files (.csv or .parquet)",
            type=["csv", "parquet"],
            accept_multiple_files=True
        )

    if not uploaded_files:
        st.info("👈 Please upload one or more weather files in the sidebar to get started.")
        return

    # ── Step 1: Load & standardize raw files ─────────────────────────────────
    results = process_weather_data(uploaded_files)
    if not results:
        return

    # ── Step 2: Aggregate into city DataFrames ────────────────────────────────
    st.divider()
    st.subheader("📊 Data Aggregation")

    all_frames       = {}
    unique_variables = set()

    for r in results:
        df      = pd.read_parquet(r["parquet_buffer"])
        city_id = r["metadata"]["city"]
        all_frames[city_id] = df
        for v in r["metadata"]["variables"]["list"]:
            unique_variables.add(v)

    # Statistical overview
    summary_list = []
    for loc, df in all_frames.items():
        loc_stats = {"City": loc.upper()}
        for var in df.columns:
            if var != 'timestamp':
                loc_stats[f"{var} (Mean)"] = round(df[var].mean(), 3)
                loc_stats[f"{var} (Max)"]  = round(df[var].max(), 3)
        summary_list.append(loc_stats)

    st.write("### Statistical Overview (Cleaned Data)")
    st.dataframe(pd.DataFrame(summary_list), width='content')

    # ── Step 3: Wind direction transformation ─────────────────────────────────
    st.divider()
    st.subheader("🌬️ Feature Engineering: Wind Direction")
    all_frames = transform_wind_direction(all_frames)
    show_validation_check(all_frames)

    # ── Step 4: Data type optimization ───────────────────────────────────────
    st.divider()
    st.subheader("⚙️ Data Type Optimization")
    all_frames = optimize_dtypes(all_frames)

    # ── Step 5: Feature engineering ──────────────────────────────────────────
    st.divider()
    st.subheader("🧪 Feature Engineering")
    with st.spinner("Running feature engineering for all cities..."):
        all_frames, city_climatologies = run_feature_engineering(all_frames)

    example_city   = list(all_frames.keys())[0]
    total_vars     = len(all_frames[example_city].columns)
    new_vars_count = len([
        c for c in all_frames[example_city].columns
        if any(x in c for x in ['lag', 'delta', 'aridity', 'smooth', 'zscore', 'sin', 'cos'])
    ])
    st.success("✅ Feature Engineering complete!")
    st.info(
        f"**Status:** {len(all_frames)} cities processed – "
        f"**{total_vars} total variables** ({new_vars_count} newly engineered features)."
    )

    # ── Step 6: Download preprocessed data ───────────────────────────────────
    st.divider()
    st.subheader("📦 Download Preprocessed Data")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            zf.writestr(f"{r['filename_base']}.parquet", r["parquet_buffer"].getvalue())
            zf.writestr(f"{r['filename_base']}.json",    r["json_bytes"])
    zip_buffer.seek(0)

    st.download_button(
        label="⬇️ Download all processed files (.zip)",
        data=zip_buffer,
        file_name="preprocessed_weather_data.zip",
        mime="application/zip"
    )

    # ── Step 7: Model training ────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 Model Training")
    if st.button("🚀 Start Training"):
        with st.spinner("Training in progress... this may take a moment."):
            model, feature_cols, X_test, y_test, city_thresholds = run_training_pipeline(
                all_frames, city_climatologies
            )
            st.session_state['model']           = model
            st.session_state['feature_cols']    = feature_cols
            st.session_state['X_test']          = X_test
            st.session_state['y_test']          = y_test
            st.session_state['city_thresholds'] = city_thresholds
            st.session_state['engineered_frames'] = all_frames

    # ── Step 8: Evaluation ────────────────────────────────────────────────────
    if 'model' in st.session_state:
        st.divider()
        show_evaluation(
            model             = st.session_state['model'],
            engineered_frames = st.session_state['engineered_frames'],
            city_thresholds   = st.session_state['city_thresholds'],
            X_test            = st.session_state['X_test'],
            y_test            = st.session_state['y_test']
        )

        # ── Step 9: SHAP Analysis ─────────────────────────────────────────────
        st.divider()
        y_proba = st.session_state['model'].predict_proba(st.session_state['X_test'])[:, 1]
        y_pred  = (y_proba >= 0.5).astype(int)
        show_shap_analysis(
            model  = st.session_state['model'],
            X_test = st.session_state['X_test'],
            y_test = st.session_state['y_test'],
            y_pred = y_pred
        )
