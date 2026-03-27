  """
Weather Prediction with Machine Learning
Multi-output regression + classification for temperature, rain, and conditions.
Install: pip install scikit-learn pandas numpy matplotlib seaborn xgboost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── 1. Generate Realistic Synthetic Weather Dataset ───────────────────────────
np.random.seed(42)
N_DAYS = 1461  # 4 years of daily data

dates = pd.date_range('2021-01-01', periods=N_DAYS, freq='D')

# Seasonal temperature baseline
day_of_year = np.arange(N_DAYS)
season_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

# Long-term trend + noise
trend = np.linspace(0, 0.8, N_DAYS)
noise = np.random.normal(0, 3, N_DAYS)

temp_max = season_temp + trend + noise + 5
temp_min = season_temp + trend + noise - 8
temp_avg = (temp_max + temp_min) / 2

# Humidity (correlated with temperature)
humidity = 70 - 0.4 * temp_avg + np.random.normal(0, 10, N_DAYS)
humidity = np.clip(humidity, 20, 100)

# Pressure
pressure = 1013 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5, N_DAYS)

# Wind
wind_speed = np.abs(np.random.normal(15, 8, N_DAYS))

# Precipitation (more in winter/spring)
rain_prob_base = 0.2 + 0.15 * np.sin(2*np.pi*(day_of_year - 20)/365)
raining = np.random.rand(N_DAYS) < rain_prob_base
precipitation = np.where(raining, np.random.exponential(5, N_DAYS), 0)

# Cloud cover
cloud_cover = 40 + 25 * np.sin(2*np.pi*(day_of_year - 15)/365) + np.random.normal(0, 15, N_DAYS)
cloud_cover = np.clip(cloud_cover, 0, 100)

# Weather condition label
def get_condition(temp, prec, cloud):
    if prec > 10: return 'Heavy Rain'
    if prec > 2:  return 'Rainy'
    if cloud > 75:return 'Cloudy'
    if cloud > 40:return 'Partly Cloudy'
    if temp > 28: return 'Hot & Sunny'
    return 'Clear'

df = pd.DataFrame({
    'date':          dates,
    'temp_max':      np.round(temp_max, 1),
    'temp_min':      np.round(temp_min, 1),
    'temp_avg':      np.round(temp_avg, 1),
    'humidity':      np.round(humidity, 1),
    'pressure':      np.round(pressure, 1),
    'wind_speed':    np.round(wind_speed, 1),
    'precipitation': np.round(precipitation, 1),
    'cloud_cover':   np.round(cloud_cover, 1),
})
df['condition'] = [get_condition(t, p, c)
                   for t, p, c in zip(df.temp_avg, df.precipitation, df.cloud_cover)]
df['month']     = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['week']      = df['date'].dt.isocalendar().week.astype(int)
df['year']      = df['date'].dt.year
print(f"Dataset: {len(df)} days | Conditions: {df.condition.value_counts().to_dict()}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
for lag in [1, 2, 3, 7]:
    df[f'temp_lag{lag}']  = df['temp_avg'].shift(lag)
    df[f'hum_lag{lag}']   = df['humidity'].shift(lag)
    df[f'prec_lag{lag}']  = df['precipitation'].shift(lag)
    df[f'press_lag{lag}'] = df['pressure'].shift(lag)

for window in [3, 7, 14]:
    df[f'temp_roll{window}']  = df['temp_avg'].rolling(window).mean()
    df[f'prec_roll{window}']  = df['precipitation'].rolling(window).sum()
    df[f'hum_roll{window}']   = df['humidity'].rolling(window).mean()
    df[f'press_roll{window}'] = df['pressure'].rolling(window).mean()

df['temp_range']  = df['temp_max'] - df['temp_min']
df['pressure_chg'] = df['pressure'].diff()
df['sin_month']   = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month']   = np.cos(2 * np.pi * df['month'] / 12)
df['sin_doy']     = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['cos_doy']     = np.cos(2 * np.pi * df['day_of_year'] / 365)

df = df.dropna().reset_index(drop=True)

# ── 3. Regression: Predict tomorrow's temp_avg ────────────────────────────────
reg_features = [c for c in df.columns if c not in
                ['date', 'temp_avg', 'temp_max', 'temp_min', 'condition']]
X_reg = df[reg_features]
y_reg = df['temp_avg']

# Time-series split
split_idx = int(len(df) * 0.8)
X_train_r, X_test_r = X_reg.iloc[:split_idx], X_reg.iloc[split_idx:]
y_train_r, y_test_r = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
scaler = StandardScaler()
X_train_rs = scaler.fit_transform(X_train_r)
X_test_rs  = scaler.transform(X_test_r)

reg_models = {
    'Random Forest':    RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
    'Gradient Boost':   GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'Ridge':            Ridge(alpha=1.0),
}
try:
    from xgboost import XGBRegressor
    reg_models['XGBoost'] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
except ImportError:
    pass

reg_results = {}
for name, model in reg_models.items():
    model.fit(X_train_rs, y_train_r)
    preds = model.predict(X_test_rs)
    mae   = mean_absolute_error(y_test_r, preds)
    r2    = r2_score(y_test_r, preds)
    reg_results[name] = {'preds': preds, 'mae': mae, 'r2': r2}
    print(f"{name:20s}  MAE={mae:.2f}°C  R²={r2:.4f}")

best_reg = min(reg_results, key=lambda k: reg_results[k]['mae'])
print(f"\n🏆 Best regressor: {best_reg} (MAE={reg_results[best_reg]['mae']:.2f}°C)")

# ── 4. Classification: Predict weather condition ──────────────────────────────
le  = LabelEncoder()
y_cls = le.fit_transform(df['condition'])
X_cls = df[reg_features]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42)
X_train_cs = scaler.fit_transform(X_train_c)
X_test_cs  = scaler.transform(X_test_c)

clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
clf.fit(X_train_cs, y_train_c)
cls_preds = clf.predict(X_test_cs)
cls_acc   = (cls_preds == y_test_c).mean()
print(f"\nCondition Classifier Accuracy: {cls_acc*100:.1f}%")
print(classification_report(y_test_c, cls_preds, target_names=le.classes_))

# ── 5. Visualizations ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Weather Prediction Dashboard', fontsize=16, fontweight='bold')

# Time series: actual vs predicted temperature
test_dates = df['date'].iloc[split_idx:].values
axes[0,0].plot(test_dates[:120], y_test_r.values[:120], label='Actual', alpha=0.8)
axes[0,0].plot(test_dates[:120], reg_results[best_reg]['preds'][:120],
               '--', label='Predicted', alpha=0.8)
axes[0,0].set_title(f'Temp Prediction — {best_reg}'); axes[0,0].legend()
axes[0,0].tick_params(axis='x', rotation=30)

# Scatter: predicted vs actual
axes[0,1].scatter(y_test_r, reg_results[best_reg]['preds'], alpha=0.3, s=10)
lo, hi = y_test_r.min(), y_test_r.max()
axes[0,1].plot([lo, hi], [lo, hi], 'r--'); axes[0,1].set_xlabel('Actual °C')
axes[0,1].set_ylabel('Predicted °C'); axes[0,1].set_title('Actual vs Predicted')

# Error distribution
errors = y_test_r.values - reg_results[best_reg]['preds']
axes[0,2].hist(errors, bins=40, color='steelblue', edgecolor='white')
axes[0,2].axvline(0, color='red', ls='--')
axes[0,2].set_title('Prediction Error Distribution'); axes[0,2].set_xlabel('Error (°C)')

# Model comparison
names = list(reg_results.keys())
maes  = [reg_results[n]['mae'] for n in names]
r2s   = [reg_results[n]['r2']  for n in names]
x     = np.arange(len(names))
axes[1,0].bar(x, maes, color='coral'); axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(names, rotation=15); axes[1,0].set_title('MAE by Model')
for i, v in enumerate(maes): axes[1,0].text(i, v + 0.02, f'{v:.2f}', ha='center')

# Monthly average temperature
monthly = df.groupby('month')['temp_avg'].mean()
axes[1,1].bar(monthly.index, monthly.values,
              color=plt.cm.coolwarm(np.linspace(0, 1, 12)))
axes[1,1].set_xticks(range(1,13))
axes[1,1].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
axes[1,1].set_title('Monthly Avg Temperature'); axes[1,1].set_ylabel('°C')

# Condition distribution
cond_counts = df['condition'].value_counts()
colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4']
axes[1,2].pie(cond_counts, labels=cond_counts.index, autopct='%1.0f%%', colors=colors)
axes[1,2].set_title('Weather Condition Distribution')

# Feature importance
if hasattr(list(reg_models.values())[0], 'feature_importances_'):
    rf_model = reg_models['Random Forest']
    feat_imp = pd.Series(rf_model.feature_importances_, index=reg_features).nlargest(12)
    feat_imp.plot(kind='barh', ax=axes[2,0], color='teal')
    axes[2,0].set_title('Top 12 Feature Importances (RF Regressor)')

# Temp vs Humidity scatter
scatter = axes[2,1].scatter(df['temp_avg'], df['humidity'], c=df['precipitation'],
                             cmap='Blues', alpha=0.3, s=5)
plt.colorbar(scatter, ax=axes[2,1])
axes[2,1].set_xlabel('Temp (°C)'); axes[2,1].set_ylabel('Humidity (%)')
axes[2,1].set_title('Temp vs Humidity (color=precipitation)')

# Condition confusion matrix
cm = pd.crosstab(le.inverse_transform(y_test_c),
                 le.inverse_transform(cls_preds), rownames=['True'], colnames=['Pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[2,2])
axes[2,2].set_title('Weather Condition Confusion Matrix')
axes[2,2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('weather_prediction_results.png', dpi=150)
print("📊 Saved: weather_prediction_results.png")

# ── 6. 7-day Forecast ──────────────────────────────────────────────────────────
def forecast_next_7_days(model, scaler, df):
    """Simple iterative 7-day temperature forecast."""
    last_row = df[reg_features].iloc[-1:].copy()
    forecast = []
    for day in range(1, 8):
        scaled = scaler.transform(last_row)
        pred   = model.predict(scaled)[0]
        forecast.append(pred)
        last_row = last_row.copy()
        for lag in [1, 2, 3, 7]:
            if f'temp_lag{lag}' in last_row.columns:
                last_row[f'temp_lag{lag}'] = df['temp_avg'].iloc[-(lag+1)]
    return forecast

best_model = reg_models[best_reg]
forecast = forecast_next_7_days(best_model, scaler, df)
print("\n── 7-Day Forecast ──")
for i, t in enumerate(forecast, 1):
    bar = '█' * int(t / 2)
    print(f"  Day +{i}: {t:5.1f}°C  {bar}")
  
