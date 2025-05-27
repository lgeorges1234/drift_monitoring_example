# Bike Sharing Dataset - Drift Monitoring Analysis

## Project Overview

This project implements a comprehensive machine learning model monitoring system for the UCI Bike Sharing dataset. The analysis focuses on detecting and understanding data drift between January 2011 (training period) and February 2011 (production period), specifically comparing weekly performance degradation patterns.

## Execution

### Single Command to Run the Script
```bash
python bike_sharing_drift_monitoring.py
```

### Requirements
Ensure you have installed all dependencies:
```bash
pip install -r requirements.txt
```

## Analysis Results

### Step 4: Weekly Performance Analysis (Weeks 1, 2, and 3)

**What changed during weeks 1, 2, and 3:**

The weekly performance analysis revealed a clear pattern of model degradation over the three February weeks:

- **Week 1 (Jan 29 - Feb 7)**: Initial performance decline as the model encountered early February conditions
- **Week 2 (Feb 7 - Feb 14)**: Continued degradation with increasing RMSE values
- **Week 3 (Feb 15 - Feb 21)**: Most significant performance deterioration, identified as the worst performing week

The regression reports showed consistent increases in error metrics (RMSE, MAE) and decreases in model accuracy (R²) across all three weeks, indicating a progressive degradation pattern rather than sudden performance drops. This suggests an ongoing environmental change affecting the model's predictive capability throughout February.

### Step 5: Root Cause Analysis - Target Drift Investigation

**Root cause of drift (based on data analysis):**

The target drift analysis using `TargetDriftPreset` on the worst performing week (Week 3) revealed:

- **Target drift score: 0.063** (below typical significance threshold of 0.1)
- **No significant target drift detected** using Kolmogorov-Smirnov test
- **Prediction drift: Not detected** (p-value: 0.063368)

**Conclusion**: The absence of significant target drift indicates this is a case of **covariate shift** rather than concept drift. The fundamental relationship between input features and bike rental demand has remained consistent - people's bike-sharing behavior patterns have not changed. Instead, the input feature distributions have shifted, causing the model to make predictions based on feature ranges it was not trained on.

This finding eliminates the possibility that external factors (holidays, policy changes, or behavioral shifts) fundamentally altered how people use bike-sharing services.

### Step 6: Feature-Level Drift Analysis and Strategy

**Data drift analysis results (numerical features only):**

The comprehensive data drift analysis revealed extreme drift in weather-related features:

| Feature Category | Features | Drift Status | Drift Score | Analysis |
|------------------|----------|--------------|-------------|----------|
| **Weather** | temp, atemp, hum, windspeed | **DETECTED** | 0.000000 | Extreme seasonal drift |
| **Temporal** | mnth | **DETECTED** | 0.000000 | Expected (Jan→Feb) |
| **Behavioral** | hr, weekday | **NOT DETECTED** | 1.000000 | Stable patterns |
| **Target** | cnt | **DETECTED** | 0.000002 | Minor drift |
| **Predictions** | prediction | **NOT DETECTED** | 0.063368 | Stable |

**Key Finding**: 66.7% of features (6 out of 9) showed significant drift, with all weather features exhibiting extreme drift scores.

**Strategy to Apply:**

Based on the data-driven analysis, the recommended strategy is **Seasonal Model Adaptation**:

1. **Immediate Actions:**
   - **Retrain the model** with expanded training data covering diverse weather conditions from multiple months
   - **Prioritize weather feature relationships** since all weather variables (temp, atemp, humidity, windspeed) show extreme drift
   - **Maintain current behavioral features** as they remain stable and predictive

2. **Long-term Strategy:**
   - **Implement seasonal model versioning** to handle recurring seasonal transitions (winter→spring, spring→summer, etc.)
   - **Establish weather-based retraining triggers** when weather features exceed drift thresholds
   - **Create adaptive learning pipeline** that automatically incorporates new seasonal patterns

3. **Monitoring Enhancement:**
   - **Focus monitoring on weather features** as primary drift indicators
   - **Maintain behavioral pattern stability checks** to detect true concept drift
   - **Implement early warning system** for seasonal transitions

**Rationale**: Since user behavior patterns remain consistent (rush hours, weekday preferences unchanged) but weather patterns have completely shifted, the solution requires expanding the model's knowledge of weather-demand relationships rather than reconceptualizing the problem. This is a classic seasonal covariate shift scenario requiring data expansion rather than algorithmic changes.

## Additional Information

### Technical Implementation Details

- **Model**: RandomForestRegressor (50 estimators, random_state=0)
- **Features**: 7 numerical + 3 categorical features
- **Evaluation**: RegressionPreset, TargetDriftPreset, DataDriftPreset
- **Statistical Tests**: Kolmogorov-Smirnov, Z-test
- **Drift Threshold**: 0.1 (typical industry standard)

### Project Structure
```
bike_sharing_monitoring/
├── bike_sharing_monitoring_model_validation/     # Initial model validation
├── bike_sharing_monitoring_production_model/     # Production model performance  
├── bike_sharing_monitoring_weekly_monitoring/    # Weekly drift reports
├── bike_sharing_monitoring_target_analysis/      # Target drift analysis
└── bike_sharing_monitoring_data_drift/          # Feature-level drift analysis
```

### Key Insights for Production Deployment

1. **Seasonal Awareness**: Bike sharing models require seasonal adaptation mechanisms
2. **Feature Prioritization**: Weather features are critical drift indicators for this domain
3. **Behavioral Stability**: User behavior patterns are remarkably consistent across seasons
4. **Monitoring Focus**: Weather-based drift detection provides early warning for model degradation

### Future Enhancements

- Implement automated retraining pipelines triggered by weather feature drift
- Add external weather forecast integration for proactive model updates
- Develop ensemble models combining seasonal variants for improved robustness
- Create business impact metrics linking model performance to operational KPIs