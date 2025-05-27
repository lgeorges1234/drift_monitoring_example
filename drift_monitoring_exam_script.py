import evidently
import datetime
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import json
import logging

from sklearn import datasets, ensemble, model_selection
from scipy.stats import anderson_ksamp

from evidently.metrics import RegressionQualityMetric, RegressionErrorPlot, RegressionErrorDistribution
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.ui.workspace import Workspace

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def _fetch_data() -> pd.DataFrame:
    content = requests.get("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", verify=False).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday']) 
    return raw_data

def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.index = raw_data.apply(lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)), axis=1)
    return raw_data

def generate_regression_performance_report(reference_data, current_data, metrics, column_mapping=None):
    """
    Generates a regression performance report using Evidently.
    """
    regression_performance_report = Report(metrics=metrics)
    regression_performance_report.run(
        reference_data=reference_data.sort_index() if reference_data is not None else None, 
        current_data=current_data.sort_index(),
        column_mapping=column_mapping
    )
    return regression_performance_report

def add_report_to_workspace(workspace, project_name, project_description, report, report_name=None):
    """
    Adds a report to an existing or new project in a workspace.
    """
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # Add report to the project
    workspace.add_report(project.id, report)
    report_desc = f" ({report_name})" if report_name else ""
    print(f"New report{report_desc} added to project {project_name}")

if __name__ == "__main__":
    # set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting drift monitoring script")

    WORKSPACE_NAME = "datascientest-workspace"
    BASE_PROJECT_NAME = "bike_sharing_monitoring"
    PROJECT_DESCRIPTION = "Exam - Drift Monitoring Dashboards for Bike Sharing Dataset"

    workspace = Workspace.create(WORKSPACE_NAME)
    logger.info(f"Workspace '{WORKSPACE_NAME}' created")

    #######################################################
    # STEP 1: Fetch and process the data
    #######################################################
    logger.info("Fetching and processing data")
    raw_data = _process_data(_fetch_data())

    # Feature selection
    target = 'cnt'
    prediction = 'prediction'
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
    categorical_features = ['season', 'holiday', 'workingday']

    # Reference and current data split
    reference_jan11 = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
    current_feb11 = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']

    # Train test split ONLY on reference_jan11
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        reference_jan11[numerical_features + categorical_features],
        reference_jan11[target],
        test_size=0.3
    )

    # Model training
    regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
    regressor.fit(X_train, y_train)

    logger.info("Step 1 completed successfully")

    #######################################################
    # STEP 2: Model validation with RegressionPreset
    #######################################################
    logger.info("Validating model with RegressionPreset")

    # Predictions
    preds_train = regressor.predict(X_train)
    preds_test = regressor.predict(X_test)

    # Add actual target and prediction columns to the training data
    X_train = X_train.copy()
    X_train['target'] = y_train
    X_train['prediction'] = preds_train

    # Add actual target and prediction columns to the test data
    X_test = X_test.copy()
    X_test['target'] = y_test
    X_test['prediction'] = preds_test

    # Initialize the column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = 'target'
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    # Generate the regression performance report
    regression_report = generate_regression_performance_report(
        reference_data=X_train,
        current_data=X_test,
        metrics=[RegressionPreset()],
        column_mapping=column_mapping
    )

    # Add report to workspace
    add_report_to_workspace(workspace, f"{BASE_PROJECT_NAME}_model_validation", PROJECT_DESCRIPTION, regression_report, "Model Validation")
    logger.info("Step 2 completed successfully")

    #######################################################
    # STEP 3: Production model on January data
    #######################################################
    logger.info("Building production model on whole January dataset")

    # Train production model on full January data
    production_regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
    production_regressor.fit(
        reference_jan11[numerical_features + categorical_features], 
        reference_jan11[target]
    )

    # Make predictions on January data
    ref_predictions = production_regressor.predict(
        reference_jan11[numerical_features + categorical_features]
    )

    # Prepare reference data with predictions
    reference_jan11_with_predictions = reference_jan11.copy()
    reference_jan11_with_predictions['prediction'] = ref_predictions

    # Column mapping for production
    production_column_mapping = ColumnMapping()
    production_column_mapping.target = target
    production_column_mapping.prediction = prediction
    production_column_mapping.numerical_features = numerical_features
    production_column_mapping.categorical_features = categorical_features

    # Generate production model report
    production_regression_report = generate_regression_performance_report(
        reference_data=None,
        current_data=reference_jan11_with_predictions,
        metrics=[RegressionPreset()],
        column_mapping=production_column_mapping
    )

    # Add report to workspace
    add_report_to_workspace(workspace, f"{BASE_PROJECT_NAME}_production_model", PROJECT_DESCRIPTION, production_regression_report, "Production Model Performance")
    logger.info("Step 3 completed successfully")

    #######################################################
    # STEP 4: Weekly drift monitoring reports
    #######################################################
    logger.info("Generating weekly drift monitoring reports")
    
    weeks = {
        'week_1': ('2011-01-29 00:00:00', '2011-02-07 23:00:00'),
        'week_2': ('2011-02-07 00:00:00', '2011-02-14 23:00:00'),
        'week_3': ('2011-02-15 00:00:00', '2011-02-21 23:00:00')
    }

    weekly_performances = {}  # Store performance metrics for each week

    for week_name, (start_date, end_date) in weeks.items():
        logger.info(f"Processing {week_name} data from {start_date} to {end_date}")

        # Filter data for the week
        current_week_data = current_feb11.loc[start_date:end_date].copy()

        # Make predictions
        current_week_predictions = production_regressor.predict(
            current_week_data[numerical_features + categorical_features]
        )

        # Prepare data with predictions
        current_week_data_with_predictions = current_week_data.copy()
        current_week_data_with_predictions['prediction'] = current_week_predictions

        # Generate regression report
        weekly_regression_report = generate_regression_performance_report(
            reference_data=reference_jan11_with_predictions,
            current_data=current_week_data_with_predictions,
            metrics=[RegressionPreset()],
            column_mapping=production_column_mapping
        )

        # Store performance data for later analysis
        weekly_performances[week_name] = {
            'data': current_week_data_with_predictions,
            'report': weekly_regression_report
        }

        # Add report to workspace with clear project naming
        project_name = f"{BASE_PROJECT_NAME}_weekly_monitoring"
        add_report_to_workspace(workspace, project_name, f"{PROJECT_DESCRIPTION} - Weekly Analysis", weekly_regression_report, f"{week_name.replace('_', ' ').title()} Performance")
        logger.info(f"{week_name} report added to workspace")

    logger.info("Step 4 completed successfully")

    #######################################################
    # STEP 5: Target drift analysis for worst week
    #######################################################
    logger.info("Analyzing target drift for worst performing week")

    # Determine worst performing week by extracting RMSE from reports
    week_performance_metrics = {}
    
    for week_name, week_data in weekly_performances.items():
        report_dict = week_data['report'].as_dict()
        
        # Navigate through the report structure to find RMSE
        # RegressionPreset contains RegressionQualityMetric
        for metric in report_dict['metrics']:
            if metric['metric'] == 'RegressionQualityMetric':
                current_rmse = metric['result']['current']['rmse']
                week_performance_metrics[week_name] = current_rmse
                logger.info(f"{week_name} RMSE: {current_rmse:.4f}")
                break
    
    # Find the week with highest RMSE (worst performance)
    worst_week = max(week_performance_metrics.keys(), key=lambda k: week_performance_metrics[k])
    worst_rmse = week_performance_metrics[worst_week]
    
    logger.info(f"Worst performing week: {worst_week} with RMSE: {worst_rmse:.4f}")
    
    worst_week_data = weekly_performances[worst_week]['data']

    # Target drift report
    target_drift_report = generate_regression_performance_report(
        reference_data=reference_jan11_with_predictions,
        current_data=worst_week_data,
        metrics=[TargetDriftPreset()],
        column_mapping=production_column_mapping
    )

    # Add report to workspace
    add_report_to_workspace(workspace, f"{BASE_PROJECT_NAME}_target_analysis", f"{PROJECT_DESCRIPTION} - Target Drift", target_drift_report, f"Target Drift - {worst_week.replace('_', ' ').title()}")
    logger.info("Step 5 completed successfully")

    #######################################################
    # STEP 6: Data drift analysis for last week (numerical only)
    #######################################################
    logger.info("Analyzing data drift for last week (numerical features only)")

    # Use week 3 data (last week)
    last_week_data = weekly_performances['week_3']['data']

    # Column mapping for numerical features only
    numerical_only_mapping = ColumnMapping()
    numerical_only_mapping.target = target
    numerical_only_mapping.prediction = prediction
    numerical_only_mapping.numerical_features = numerical_features
    numerical_only_mapping.categorical_features = []  # Empty list for numerical only

    # Data drift report for numerical features only
    data_drift_report = generate_regression_performance_report(
        reference_data=reference_jan11_with_predictions,
        current_data=last_week_data,
        metrics=[DataDriftPreset()],
        column_mapping=numerical_only_mapping
    )

    # Add report to workspace
    add_report_to_workspace(workspace, f"{BASE_PROJECT_NAME}_data_drift", f"{PROJECT_DESCRIPTION} - Data Drift", data_drift_report, "Week 3 Data Drift (Numerical Features)")
    logger.info("Step 6 completed successfully")

    logger.info("All reports generated successfully! Check the Evidently UI for results.")
    logger.info("Projects created:")
    logger.info(f"1. {BASE_PROJECT_NAME}_model_validation - Model validation report")
    logger.info(f"2. {BASE_PROJECT_NAME}_production_model - Production model performance")
    logger.info(f"3. {BASE_PROJECT_NAME}_weekly_monitoring - All 3 weekly reports")
    logger.info(f"4. {BASE_PROJECT_NAME}_target_analysis - Target drift analysis")
    logger.info(f"5. {BASE_PROJECT_NAME}_data_drift - Feature drift analysis")