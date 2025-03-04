# Optimizing E-Commerce Delivery Performance Using Data Analysis

This project analyzes e-commerce delivery performance using Python, PostgreSQL, and Power BI to extract insights and provide actionable business recommendations.

## Files
- `ecommerce_analysis.py`: Python script for data cleaning, analysis, and modeling.
- `cleaned_data.csv`: Processed dataset with delivery times and estimates.
- `state_delivery.csv`: Average delivery times by state.
- `anomalies.csv`: Top 10 longest deliveries.
- `predictions.csv`: Random Forest model predictions.
- `delivery_time_histogram.png`: Histogram of delivery times.
- `delivery_time_boxplot.png`: Boxplot of delivery times by state.
- `Delivery_Performance_Dashboard.pbix`: Power BI dashboard visualizing the analysis.

## Usage
1. Run `ecommerce_analysis.py` to generate CSVs and PNGs (requires Python, pandas, sklearn, seaborn, psycopg2).
2. Open `Delivery_Performance_Dashboard.pbix` in Power BI Desktop to view the interactive dashboard.

## Requirements
- Python 3.x
- PostgreSQL
- Power BI Desktop