 Air Cargo Data Processing & Risk Analysis (PySpark Project)
 Project Overview:

This project processes air cargo shipment data using PySpark and derives meaningful insights such as approval risk levels.

The goal is to simulate a real-world aviation data pipeline where shipment approval status is analyzed and transformed into actionable intelligence.

Technologies Used:
Python
PySpark
Google Colab
📂 Dataset

The dataset contains the following fields:

AWB Number
Shipment ID
Export Origin
Destination
Approval Status
Cargo Name
Weight

Data is stored in a pipe (|) delimited text file.

How It Works
1. Initialize Spark Session

Creates a Spark environment for processing large-scale data.

2. Load Data

Reads the dataset using PySpark:

spark.read.option("delimiter", "|").option("header", True).csv("file_path")
3. Feature Engineering
 Approval Risk Calculation

A custom logic is applied:

REJECTED → High Risk (3)
PENDING → Medium Risk (2)
APPROVED → Low Risk (1)

This is implemented using a User Defined Function (UDF).

 Weight Classification:

Shipments are classified as:

Heavy → Weight > 300
Light → Weight ≤ 300
Output:

The processed dataset includes:

Original shipment data
Derived column: approval_risk
Derived column: is_heavy
Use Cases 
Logistics risk analysis
Shipment approval monitoring
Aviation data insights
Preprocessing for ML models
Future Enhancements 
Build ML model to predict approval status
Deploy as REST API using Flask/FastAPI
Create UI dashboard for visualization

Author:
G.Madhu Mathi

Notes:

This project demonstrates:

Data processing using PySpark
Feature engineering
Real-world business logic implementation
