# ML-Portfolio

This repository implements a "Credit Card Fraud Detection System" built with a modular machine learning architecture - from "data generation and feature engineering" to "model training, experiment tracking, and real-time serving".

It demonstrates end-to-end orchestration of a production-grade ML pipeline using open source technologies - Pytorch, Scikit-learn, XGBoost, MLflow, Grafana, FastAPI.

# Overview 
The system predicts fraudulent transactions using both synthetic and file based datasets. 

A clean ML system design is followed in this project.
1.Data ingestion - synthetic or real datasets
2.Feature engineering - time, frequency, and categorical transformations
3.Model trainig - orchestration via a YAML config driven pipeline
4.Tracking - MLflow integration
5.Model serving - FastAPI endpoint for online predictions
6.Monitoring - Grafana, Prometheus, and logs directly

# Tech Stack
Language - Python3.10+ Core development
ML Frameworks - XGBoost, Scikit-learn (Modeling and evaluation)
Data Handling - Pandas, Numpy ( Data transformation)
Logging - Custom logger.py (Unified file + console logs)
Tracking - MLflow
Serving - FASTAPI, Uvicorn
Monitoring - Grafana, Prometheus (Metrics visualisation)
Infrastructure - Docker compose, Kafka (Orchestration and streaming simulation)
