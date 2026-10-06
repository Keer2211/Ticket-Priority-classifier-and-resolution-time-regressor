# Customer Support AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20VADER-green.svg)](https://www.nltk.org/)

**Eliminating Manual Support Bottlenecks: An End-to-End Dual-Stage Machine Learning Pipeline Leveraging VADER Sentiment Analysis for Priority Classification and Resolution Time Regression**

---

## 📌 Problem Statement & Solution

In traditional customer support workflows, manual ticket triage introduces operational delays and human error. Agents frequently miscalculate ticket urgency, leading to missed **Service Level Agreements (SLAs)**, frustrated customers, and costly escalations.

This repository implements a **dual-stage predictive machine learning pipeline** that automates ticket triage and estimates resolution times:

* **Sentiment & Feature Engineering:** Extracts compound sentiment scores using NLTK's VADER lexicon and calculates operational features like `purchase_age_days`.
* **Stage 1 — Priority Classification (`LinearSVC`):** Predicts urgency classes (`Low`, `Medium`, `High`, `Critical`) directly from ticket metadata and sentiment.
* **Stage 2 — Resolution Time Regression:** Feeds the classifier's predicted priority class into a log-transformed regressor to accurately forecast time-to-resolution in hours.
* **Interactive Dashboard:** Built with Streamlit for real-time ticket ingestion and predictive analytics visualization.

---

## 📂 Project Structure

```text
customer_support-ai/
├── data/                  # Source customer support ticket datasets
├── notebooks/             # Exploratory data analysis & model development
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_priority_classification.ipynb
│   ├── 05_resolution_time_regression.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_genai.ipynb
├── src/                   # Reusable preprocessing, sentiment, and ML modules
├── models/                # Serialized model artifacts (.pkl)
├── app/                   # Streamlit web interface
│   └── main.py
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

WORKFLOW ARCHITECTURE:

[ Raw Ticket Input ] ──► [ Text Normalization & VADER Sentiment ] ──► [ Feature Engineering ]
                                                                             │
┌────────────────────────────────────────────────────────────────────────────┘
▼
[ Stage 1: LinearSVC Classifier ] ──► (Predicted Priority)
                                                 │
┌────────────────────────────────────────────────┘
▼
[ Stage 2: Resolution Time Regressor ] ──► (Predicted Time to Resolution in Hours)


🚀 Getting Started
Prerequisites
Python 3.9 or higher

Git

1. Environment Setup
Clone the repository and activate your virtual environment:

PowerShell
# Navigate to project directory
cd customer_support-ai

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
2. Install Dependencies
PowerShell
pip install -r requirements.txt
3. Run Application
Launch the Streamlit web dashboard:

PowerShell
streamlit run app/main.py



































































<!-- # Customer Support AI

A workspace for understanding, preprocessing, modeling, and augmenting customer
support tickets with machine learning and generative AI.

## Project structure

- `data/` - source customer support ticket data
- `notebooks/` - analysis and modeling notebooks
- `src/` - reusable preprocessing, ML, sentiment, and LLM utilities
- `models/` - saved model artifacts
- `app/` - application entry point

## Getting started

1. Activate the existing virtual environment:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Run the scaffold application:

   ```powershell
   python app\main.py
   ```

Open the notebooks in order from `01_data_understanding.ipynb` through
`07_genai.ipynb` as the project is developed.


Learn while coding

value_counts()-gives frequency of each element in a DF.
Value_counts(bins=10)-this divides the range into 10 divisions
.value_counts(normalize=True) * 100 – gives percentage
sort_index()  sorts the index 

Project Title

Eliminating Manual Support Bottlenecks: An End-to-End Dual-Stage Machine Learning Pipeline Leveraging VADER Sentiment Analysis for Priority Classification and Resolution Time Regression

Problems Handled:

Escalations happen when the agent fails to set the priority according to the customer expectation.


To make the problem clear immediately upon reading, the title needs to explicitly frame **what is broken** (e.g., manual triage delays, unpredictable SLA bottlenecks, or missed priority tickets) and **how your dual-stage pipeline solves it** (automated priority routing & time-to-resolution forecasting).

Here are combinations tailored to different presentation styles:

### 1. Problem-First & Action-Oriented (Best for Impact)

* **Automating Support Triage: A Dual-Stage NLP Pipeline to Eliminate SLA Bottlenecks via Priority Classification and Time-to-Resolution Forecasting**
* **Problem highlighted:** SLA bottlenecks and manual triage delays.
* **Solution highlighted:** Automated dual-stage NLP pipeline for priority and time estimation.


* **Reducing Resolution Delays: End-to-End Sentiment-Aware Machine Learning Framework for Automated Ticket Prioritization and SLA Prediction**
* **Problem highlighted:** Resolution delays and unorganized tickets.
* **Solution highlighted:** Sentiment-aware ML framework predicting priority and SLA.



---

### 2. Standard Academic / Project Format (Title: Subtitle)

* **Overcoming Manual Support Delays: Dual-Stage NLP and Regression Pipeline for Ticket Priority Classification and Resolution Time Estimation**
* **Problem highlighted:** Manual support delays.
* **Solution highlighted:** Dual-stage classification + regression pipeline.


* **Predictive Helpdesk Routing: Mitigating High Resolution Times using Sentiment Analysis, Priority Classification, and SLA Regression**
* **Problem highlighted:** High resolution times and poor routing.
* **Solution highlighted:** Sentiment analysis + classification + regression.



---

### 3. Industry & Production-Focused

* **Smart Support Triage: Streamlining High-Volume Customer Tickets with Dual-Stage Priority and Resolution-Time ML Models**
* **Problem highlighted:** High-volume ticket overload and slow processing.
* **Solution highlighted:** Streamlined triage via dual-stage ML models.



---

### Key Formula Used to Build These

$$\text{[Overcoming the Problem]} + \text{[Technical Architecture]} + \text{[Core Outcomes]}$$

* **Problem:** *Manual Triage Delays / SLA Bottlenecks / High Resolution Times*
* **Architecture:** *Dual-Stage NLP & ML Pipeline (VADER + LinearSVC + Regressor)*
* **Outcome:** *Automated Priority Classification & Time-to-Resolution Estimation*
 -->
