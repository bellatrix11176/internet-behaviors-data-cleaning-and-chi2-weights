# Internet Behaviors — Data Cleaning & Chi-Square Attribute Weights

This project explores internet usage behaviors using **Chi-Square attribute weighting** to identify which behavioral features are most strongly associated with **online shopping activity**.

The focus of this repository is **fundamentals**:
- responsible data cleaning
- categorical statistical testing
- interpretation of statistical outputs
- transparent, reproducible analysis in Python

This is an exploratory analysis project intended to demonstrate core data science reasoning rather than predictive modeling.

---

## Project Objective

To evaluate which internet behavior attributes show meaningful statistical association with **Online_Shopping** using the Chi-Square test of independence.

Chi-Square is appropriate here because:
- the target variable is categorical
- most predictors are categorical or discretized
- the goal is **association strength**, not prediction accuracy

---

## Dataset

**Internet-Behaviors.csv**

The dataset contains demographic and behavioral information related to internet usage, including:
- online activity indicators
- time-based usage metrics
- platform preferences

Sensitive attributes are removed before exporting a cleaned, shareable dataset.

---

## Methodology Overview

1. Minimal data cleaning (no aggressive row removal)
2. Normalization of Yes/No style fields
3. Discretization of numeric variables for Chi-Square testing
4. Chi-Square tests between each predictor and Online_Shopping
5. Ranking attributes by Chi-Square statistic
6. Interpretation of p-values and statistical relevance
7. Export of clean datasets, weights, charts, and written summary

---

## Outputs (auto-generated)

All outputs are recreated each time the script is run:

| File | Description |
|----|----|
| `internet_behaviors_deidentified_clean.csv` | Cleaned dataset with sensitive columns removed |
| `chi2_attribute_weights.csv` | Chi-Square statistics, p-values, and degrees of freedom |
| `chi2_attribute_weights.png` | Bar chart visualization of attribute weights |
| `results_summary.txt` | Plain-language interpretation of results |

---

## Repository Structure

