
# Quishing Detection

## Project Overview

This project uses machine learning to detect phishing QR codes and compare them with safe QR codes.

## Dataset

I used the Quishing dataset by Trad and Chehab (2025).

Dataset link: https://github.com/fouadtrad/Detecting-Quishing-Attacks-with-Machine-Learning-Techniques-Through-QR-Code-Analysis

The dataset contains 9,987 QR codes labeled as benign (0) or phishing (1).

## Random Forest Results

I tested a Random Forest model using a 70% training, 15% validation, and 15% testing split.

- AUROC: 0.8946
- Average Precision: 0.9009
- Precision: 0.7455
- Recall: 0.8810
- Threshold: 0.40

## Project Files

- `random_forest.py` — Random Forest code
- `split.csv` — data split information
- `scores.csv` — test results
- `check_checkpoint.py` — checks the results

## References

- Trad and Chehab (2025). *Detecting Quishing Attacks with Machine Learning Techniques Through QR Code Analysis*.
- Scikit-learn documentation: https://scikit-learn.org/stable/supervised_learning.html
- Hugging Face documentation: https://huggingface.co/docs/transformers/tasks/image_classification
