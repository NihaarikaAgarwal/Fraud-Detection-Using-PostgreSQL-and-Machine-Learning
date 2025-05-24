
# Fraud Detection Using SparkSQL and Machine Learning on IDNet Dataset

##  Project Overview

This project presents a scalable and intelligent fraud detection system built using Apache Spark and deep learning, applied to the IDNet dataset. It identifies and classifies forged identity documents with high accuracy, utilizing state-of-the-art CNN architectures (MobileNetV3 and ResNet18). The system integrates model inference into distributed SparkSQL pipelines to allow high-throughput, low-latency fraud analytics.

##  Problem Statement

Online identity verification systems face increasing threats from forged documents. Traditional rule-based validation often fails to detect subtle tampering like copy-paste or digital inpainting. This project builds a robust ML pipeline to:
- Detect whether an ID is genuine or fraudulent.
- Classify the type of fraud (crop-and-replace vs. inpaint-and-rewrite).
- Enable scalable querying over large datasets using SparkSQL. 

##  Dataset

- **Source:** IDNet Dataset (Estonia ID subset)
- **Classes:**
  - 3,000 Genuine IDs
  - 3,000 Crop-and-Replace Fakes
  - 3,000 Inpaint-and-Rewrite Fakes

##  Models Used

###  MobileNetV3
- Task: Genuine vs Fraud classification
- - Accuracy: **91.69%**
- Lightweight architecture optimized for efficiency and mobile deployment.

###  ResNet18
- Task: Fraud Type Classification
- - Accuracy: **91.64%**
- Powerful residual network capable of extracting deep semantic patterns.

Both models use pretrained weights (ImageNet) and were fine-tuned using data augmentation and regularization techniques.

##  Methodology

###  Preprocessing
- Images loaded either as file paths or base64 strings
- PyTorch preprocessing pipelines applied for model input

###  Inference with SparkSQL
Two Spark User-Defined Functions (UDFs) were developed:
- `infer_fraud_udf`: Classifies an ID as genuine or fraudulent
- `infer_fraud_type_udf`: Classifies type of fraud (crop or inpaint)

##  Query Examples

```sql
-- Query 1: Classify embedded image strings
SELECT fraud_label, COUNT(*) 
FROM (
  SELECT infer_fraud_udf(base64_image) AS fraud_label
  FROM image_data
)
GROUP BY fraud_label;
```

```sql
-- Query 2: Classify images by file path
SELECT predicted_class, COUNT(*)
FROM (
  SELECT infer_fraud_type_udf(image_path) AS predicted_class
  FROM file_table
)
GROUP BY predicted_class;
```

## Performance

| Query | Model        | Images | Total Time (s) | Latency (ms/img) |
|-------|--------------|--------|----------------|------------------|
| Q1    | MobileNetV3  | 6000   | 61.68          | 10.3             |
| Q2    | ResNet18     | 5000   | 40.74          | 8.1              |

## 📈 Results

| Metric      | MobileNetV3 | ResNet18 |
|-------------|-------------|----------|
| Accuracy    | 91.69%      | 91.64%   |
| Precision   | 0.9951      | 0.9173   |
| Recall      | 0.8100      | 0.9164   |
| F1 Score    | 0.8931      | 0.9163   |

## 📝 References

1. [Spark SQL](https://doi.org/10.1145/2723372.2742797) - Armbrust et al.
2. [DeepFake Detectors for Facial Parts](https://doi.org/10.3390/electronics12183932)
3. [MobileNetV3 for Image Classification](https://doi.org/10.1109/ICBAIE52039.2021.9389905)
4. https://arxiv.org/html/2408.01690v1

