# 🌿 Parthenium Weed Detection App

This project uses a Deep Learning model (`parthenium_detector.h5`) trained on Parthenium and non-weed images.
It detects whether an uploaded image contains the Parthenium weed.

# Model Details

- Architecture: Custom CNN (Convolutional Neural Network)
- Input Size: 224x224 RGB
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metrics: Accuracy

# Performance Summary

| Metric | Train | Validation |
|---------|--------|-------------|
| Accuracy | 90.4% | 94.8% |
| Loss | 0.26 | 0.17 |

# Training Log Summary

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|:------:|:---------------:|:-------------:|:------------:|:-----------:|
| 1 | 75.6% | 81.4% | 0.50 | 0.39 |
| 3 | 87.6% | 82.4% | 0.28 | 0.33 |
| 5 | 87.6% | 85.5% | 0.26 | 0.28 |
| 7 | 91.0% | 91.7% | 0.21 | 0.23 |
| 9 | 92.6% | 90.7% | 0.17 | 0.22 |

# Observations & Limitations

- Performs well on leaf-based Parthenium images.  
- Sometimes misclassifies flower-only Parthenium** due to dataset limitations.  
- Expanding dataset to include flowering & dry-stage weeds will improve accuracy.  
- Future goal: integrate transfer learning (MobileNetV2) for faster convergence and robustness.

#  How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
