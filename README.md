# cs179-imdb

This project is an **exploration-driven research study** on improving the **Multinomial Naive Bayes (MNB)** model for sentiment classification. Using the [IMDb movie review dataset from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data), we investigate **three distinct methods** to enhance the model’s performance — focusing on accuracy, feature quality, and computational efficiency — while preserving the model’s simplicity and interpretability.

## Research Objective

While MNB is widely used for text classification due to its simplicity and interpretability, it has known limitations. This project explores three complementary methods to improve MNB:

1. **Hyperparameter Tuning**  
   Fine-tuning model parameters (e.g., smoothing factor `alpha`, n-gram range) to increase predictive accuracy.

2. **Enhancement Techniques**  
   Improving feature representation by:
   - Replacing `CountVectorizer` with `TfidfVectorizer`
   - Boosting the importance of discriminative keywords identified from the model

3. **Efficiency Optimization** *(Planned)*  
   Applying methods such as dimensionality reduction and sparse matrix optimization to reduce runtime and memory usage without sacrificing performance.

---

### 📘 *Systematic Improvement of Multinomial Naive Bayes Classifier for Sentiment Classification: Parameter Tuning, Enhancement Techniques, and Efficiency Optimization*

## Project Structure

```
.
├── data
│   └── IMDB-Dataset.csv
├── efficiency_optimization
│   ├── DiminishingLatency_ConfusionMatrix.png
│   └── MNB_Model_Diminishing_Latency_Study.py
├── enhancement_techniques
│   └── MNB_Model_Enhancement_Techniques_Study.ipynb
├── MNB_base_model.py
├── parameter_tuning
│   └── MNB_Model_ParameterTuning.py
└── README.md
```
