# Multinomial Naive Bayes Sentiment Classifier Optimization

**[Final Research Paper (PDF)](./MNB_Sentiment_Classifier_Optimization.pdf)**  
*Systematic Improvement of Multinomial Naive Bayes Classifier for Sentiment Classification: Parameter Tuning, Enhancement Techniques, and Efficiency Optimization*  

---

This project is an **exploration-driven research study** on improving the **Multinomial Naive Bayes (MNB)** model for sentiment classification. 

Using the [IMDb movie review dataset from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data), we investigate **three distinct methods** to enhance the model’s performance — focusing on accuracy, feature quality, and computational efficiency — while preserving the model’s simplicity and interpretability.

## Research Objective

While MNB is widely used for text classification due to its simplicity and interpretability, it has known limitations. This project explores three complementary methods to improve MNB:

1. **Hyperparameter Tuning**  
   Fine-tuning model parameters, optimize `alpha`, `n-gram` ranges, and `thresholds` to increase predictive accuracy.

2. **Enhancement Techniques**  
   Improving feature representation by replacing `CountVectorizer` with `TfidfVectorizer`, and boosting the importance of discriminative keywords identified from the model

3. **Efficiency Optimization**
   Pruning vocabulary (`χ²` feature selection) to accelerate inference while maintaining accuracy.

---

## Result Summary  
- **Baseline MNB**: ~88.2% accuracy, F1 ≈ 0.882  
- **Hyperparameter Tuning**: accuracy improved to **88.6%** (better precision-recall balance)  
- **Enhancements (TF-IDF + keyword boosting)**: boosted accuracy to **89.1%**, F1 ≈ 0.889  
- **Efficiency Optimization (χ² pruning)**: achieved a **70× speed-up** (0.23 ms latency) with only a 1.7% accuracy drop  

**Conclusion:** Classic models like MNB can remain competitive when carefully tuned, enhanced, and optimized for efficiency.  

---

## Project Structure

```
.
├── data
│   └── IMDB-Dataset.csv
├── efficiency_optimization
│   ├── DiminishingLatency_Accuracy vs Latency Tradeoff.png
│   ├── DiminishingLatency_ConfusionMatrix.png
│   ├── DiminishingLatency_Output.jpg
│   └── MNB_Model_Diminishing_Latency_Study.py
├── enhancement_techniques
│   ├── MNB_Model_Enhancement_Techniques_Study.ipynb
│   ├── Model Performance Comparison.png
│   └── Top Distinctive Features by Log Probability Difference.png
├── MNB_base_model.py
├── MNB_Sentiment_Classifier_Optimization.pdf
├── parameter_tuning
│   ├── ConfusionMatrix.jpg
│   ├── MNB_Model_ParameterTuning.py
│   ├── Precision-Recall.jpg
│   └── Result.jpg
└── README.md
```
