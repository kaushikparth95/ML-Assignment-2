# ML Assignment 2 - Online Shoppers Purchase Prediction

M.Tech (AIML), Machine Learning, Assignment 2

- **Name:** Parth Kaushik
- **BITS ID:** 2026AJ05016
- **GitHub repository:** https://github.com/kaushikparth95/ML-Assignment-2
- **Live Streamlit app:** https://parth-ml-assignment-2.streamlit.app/

---

## a. Problem statement

In online shopping, a large number of people visit the website but only a small
number of them actually buy something. In this assignment the aim is to predict,
using the data collected during a browsing session, whether the session will end in a
purchase (`Revenue = True`) or not. So it is a **binary classification** problem.

If we can predict this while the user is still on the website, the company can take
some action, for example show a discount offer to a user who is likely to buy but is
about to leave. In this dataset only around 15.5% sessions end in a purchase, so the
classes are imbalanced. A model that always says "no purchase" will already get about
84% accuracy, which is why I have compared the models mainly on AUC, F1 and MCC and not
only on accuracy.

I trained five classifiers on the same data: Logistic Regression, Decision Tree,
K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine. All of them are
evaluated on the same test set using six metrics, and a Streamlit app is made to show
the results.

## b. Dataset description

| Property | Details |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (dataset id 468): https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Paper | Sakar, C.O., Polat, S.O., Katircioglu, M. and Kastro, Y. (2019). Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. Neural Computing and Applications, 31, 6893-6908. |
| Instances | 12,330 sessions (12,205 after removing 125 duplicate rows) |
| Features | 17 input features (10 numerical, 7 categorical) and 1 target |
| Target | `Revenue`: True = purchase (15.5%), False = no purchase (84.5%) |
| Missing values | None |
| Collection | Sessions collected over 1 year from an e-commerce website. Each session belongs to a different user, so there is no bias towards any particular day, user or campaign. |

**Features**

| Feature | Type | Meaning |
|---|---|---|
| Administrative, Informational, ProductRelated | numeric (count) | Number of pages of that type visited in the session |
| Administrative_Duration, Informational_Duration, ProductRelated_Duration | numeric (seconds) | Total time spent on that type of page |
| BounceRates | numeric (0 to 0.2) | Average bounce rate (Google Analytics) of the pages visited |
| ExitRates | numeric (0 to 0.2) | Average exit rate of the pages visited |
| PageValues | numeric | Average page value of the pages visited before a transaction |
| SpecialDay | numeric (0 to 1) | How close the visit date is to a special day like Mother's Day or Valentine's Day |
| Month | categorical (10 values) | Month of the visit |
| OperatingSystems, Browser, Region, TrafficType | categorical (integer codes) | Technical details and traffic source of the visitor |
| VisitorType | categorical | Returning_Visitor, New_Visitor or Other |
| Weekend | boolean | Whether the visit was on a weekend |

**Preprocessing** (this is saved inside each model pipeline, so the app does the same steps automatically)

1. Removed 125 exact duplicate rows, so that the same row cannot come in both train and test.
2. Stratified 80/20 train-test split with `random_state = 2611`. This gives 9,764 train rows
   and 2,441 test rows. The test part is saved as it is in [`test_data.csv`](test_data.csv).
3. `log1p` and then `StandardScaler` on the 7 right skewed columns (page counts, durations and PageValues).
4. `StandardScaler` on BounceRates, ExitRates and SpecialDay.
5. `OneHotEncoder(handle_unknown="ignore")` on Month, VisitorType, Weekend, OperatingSystems,
   Browser, Region and TrafficType. The integer codes are just labels, so treating them as
   numbers would not make sense.

Hyperparameters of every model were tuned using 5-fold stratified `GridSearchCV` on the
training data only, with **F1** as the scoring metric.

## c. GitHub repository link

https://github.com/kaushikparth95/ML-Assignment-2

```
ml-assignment-2/
├── app.py                 # Streamlit app (loads the saved models, does not retrain)
├── requirements.txt       # libraries needed for Streamlit Cloud
├── README.md
├── test_data.csv          # 20% test split (17 features + Revenue), upload this in the app
├── .streamlit/config.toml # app theme
├── data/
│   └── online_shoppers_intention.csv   # original dataset from UCI
└── model/
    ├── train_models.ipynb     # notebook with EDA, training, tuning and evaluation
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics_summary.csv    # test set metrics of all 5 models
    └── model_metadata.json    # feature list, best hyperparameters, split details
```

## d. Models used

All five models are trained on the same 9,764 training rows and tested on the same
2,441 test rows (`test_data.csv`). Precision, Recall and F1 are calculated for the
positive class (Purchase).

| Model | Best hyperparameters (grid search) |
|---|---|
| Logistic Regression | C = 0.01, class_weight = balanced |
| Decision Tree | criterion = gini, max_depth = 12, min_samples_leaf = 30, class_weight = None |
| KNN | n_neighbors = 21, weights = distance, p = 2 (Euclidean) |
| Naive Bayes (Gaussian) | var_smoothing = 0.1 |
| Support Vector Machine | kernel = rbf, C = 0.5, gamma = 0.01, class_weight = balanced |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8828 | 0.9261 | 0.5930 | 0.8010 | 0.6815 | 0.6221 |
| Decision Tree | **0.9058** | 0.9196 | **0.7452** | 0.6047 | 0.6676 | 0.6180 |
| KNN | 0.8914 | 0.9189 | 0.6997 | 0.5366 | 0.6074 | 0.5521 |
| Naive Bayes | 0.7538 | 0.8846 | 0.3768 | **0.8770** | 0.5271 | 0.4590 |
| Support Vector Machine | 0.8849 | **0.9273** | 0.5973 | 0.8115 | **0.6881** | **0.6305** |

(Bold = best value in that column.)

Confusion matrix values on the test set:

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 1849 | 210 | 76 | 306 |
| Decision Tree | 1980 | 79 | 151 | 231 |
| KNN | 1971 | 88 | 177 | 205 |
| Naive Bayes | 1505 | 554 | 47 | 335 |
| Support Vector Machine | 1850 | 209 | 72 | 310 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed really well for a linear model, with AUC 0.926, F1 0.682 and MCC 0.622, which is almost the same as SVM. I think the log1p transform on the skewed columns helped a lot here. Grid search selected strong regularisation (C = 0.01) and balanced class weights, so recall is high (0.801) but precision is lower (0.593). PageValues has the biggest positive coefficient (+1.38), followed by Month = Nov (+0.76), while visits in May/March and higher ExitRates push the prediction towards "no purchase". It is also the smallest model (around 7 KB), one of the fastest, and easy to interpret. |
| Decision Tree | Gave the highest accuracy (0.906) and highest precision (0.745). Out of the sessions it predicted as purchase, roughly 3 out of 4 were correct, with only 79 false positives. But recall is only 0.605 (151 buyers missed) because grid search chose no class weighting. min_samples_leaf = 30 helped keep the tree (depth 12, 133 leaves) from overfitting. PageValues alone gives about 72% of the feature importance, and BounceRates comes next. AUC (0.920) is a bit lower than LR and SVM because a tree gives the same probability for all samples in one leaf, so the ranking is not very smooth. |
| KNN | AUC is decent (0.919) and precision is second best (0.700), but it has the lowest recall of all five models (0.537, 177 buyers missed) and its F1 (0.607) is only better than Naive Bayes. KNN has no class_weight option, and since about 84% of sessions are non-buyers, most neighbours of a point belong to the majority class. Also, 65 out of the 75 columns after encoding are one-hot columns, which reduces the effect of the few useful numeric columns in the distance calculation. Best k was 21 with distance weighting. It stores the full training data, so the model file is the biggest (around 6 MB) and prediction is much slower than the other models. |
| Naive Bayes | Lowest accuracy (0.754), precision (0.377) and MCC (0.459), but the highest recall (0.877, only 47 buyers missed). This is because it predicts purchase too often and gives 554 false positives. The independence assumption does not hold here since many features are correlated (page counts with their durations, BounceRates with ExitRates), and one-hot columns are not Gaussian. The large var_smoothing (0.1) selected by grid search also shows that the model needed a lot of smoothing. It can be used as a quick baseline or where catching most buyers is more important than false alarms. |
| Support Vector Machine | Best model overall, with the highest AUC (0.927), F1 (0.688) and MCC (0.631), recall 0.812 and precision 0.597. RBF kernel with small gamma (0.01) and balanced class weights gives a smooth non-linear boundary, but the improvement over Logistic Regression is very small (about +0.007 F1 and +0.008 MCC). It is also the most expensive model: 3,709 support vectors, the longest grid search time (around 20 s) and an extra internal cross-validation for probability estimates. One thing I noticed is that for SVM, predict() uses the decision function and predict_proba() uses Platt scaling, so for some rows near the boundary the two do not match exactly at 0.5. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF)**, because it is best on AUC, F1 and MCC, which are the more reliable metrics for imbalanced data, and it catches about 81% of the buyers. Logistic Regression is a very close second and is much cheaper and easier to explain, so in a real project I would also consider it. If the business wants fewer wrong offers (higher precision), Decision Tree is a better choice. In all the models PageValues came out as the most important feature. |
