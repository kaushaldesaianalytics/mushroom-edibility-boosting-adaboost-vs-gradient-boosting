# Mushroom Edibility Classification
## Boosting Methods: AdaBoost and Gradient Boosting

Can physical characteristics reliably distinguish edible mushrooms from poisonous ones? This project applies two sequential boosting methods to the UCI Mushroom dataset, training AdaBoost and Gradient Boosting classifiers on the same data to compare how each builds its ensemble, weighs features, and performs on a high-stakes binary classification task.

---

## Overview

Both AdaBoost and Gradient Boosting build ensembles of weak learners sequentially, with each new learner correcting the errors of those before it. They differ in how that correction happens: AdaBoost reweights misclassified samples so subsequent learners focus on them, while Gradient Boosting fits each new tree to the residual errors (gradients of the loss function) of the current ensemble. This project trains both on identical data, enabling a direct comparison of accuracy, feature importances, and sensitivity to ensemble size.

---

## Dataset

The [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom) contains 8,124 records across 22 categorical physical features. The binary target indicates whether a mushroom is edible or poisonous.

| Feature Group | Examples |
|---|---|
| Cap attributes | cap-shape, cap-surface, cap-color |
| Gill attributes | gill-attachment, gill-spacing, gill-size, gill-color |
| Stalk attributes | stalk-shape, stalk-root, stalk-surface, stalk-color |
| Veil and ring | veil-type, veil-color, ring-number, ring-type |
| Other attributes | odor, bruises, spore-print-color, population, habitat |
| **Target** | **class: edible (e) or poisonous (p)** |

The dataset is nearly balanced (51.8% edible, 48.2% poisonous), making accuracy a reliable evaluation metric without resampling.

---

## Workflow

### Part 1: EDA
Class distribution confirms a near-balanced split. A feature cardinality bar chart shows the number of unique values per categorical feature, motivating one-hot encoding. A countplot of odor by class reveals that certain odor types (foul, pungent, spicy) appear almost exclusively in poisonous mushrooms, while almond and anise odors are strong edibility signals.

### Part 2: Feature Engineering
All 22 categorical features are one-hot encoded with `drop_first=True`, expanding the feature space to 95 binary columns. An 85/15 train/test split is fixed with a consistent random state and shared across both models.

### Part 3: AdaBoost
A single decision stump (n_estimators=1) is trained first as the baseline, establishing the minimum performance a one-level tree achieves. An error rate sweep from 1 to 95 estimators shows how quickly the ensemble converges. Feature importances from the final model identify which mushroom characteristics drive the most decisions across the full ensemble, with zero-importance features filtered out.

### Part 4: Gradient Boosting with GridSearchCV
A 5-fold GridSearchCV sweeps six n_estimators values and four max_depth values (24 parameter combinations) to identify the optimal Gradient Boosting configuration. The best estimator is evaluated on the held-out test set. Feature importances are extracted and compared against AdaBoost to show how the two methods distribute weight differently across features.

### Part 5: Model Comparison
Both tuned models are evaluated side-by-side on the same test set using accuracy, classification report, and a dual confusion matrix panel. This highlights any differences in error patterns between the two boosting approaches.

---

## Results

| Model | Accuracy |
|---|---|
| AdaBoost (1 estimator baseline) | ~72% |
| AdaBoost (95 estimators) | ~100% |
| Gradient Boosting (GridSearchCV) | ~100% |

Both tuned models achieve near-perfect accuracy on this dataset. Odor is the dominant feature for both methods, consistent with EDA findings.

---

## Key Concepts

**AdaBoost vs. Gradient Boosting:** AdaBoost adapts by reweighting samples, placing more emphasis on hard-to-classify examples. Gradient Boosting adapts by fitting new trees to the gradient of the loss function, making it more generalizable to arbitrary loss functions and typically more robust to noise.

**Decision Stumps as Weak Learners:** AdaBoost's default weak learner is a one-level decision tree. A single stump can only make one binary decision, which is why accuracy starts low and improves sharply as more stumps are added.

**Error Rate Sweep:** Plotting error rate vs. n_estimators reveals how fast the ensemble converges. On clean, well-separated data like this mushroom dataset, convergence is rapid. Noisier datasets show slower, sometimes non-monotonic improvement.

**max_depth in Gradient Boosting:** Deeper trees per stage capture more complex patterns but risk overfitting. GridSearchCV identifies the right depth for this data, balancing expressiveness with generalization.

**Feature Importance Comparison:** AdaBoost tends to concentrate importance on a small number of highly discriminative features (odor dominates). Gradient Boosting often distributes importance more broadly. Comparing both highlights the interaction between the boosting strategy and feature selection behavior.

---

## Stack

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (AdaBoostClassifier, GradientBoostingClassifier, GridSearchCV, ConfusionMatrixDisplay)

---

## File Structure

```
boosting-mushroom-classification/
├── boosting_mushroom_classification.ipynb   # Main project notebook
├── mushrooms.csv                            # UCI Mushroom dataset
└── README.md
```

---

## How to Run

1. Clone the repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open `boosting_mushroom_classification.ipynb` in Jupyter and run all cells
