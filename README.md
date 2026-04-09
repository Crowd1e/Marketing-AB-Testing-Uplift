# E-Commerce A/B Testing & Uplift Modeling

A end-to-end marketing analytics project covering statistical hypothesis testing, behavioral segmentation, and machine learning-based uplift modeling on a 588K-user e-commerce dataset.

---

## Business Question

> Does showing ads actually increase conversions — and if so, for which users?

The aggregate A/B test says yes. The deeper analysis reveals the answer is far more nuanced.

---

## Key Findings

| Finding | Detail |
|--------|--------|
| Statistically significant, small effect | Z = 7.37, p < 0.001, Cohen's h = 0.053 |
| Effect is highly heterogeneous | High-exposure segment: +6.54pp lift vs. +0.14pp for low-exposure |
| Optimal ad exposure band | 20–100 ads per user yields 13.4% uplift |
| 35.5% of users are Sleeping Dogs | Showing ads to them produces no benefit |
| Best targeting window | Monday at 16:00 (3.28% conversion rate) |

---

## Project Structure

```
├── 01_EDA.ipynb                          # Exploratory data analysis
├── 02_AB_Testing.ipynb                   # Hypothesis testing & segmentation
├── 03_Feature_Engineering_Uplift.ipynb   # XGBoost, SHAP, uplift modeling
└── Marketing_ABTesting_Report.pdf        # Full project report
```

### Notebook Breakdown

**01_EDA.ipynb**
- Dataset overview (588K rows, 7 columns)
- Conversion rate distribution
- Ad exposure analysis and outlier detection
- Time-based conversion patterns (hour, day)

**02_AB_Testing.ipynb**
- Two-proportion Z-test
- Cohen's h effect size
- Power analysis — overpowered test discussion
- K-Means segmentation (k=4) on exposure behavior
- Heterogeneous treatment effect by segment

**03_Feature_Engineering_Uplift.ipynb**
- Feature engineering (`day_num`, `is_ad` encoding)
- XGBoost classifier with `scale_pos_weight` for class imbalance
- Optuna hyperparameter tuning (50 trials, 3-fold CV)
- SHAP explainability — feature importance
- T-Learner uplift model
- Qini curve validation

---

## Analytical Pipeline

```
Raw Data (588K users)
       │
       ▼
   EDA & Cleaning
   - Outlier removal (>500 ads)
   - Time pattern analysis
       │
       ▼
   A/B Test (Layer 1)
   - Z-test: p < 0.001 ✓
   - Cohen's h = 0.053 — small effect
   - Power analysis: test overpowered
       │
       ▼
   Segmentation (Layer 2)
   - K-Means (k=4)
   - Cluster 3: ad doubles conversion rate
       │
       ▼
   ML & Uplift (Layer 3)
   - XGBoost AUC: 0.857
   - SHAP: total_ads is dominant feature
   - T-Learner: 64.5% Persuadable users
   - Qini curve: top 5K users → 3.5% uplift
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data manipulation | pandas, numpy |
| Statistics | scipy, statsmodels |
| Machine learning | scikit-learn, xgboost |
| Explainability | shap |
| Hyperparameter tuning | optuna |
| Visualization | matplotlib |

---

## Dataset

[Marketing A/B Testing Dataset](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing) — Kaggle

| Column | Description |
|--------|-------------|
| `user id` | Anonymous user identifier |
| `test group` | `ad` (treatment) or `psa` (control) |
| `converted` | Boolean — did the user purchase? |
| `total ads` | Number of ad impressions shown |
| `most ads day` | Day with highest ad exposure |
| `most ads hour` | Hour with highest ad exposure |

---

## Results Summary

### A/B Test
- Ad group conversion: **2.55%** vs. PSA group: **1.79%**
- Difference is statistically significant but effect size is small
- With 588K users, the test is overpowered — p-value alone is misleading

### Segmentation
| Cluster | Users | Avg. Ads | Ad Conv. | PSA Conv. | Lift |
|---------|-------|----------|----------|-----------|------|
| Morning Crowd | 272K | 15 | 1.07% | 0.71% | +0.36pp |
| Evening Crowd | 258K | 15 | 1.43% | 1.29% | +0.14pp |
| Super Exposed | 7.5K | 257 | 15.46% | 14.08% | +1.38pp |
| **High Exposed** | **50K** | **89** | **14.35%** | **7.81%** | **+6.54pp** |

### Uplift Model
- Persuadable users: **75,837 (64.5%)**
- Sleeping Dogs: **41,667 (35.5%)**
- Average uplift score: **0.049**

---

## Author

**Furkan Eren Durmaz**
Industrial Engineering, Koç University (2023–2027)
[github.com/Crowd1e](https://github.com/Crowd1e)
