# Predicting Spending and Churn in Streaming Services: A Comparative Machine Learning Study

## Summary

I built a complete machine learning pipeline to predict customer spending behaviour and churn in a streaming service dataset of 5,000 users. The work spanned single-variable regression, multi-variable models, neural network regression, three classification algorithms for churn prediction, and two clustering approaches for customer segmentation. The emphasis throughout was on fair comparison, not just on finding the highest number.

## Context and Motivation

Streaming platforms operate in a market where customer retention is everything. Acquiring a new subscriber costs several times more than keeping an existing one, which makes predicting who will leave, and understanding how much different customers spend, a problem with real financial weight. But the interesting question is not just "can we predict churn?" It is: which modelling approaches capture different aspects of the problem, and what does each method reveal that the others miss?

I wanted to use this project to test my understanding of the full supervised and unsupervised learning toolkit, not by applying methods mechanically, but by building each model on the same carefully prepared data and comparing them on equal terms.

## Problem Definition

Two prediction tasks and one discovery task:

1. **Regression:** Predict monthly spending from customer attributes (age, subscription length, support tickets raised, satisfaction score, discount offered, last activity, plus categorical features like gender, region, and payment method).
2. **Classification:** Predict whether a customer has churned (binary), using the same feature set.
3. **Clustering:** Discover natural customer segments without using spending or churn labels.

## Why the Problem Mattered

Each of these tasks mirrors a real operational question. Spending prediction informs pricing and promotion strategies. Churn prediction enables targeted retention efforts. Customer segmentation reveals structure in the user base that neither regression nor classification can surface. Together, they form a more complete picture than any single model could provide.

## My Role and Contribution

I designed and implemented the entire pipeline: data preprocessing, feature engineering, model selection, training, evaluation, and comparison. All preprocessing was shared across tasks to ensure that performance differences reflected model behaviour, not data preparation inconsistencies.

## Approach and Methodology

**Data preparation** was the foundation. The dataset had 5,000 records with 12 columns, including 500 missing values in Age and missing values in Satisfaction Score. I imputed missing numerics using median values and built reusable preprocessing pipelines that handled numeric scaling and categorical one-hot encoding consistently across all tasks. I created a single stratified train-test split for regression and a separate one for classification, then reused these indices throughout. This matters. If different models see different train-test splits, comparison becomes unreliable.

**Single-variable regression** tested each numeric feature individually with both linear and polynomial (degree 2) models. This revealed which features had the strongest individual relationships with monthly spending and whether those relationships were linear. It is a simple step, but it grounds the later multi-variable work in understanding of the individual feature contributions.

**Multi-variable regression** combined all numeric features in a single linear regression model with imputation and scaling. The improvement over the best single-variable model showed how much predictive information the features carry jointly versus individually.

**Mixed-feature regression** brought in categorical variables (gender, region, payment method) using a ColumnTransformer pipeline and Random Forest Regressor. The shift from linear regression to Random Forest was deliberate: I wanted to see whether non-linear interactions between features mattered, and whether categorical variables carried information that pure numeric models missed.

**ANN regression** used a two-hidden-layer neural network (64 and 32 neurons, ReLU activations, linear output) trained with Adam and MSE loss. Early stopping with patience prevented overfitting. I included this not because I expected it to dominate on a 5,000-row dataset, but because understanding when neural networks help and when they do not is itself informative.

**Model comparison** ranked all regression approaches by test RMSE and R-squared on the same held-out data. This is where the real insight lives: not in any single model's performance, but in the pattern of results across methods.

**Churn classification** trained Logistic Regression, Random Forest, and SVM (RBF kernel) on the same preprocessed features. I used class weights to handle any imbalance, evaluated with accuracy, precision, recall, F1, and ROC-AUC, and plotted ROC curves for the best model. Comparing three fundamentally different classifiers on the same problem reveals which aspects of the decision boundary matter most.

**k-Means clustering** used the elbow method and silhouette scores to select the number of clusters, then profiled each cluster by its feature distributions. **Agglomerative clustering** with the same k provided a comparison, including a dendrogram on a 200-customer sample for interpretability. The silhouette scores from both methods quantified which algorithm produced more coherent groupings.

## Key Technical Decisions

**Shared preprocessing pipelines.** Building a single ColumnTransformer that handled numeric and categorical features identically across all models eliminated a common source of error in comparative studies. If preprocessing differs between models, you are not comparing models; you are comparing preprocessing choices.

**Stratified splits with fixed indices.** Reusing the same train-test indices across regression tasks (and separately across classification tasks) ensured fair comparison. This sounds basic, but I have seen projects where different random seeds produce different splits, and conclusions change accordingly.

**Random Forest for mixed features rather than encoding categoricals into linear regression.** One-hot encoding categorical variables into a linear model creates a high-dimensional sparse feature space that linear regression handles poorly. Random Forest naturally accommodates mixed feature types and captures interactions without explicit feature engineering.

**Including the ANN despite dataset size.** Five thousand rows is small for a neural network. I expected the ANN to perform comparably to or slightly worse than the Random Forest, and I wanted to verify that expectation empirically rather than assume it.

## Challenges, Tradeoffs, and Constraints

The 500 missing Age values required a decision about imputation strategy. Median imputation is conservative and does not inflate variance, but it also does not capture the uncertainty that missing data introduces. For a larger project, I would consider multiple imputation or at least sensitivity analysis to understand how imputation choices affect downstream predictions.

With only 5,000 observations, model capacity had to be managed carefully. Overly complex models (deep networks, large random forests) would fit training noise. The evaluation framework helped catch this, but the fundamental constraint was data volume.

The clustering evaluation was limited by the absence of ground truth. Silhouette scores measure internal coherence, not whether the discovered segments correspond to meaningful business categories. This is an inherent limitation of unsupervised learning on observational data.

## What I Learned

The most important insight was methodological: the value of a machine learning project lies less in any single model's accuracy and more in the rigour of the comparison framework. When you control data preparation, splitting, and evaluation criteria, you can make trustworthy statements about which methods suit a problem. When you do not, you are generating numbers that look precise but mean little.

I also developed a stronger intuition for when model complexity pays off and when it does not. On this dataset, the Random Forest with mixed features outperformed simpler models meaningfully, while the neural network added complexity without commensurate improvement. That result is not universal, but the ability to observe it in a controlled setting was valuable.

## Outcome

The complete pipeline produced comparative evaluation tables for regression and classification, cluster profiles, and visualisations including ROC curves and dendrograms. The Random Forest with mixed features was the strongest regressor. Classification models performed comparably, with ROC-AUC providing the clearest separation between them. k-Means produced slightly more coherent clusters than agglomerative clustering on this dataset.

## Why This Work Matters for Research Readiness

Research demands the ability to set up fair experiments, control variables, compare methods honestly, and draw conclusions that are supported by evidence rather than hope. This project is structured exactly that way. It demonstrates not just familiarity with a range of ML techniques, but the discipline of using them comparatively and the judgment to interpret results in context.

## What I Would Investigate Next

Feature importance analysis across the regression and classification models would reveal whether the same features drive spending and churn or whether different customer attributes matter for different outcomes. I would also explore time-series approaches if longitudinal data were available, since churn is fundamentally a temporal phenomenon that cross-sectional models can only approximate.
