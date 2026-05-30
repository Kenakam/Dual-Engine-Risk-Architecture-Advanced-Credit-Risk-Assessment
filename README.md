# Dual-Engine Risk Architecture: Advanced Credit Risk Assessment via Machine Learning Pipelines & Deep Neural Optimization



## 📌 Project Overview



This repository hosts a production-grade, end-to-end multi-paradigm predictive framework designed to solve structural credit risk modeling and financial risk classification using the comprehensive **Home Credit Default Risk dataset**. The primary operational objective is to forecast individual probability of loan default (the binary target variable `TARGET`), optimizing for risk-adjusted consumer lending profiles.

To deliver absolute excellence, the project maps this structural problem through a comprehensive, dual-engine topology bifurcated into two highly structured execution pipelines:

1. **Classical Machine Learning Pipeline (`(ML) Dual-Engine Risk Architecture Advanced Credit Risk Assessment.ipynb`)**: An exhaustive pipeline orchestrating over 20 concurrent model-data configurations across multi-collinearity filtering (VIF), empirical Weight of Evidence (WoE) mapping, Principal Component Analysis (PCA) feature spaces, high-capacity ensemble methods (Random Forest, LightGBM, XGBoost, CatBoost), and meta-learning configurations (Soft Voting and Stacking Architectures).


2. **Deep Neural Optimization Engine (`(DL) Dual-Engine Risk Architecture Advanced Credit Risk Assessment.ipynb`)**: A robust Deep Learning pipeline engineered around a dynamic multi-layer Feed-Forward Artificial Neural Network (ANN). The model incorporates dropout regularizations, early stopping mechanisms, and automated architecture synthesis driven via an evolutionary hyperparameter searching process using the **Optuna framework**.



Both engines leverage advanced threshold tunings, global evaluation metrics (ROC-AUC, Gini Index), and feature attribution maps to construct an industry-compliant, auditable credit-scoring engine.

---

## 🛠️ Unified Architectural Blueprint



The execution pipeline standardizes feature validation, transformations, and mathematical metrics before feeding the specialized learning modules.

```
                                  [ Raw Source Credit Application Data ]
                                                    │
                                      (50,000 Sample Balanced Subset)
                                                    │
                                                    ▼
                                   [ Structured Data Preprocessing ]
                                    • Dropped Columns > 60% Missing
                                    • Numerical Mean/Median Imputation
                                    • Categorical Mode Imputation
                                                    │
                                                    ▼
                                   [ Domain Feature Engineering ]
                             • INCOME_PER_PERSON      • CREDIT_TO_INCOME_RATIO
                             • ANNUITY_INCOME_RATIO   • CREDIT_TO_ANNUITY_RATIO
                             • EXT_SOURCES_PROD       • EXT_SOURCES_MEAN
                             • DAYS_EMPLOYED_PERCENT  • BUR_AMT_REQ_SUM
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
   [ Classical Machine Learning Engine ]                           [ Deep Neural Optimization Engine ]
                   │                                                                 │
    ┌──────────────┼──────────────┐                                                 ▼
    ▼              ▼              ▼                                   [ Scaled & Standardized Input Matrix ]
[ VIF Filter ]  [ WoE Maps ]   [ PCA Space ]                                         │
    │              │              │                                                 ▼
    ▼              ▼              ▼                                   [ Optuna Automated Trial Ingestion ]
[ KNN Maps ]    [ LogReg ]    [ Ensembles ]                           • Evaluates Architecture Depth
    │              │              │                                   • Explores Multi-Optimizer Nodes
    └──────────────┬──────────────┘                                   • Optimizes L1/L2 and Dropouts
                   ▼                                                                 │
      [ Multi-Model Grid Training ]                                                 ▼
     • Random Forest  • LightGBM                                      [ Top Candidate ANN Construction ]
     • XGBoost        • CatBoost                                      • Multi-layer Dense Topology
                   │                                                  • Dropout Regularized Matrix
                   ▼                                                  • Early Stopping Control Loops
    [ Meta-Learning Integration ]                                                   │
    • Soft Voting Ensemble                                                           ▼
    • Stacking Meta-Classifier                                        [ Mathematical Evaluation Matrix ]
                   │                                                   • ROC-AUC Probability Mapping
                   └────────────────────────────────┬────────────────────────────────┘
                                                    ▼
                                     [ Enterprise Risk Deployment ]
                                     • Global Gini Metric Auditing
                                     • Dynamic Task-Related Thresholds

```

### 1. Data Ingestion & Preprocessing Suite



* **Exploratory Profiling**: Processes an aggregated layout of 122 foundational dimensions across structural financial indices.


* **Missing Value Pruning**: Features exhibiting extreme localized sparsity exceeding **60% missing thresholds** (such as specific internal building features `COMMONAREA_AVG`, `NONLIVINGAPARTMENTS_MEDI`, and historical `OWN_CAR_AGE`) are explicitly dropped.


* **Imputation Protocols**: Missing numerical criteria are backfilled cleanly via the feature space mean/median metrics, preventing mathematical discontinuities during cross-validation. Missing nominal criteria are handled using mode strategy backfilling.


* **Categorical Encoding**: Transformed seamlessly using `LabelEncoder` suites to guarantee numeric compliance across all backpropagation matrix nodes.



### 2. High-Impact Mathematical Feature Engineering



To maximize the learning signal, ten custom domain-specific features were mathematically synthesized into the model matrices:

* $\text{INCOME\_PER\_PERSON} = \frac{\text{AMT\_INCOME\_TOTAL}}{\text{CNT\_FAM\_MEMBERS}}$

* $\text{ANNUITY\_INCOME\_RATIO} = \frac{\text{AMT\_ANNUITY}}{\text{AMT\_INCOME\_TOTAL}}$

* $\text{CREDIT\_TO\_INCOME\_RATIO} = \frac{\text{AMT\_CREDIT}}{\text{AMT\_INCOME\_TOTAL}}$

* $\text{CREDIT\_TO\_ANNUITY\_RATIO} = \frac{\text{AMT\_CREDIT}}{\text{AMT\_ANNUITY}}$

* $\text{EXT\_SOURCES\_PROD} = \text{EXT\_SOURCE\_1} \times \text{EXT\_SOURCE\_2} \times \text{EXT\_SOURCE\_3}$

* $\text{EXT\_SOURCES\_MEAN} = \frac{\text{EXT\_SOURCE\_1} + \text{EXT\_SOURCE\_2} + \text{EXT\_SOURCE\_3}}{3}$

* $\text{DAYS\_EMPLOYED\_PERCENT} = \frac{\text{DAYS\_EMPLOYED}}{\text{DAYS\_BIRTH}}$

* $\text{DAYS\_LAST\_PHONE\_CHANGE\_PERCENT} = \frac{\text{DAYS\_LAST\_PHONE\_CHANGE}}{\text{DAYS\_BIRTH}}$

* $\text{BUR\_AMT\_REQ\_SUM} = \sum \text{Bureau Credit Inquiries (Hour, Day, Week, Month, Qrt, Year)}$

* $\text{AMT\_DIFF\_CREDIT\_GOODS} = \text{AMT\_CREDIT} - \text{AMT\_GOODS\_PRICE}$


---

## 🦾 Engine 1: Classical Machine Learning Pipeline



This engine sets up an unyielding architectural gauntlet executing **more than 20 model-data combinations** across distinct algorithmic variants, feature-selection spaces, dimensionality reductions, and complex model stackings.

### 1. Data Subsets & Feature Space Transformations



To evaluate the absolute generalization performance of classical estimators, raw dimensions were funneled into 4 mathematical feature configurations:

* **The Variance Inflation Factor (VIF) Subset**: Designed to eradicate geometric inflation in linear spaces by filtering heavy multi-collinearity. Leftover variables are verified rigorously under localized tests to ensure structural decoupling.


* **The Weight of Evidence (WoE) Subset**: Maps categorical classifications and structural credit vectors directly into continuous logarithmic probability bins, establishing monotonic relationships optimal for parametric scoring cards.


* **The Principal Component Analysis (PCA) Subset**: Compresses highly fragmented feature variance by projecting multi-dimensional coordinates onto orthogonal principal components, effectively testing high-capacity model performance in compressed semantic spaces.


* **The Custom Categorical Subset**: Preserves high-cardinality nominal vectors natively, unlocking localized split optimization for tree algorithms like CatBoost.



### 2. Model Gauntlet & Algorithmic Configurations



The portfolio of estimators was thoroughly benchmarked:

* **K-Nearest Neighbors (KNN)**: Evaluated over both VIF and PCA subsets to track performance scaling over multi-distance spaces (Manhattan vs. Euclidean coordinates).


* **Logistic Regression**: Optimized explicitly under L1 (Lasso) and L2 (Ridge) penalty variables over both structural WoE matrices and orthogonal components.


* **Random Forest Classifier**: Built as a zero-bias deep tree ensemble setup to assess maximum cross-entropy limits over complex row matrices.


* **Gradient Boosted Decision Trees (GBDT)**: High-capacity parallel boosters including **XGBoost**, **LightGBM**, and **CatBoost** were trained with custom parameters to capture non-linear relationships.



### 3. Meta-Ensemble Learning (Voting & Stacking Topologies)



To break maximum localized convergence walls, candidate tree structures were merged into advanced ensemble paradigms:

* **Soft Voting Classifier**: Blends probability sequences linearly across top estimators to stabilize decision boundaries.


* **Meta-Stacking Classifier**: Collects the absolute localized prediction arrays generated by basic learners and feeds them into an optimized meta-logistic estimator, transforming predicted outputs into stable premium risk predictions.



---

## 🧠 Engine 2: Deep Learning Neural Engine (ANN)



The neural network implementation transitions the problem into a deep high-dimensional optimization task, mapping structural matrices into localized probability maps.

### 1. Architecture Synthesis & Feed-Forward Topology



The base neural configuration implements a high-capacity **Sequential Feed-Forward Neural Network (ANN)** wrapped inside automated matrix configurations:

* **Dense Fully Connected Layers**: Learns abstract geometric activations through deep hidden nodes.


* **Regularization Layers**: Infuses stochastic `Dropout` vectors across matching validation lines to prevent high-dimensional weight over-fitting.


* **Optimization Vectors**: Evaluates dynamic backpropagation variables across multi-optimizer nodes (`SGD`, `Adam`, `RMSprop`, `Adagrad`, `Adadelta`, `Nadam`).



### 2. Automated Hyperparameter Searching via Optuna



Rather than relying on sub-optimal heuristic hyperparameter selections, the architecture integrates a non-parametric Bayesian searching routine leveraging the **Optuna framework** with a tree-structured Parzen Estimator (`TPESampler`). The search objective evaluates and minimizes cross-entropy losses by exploring deep parameter boundaries:

* **Layer Budgets**: Dynamically explores structural layer counts and unit widths per step.


* **Dropout Densities**: Adjusts drop distribution weights smoothly from `0.1` up to `0.5`.


* **Activation Mappings**: Benchmarks distinct operational non-linear triggers (`ReLU`, `Sigmoid`).


* **Learning Rates**: Sweeps active propagation weights exponentially to converge models efficiently.



---

## 📊 Comprehensive Metric Architecture



To guarantee maximum consistency across financial risk modeling standards, performance calculation relies on two intertwined global metrics: the **Receiver Operating Characteristic Area Under Curve (ROC-AUC)** and the **Gini Index (Lorenz Curve Scoring Matrix)**.

$$\text{Gini Index} = 2 \times (\text{ROC-AUC}) - 1$$

### Evaluation & Resource Constraints Notice



> ⚠️ **Computational Boundary & Infrastructure Disclaimer**: All operational modeling runs, model benchmarking loops, and automated Optuna objective searching budgets were executed exclusively on local workstation boundaries and containerized cloud allocations (**Google Colab**). Due to constrained memory pipelines and restricted physical compute hardware, the optimization loops were bounded by compressed trial step counts and smaller data samples (50,000 instances). Additionally, due to potential memory constraints, the machine learning and deep learning models have been separated into two distinct notebooks. In an unconstrained high-performance cloud ecosystem with multi-node distributed GPU engines, scaling up optimization budgets would allow models to reach much higher convergence plateaus and significantly improved test metrics.
> 
> 

### Global Task-Related Thresholds Alignment



> 🎯 **Contextual Threshold Policy**: Several statistical metrics and transformation selection choices implemented across this portfolio (such as the selected 60% missing data threshold, localized variable inclusion limits, and specialized step constraints during Bayesian sampling) deviate purposefully from standard academic rules-of-thumb. These parameters represent **task-related thresholds** explicitly tailored to match the high structural imbalance, multi-collinearity profiles, and corporate scoring parameters typical of the Home Credit application data layout.
> 
> 

---

## 🚀 Execution & Project Replication Guide



### 1. File Infrastructure & Workspace Assembly



To run the dual classification engines, ensure your local framework contains the source application layout in the working directory:

```
.
├── application.csv                                                             # Home Credit Classification Source File
├── (ML) Dual-Engine Risk Architecture Advanced Credit Risk Assessment.ipynb    # Classical Machine Learning Engine Notebook
└── ((DL) Dual-Engine Risk Architecture Advanced Credit Risk Assessment.ipynb   # Deep Neural Optimization Engine Notebook

```

### 2. Environment Ingestion & Dependencies



Install the enterprise dependencies via your terminal interface or embedded execution environment cells:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow optuna catboost xgboost lightgbm shap statsmodels

```

### 3. Pipeline Ingestion Protocol



1. **Classical Engine Run**: Execute cells sequentially inside `(ML) Dual-Engine Risk Architecture Advanced Credit Risk Assessment.ipynb`. The workflow profiles input configurations, calculates WoE/PCA spaces, runs the 20+ model training matrix, and serializes the meta-ensemble combinations.


2. **Deep Neural Run**: Trigger cells inside `(DL) Dual-Engine Risk Architecture Advanced Credit Risk Assessment.ipynb`. The script initializes tracking parameters, launches the tree-structured Bayesian Parzen optimization engine via Optuna, and constructs the top candidates with early-stopping controls.
