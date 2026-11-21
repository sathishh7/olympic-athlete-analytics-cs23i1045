# Olympic Athlete Analytics: Big Data Analytics & Systems Hackathon

**Course:** 25CS5004 / CS 5010 Analytics & Systems of Big Data Practice (Endsem Hackathon)  
**Institution:** Indian Institute of Information Technology Design & Manufacturing Kancheepuram (IIITDM Kancheepuram)  
**Evaluation:** Maximum Marks: 50  
**Dataset:** Olympic Athlete Performance Dataset (Kaggle: `jaberimohamedhabib/olympic-athlete-performance-dataset`)  

---

## 1. Project Overview & Scope

This project delivers an end-to-end descriptive, association, predictive, and distributed big-data analytics pipeline over 112 years of modern Olympic history (1896–2008). 

### Key Academic Objectives Fulfilled
1. **Exhaustive Data Auditing & Preprocessing:** Identification and removal of 4 preamble metadata rows, deduplication of redundant athlete records, rectification of gender attribution inconsistencies, and derivation of chronological era groupings.
2. **12 Distinct Publication-Grade Visualizations:** Exploration of participation expansion, gender inclusivity, Pareto concentration, geopolitical specialization, host nation advantage, and medal tier breakdowns, each paired with empirical inferences.
3. **Association Rule Mining with Apriori:** Transaction definition based on discrete performance profiles, parameter sensitivity analysis, and extraction of high-lift geopolitical/discipline rules.
4. **Maximal Frequent Itemset Mining with MAFIA:** Genuine, pristine implementation of the vertical bitmap MAFIA algorithm utilizing Head Union Tail (HUT) superset pruning and Parent Equivalence Pruning (PEP).
5. **Predictive Modeling & Academic Reliability Safeguards:**
   - Explicit recognition that the dataset contains **medalist records only** ($N=29,215$); the predictive objective is framed strictly as multiclass **Medal Tier (Gold vs. Silver vs. Bronze)** prediction conditional on reaching the podium.
   - Prevention of target leakage via **GroupShuffleSplit partitioned by Athlete**, ensuring no athlete's multi-medal career spans both train and test folds.
   - Comprehensive benchmarking against a majority-class baseline with multiclass confusion matrix, precision, recall, and macro/weighted F1 metrics.
   - Complementary structural archetyping via **K-Modes Categorical Clustering** ($K=4$).
6. **Distributed PySpark Implementation & Fair Benchmarking:**
   - Implementation of distributed association mining via **Spark ML FP-Growth** and multiclass classification via **Spark ML DecisionTreeClassifier**.
   - Controlled, multi-iteration benchmarking comparing local execution (C-optimized Scikit-Learn and mlxtend) against local PySpark (`local[4]`).
   - Honest architectural interpretation of single-machine Spark communication overhead vs. true horizontal cluster scaling.

---

## 2. Directory Structure

```text
Olympic_Hackathon_Submission/
├── Final_Report.pdf                  # Comprehensive 50-mark academic report (centerpiece)
├── README.md                         # Detailed project guide, documentation & run instructions
├── requirements.txt                  # Strict Python package dependency manifest
├── analysis.ipynb                    # Executed, self-contained, reproducible Jupyter Notebook
├── data/
│   ├── olympics.csv                  # Raw dataset (including 4 preamble lines)
│   ├── cleaned_olympics.csv          # Cleaned dataset (29,215 records, 11 attributes)
│   ├── data_dictionary.csv           # Structured attribute schema and types
│   └── data_dictionary.md            # Markdown data dictionary table
├── figures/                          # 13 high-resolution standalone figure files (300 DPI)
│   ├── figure_01_participation_evolution.png
│   ├── figure_02_gender_inclusivity_trajectory.png
│   ├── figure_03_top_15_nations_diverging_lollipop.png
│   ├── figure_04_medal_mix_top_10_nations.png
│   ├── figure_05_pareto_sports_distribution.png
│   ├── figure_06_cross_sport_gender_heatmap.png
│   ├── figure_07_host_nation_advantage.png
│   ├── figure_08_top_all_time_olympians.png
│   ├── figure_09_discipline_noc_specialization_matrix.png
│   ├── figure_10_event_gender_distribution_eras.png
│   ├── figure_11_team_vs_individual_medal_pie.png
│   ├── figure_12_apriori_rule_metrics_scatter.png
│   └── figure_13_decision_tree_architecture.png
├── results/                          # Computed output tables, metrics, and models
│   ├── data_quality_summary.csv
│   ├── visualizations_inferences.csv
│   ├── visualizations_inferences.json
│   ├── apriori_sensitivity_analysis.csv
│   ├── apriori_rules.csv
│   ├── mafia_maximal_itemsets.csv
│   ├── classifier_metrics.csv
│   ├── confusion_matrix.csv
│   ├── decision_tree_if_then_rules.txt
│   ├── kmodes_elbow_costs.csv
│   ├── kmodes_profiles.csv
│   ├── spark_metrics.json
│   ├── spark_benchmark_comparison.csv
│   ├── spark_fpgrowth_rules.csv
│   └── spark_decision_tree_rules.txt
└── src/                              # Clean, modular Python source scripts
    ├── analysis.py                   # Local preprocessing, visualization, ARM, MAFIA, DT, K-Modes
    └── pyspark_analysis.py           # PySpark ML pipeline and local vs Spark benchmarking
```

---

## 3. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Tested on Python 3.12.12 macOS Apple Silicon)
- Java Runtime Environment (OpenJDK 17 recommended for PySpark 3.5 / 4.x)

### Setup Commands
```bash
# 1. Create and activate a virtual environment
python3 -m venv olympics-env
source olympics-env/bin/activate

# 2. Install required packages
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure JAVA_HOME (example for macOS Homebrew OpenJDK 17)
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
```

---

## 4. Execution & Reproducibility Guide

All steps can be executed sequentially or independently. Every script reads from and writes to the standardized submission directories.

### Option A: Complete Pipeline Execution via Python Scripts
```bash
# Step 1: Run core analysis, data cleaning, visualizations, Apriori, MAFIA, and Decision Tree
python src/analysis.py

# Step 2: Run PySpark distributed techniques and performance benchmark
python src/pyspark_analysis.py
```

### Option B: Interactive Execution via Jupyter Notebook
Launch Jupyter Lab or Notebook to run the fully annotated analysis top-to-bottom:
```bash
jupyter notebook analysis.ipynb
```

---

## 5. Algorithm Parameter Settings & Methodological Justifications

| Algorithm | Selected Hyperparameters | Methodological Justification |
| :--- | :--- | :--- |
| **Apriori Association Mining** | `min_support = 0.02` (2.0%, 584 records)<br>`min_confidence = 0.40` (40%) | Balances rule generalizability against statistical specificity. Captures significant cross-era and sport-gender dependencies without generating trivial combinations. Tested across sensitivity grid (Support: 0.015–0.05, Confidence: 0.30–0.60). |
| **MAFIA Maximal Mining** | `min_support = 0.03` (3.0%, 877 records)<br>Vertical Bitvector representation | Discovers the true maximal frequent itemset frontier. Employs Head Union Tail (HUT) superset pruning and Parent Equivalence Pruning (PEP) to bypass explosive exponential subset enumeration. Yields 63 unique MFIs. |
| **Decision Tree Classifier** | `criterion = 'gini'`<br>`max_depth = 5`<br>`min_samples_split = 40`<br>`min_samples_leaf = 20` | Depth constrained to 5 levels to prevent severe overfitting on high-cardinality categorical features (NOC, Sport) and guarantee human-interpretable if-then classification rules. |
| **Train/Test Partitioning** | `GroupShuffleSplit(n_splits=1, test_size=0.20)`<br>`Group = Athlete` | Prevents target leakage. Multi-medal athletes (e.g., Phelps, Latynina) have all their career records confined strictly to either train or test sets. |
| **K-Modes Clustering** | `n_clusters = 4`<br>`init = 'Huang'`<br>`n_init = 5` | Minimizes matching dissimilarity across categorical features (`Sport`, `NOC`, `Gender`, `Era`, `Medal`). Justified by cost elbow curvature between $K=2$ and $K=6$. |
| **PySpark Architecture** | `master = 'local[4]'`<br>`spark.driver.memory = '2g'`<br>`spark.sql.shuffle.partitions = '4'` | Local distributed simulation across 4 CPU cores utilizing Spark MLlib native estimators (`FPGrowth`, `DecisionTreeClassifier`). |

---

## 6. Key Empirical Findings & Systems Insights

1. **Analytical Scope & Limitation:** The dataset contains 29,215 medalist records. It is mathematically invalid to frame a binary classification model predicting whether an athlete wins a medal (0 non-medalist records exist). The valid target is multiclass **Medal Tier (Gold vs. Silver vs. Bronze)**.
2. **Predictive Performance:** The Decision Tree achieves **39.79% accuracy** (Macro F1: 0.3255), outperforming the majority-class baseline (**32.91% accuracy**). The limited performance is theoretically expected: conditional on reaching an Olympic podium, the difference between Gold, Silver, and Bronze is determined by hundredths of a second or single judging points—variance that cannot be fully explained by macro-level country and sport metadata.
3. **Big Data Systems Benchmark:**
   - **Association Mining:** Local Apriori (0.0287 s) is **16.4x faster** than PySpark FP-Growth (0.4705 s).
   - **Decision Tree:** Local Scikit-Learn (0.1218 s) is **12.2x faster** than PySpark Decision Tree (1.4903 s).
   - **Architectural Conclusion:** On a 2.4 MB dataset (29,215 rows), PySpark incurs substantial fixed overheads (JVM socket IPC, Py4J serialization, Catalyst optimizer DAG creation, task scheduling). Single-machine PySpark demonstrates distributed coordination costs without the horizontal cluster throughput gains realized on multi-node multi-gigabyte workloads.
