# 🏗️ Concrete Compressive Strength Predictor & Linear Systems Solver

### *Supervised Machine Learning via Custom Gauss-Jordan Elimination*

**Davao Oriental State University**


---

## 📌 Project Overview

In civil and structural engineering, determining the 28-day compressive strength of concrete traditionally requires preparing cylinder specimens and waiting weeks for standard water-tank curing before testing under a Universal Testing Machine (UTM)[cite: 2].

This application demonstrates how **Machine Learning** can estimate compressive strength early from mix design parameters directly from first principles[cite: 2]. Rather than relying on commercial "black box" machine learning libraries (such as Scikit-Learn), this system implements an **algorithmic numerical engine built entirely from scratch**[cite: 2]:

1. **Custom Gauss-Jordan Solver (`gauss_jordan.py`):** Solves general $n \times n$ linear systems using **Partial Pivoting** and a **scale-aware relative singularity check**, providing step-by-step row operation auditing[cite: 2, 3].
2. **Multiple Linear Regression Layer (`linear_regression.py`):** Formulates the **Normal Equations** and reuses the custom Gauss-Jordan engine to extract parameter weights ($\beta$), compute matrix inverses $(X^T X)^{-1}$, perform hypothesis testing ($t$-stats, $p$-values), and run **Leave-One-Out Cross-Validation (LOOCV)**[cite: 2, 4].
3. **Interactive Web Dashboard (`app.py`):** A Streamlit application offering live $n \times n$ linear system solving and interactive concrete mix predictions with full mathematical derivations[cite: 1, 2].

---

## 👥 Proponents & Contributors

* **John Joshua D. Ilisan**[cite: 1, 2]
* **Darius Lape**[cite: 1, 2]
* **Briel Jan M. Lacia**[cite: 1, 2]

*Department of Civil Engineering, Davao Oriental State University*[cite: 2]

---

## 🧮 Mathematical Formulation

### 1. The Normal Equations Bridge

In concrete mix evaluation, the number of experimental batch observations ($N$) exceeds the number of parameters ($p$)[cite: 2]. This yields an **overdetermined rectangular system** ($X\beta \approx y$) with no direct matrix inverse[cite: 2].

To minimize the sum of squared prediction errors ($\sum (y - \hat{y})^2$), the Ordinary Least Squares (OLS) method formulates the **Normal Equations**[cite: 2]:

$$(X^T X)\beta = X^T y \quad \iff \quad A\beta = b$$


[cite: 1, 2]

Where:

* $X$ is the $(N \times 4)$ feature design matrix augmented with a leading column of $1\text{s}$ for the intercept ($\beta_0$)[cite: 2, 4].
* $X^T X$ compresses the rectangular matrix into a square, symmetric $(4 \times 4)$ coefficient matrix ($A$)[cite: 2].
* $X^T y$ is a $(4 \times 1)$ column vector of constants ($b$)[cite: 2].
* $\beta = [\beta_0, \beta_1, \beta_2, \beta_3]^T$ is the vector of model weights[cite: 2].

### 2. Physical Engineering Variables

$$\text{Strength (MPa)} = \beta_0 + \beta_1(\text{Cement}) + \beta_2(w/c) + \beta_3(\text{Age})$$


[cite: 1, 2]

| Parameter | Type | Unit | Engineering Interpretation |
| --- | --- | --- | --- |
| **$\beta_0$** | Intercept | $\text{MPa}$ | **Baseline Intercept**: Datum plane constant[cite: 1, 2]. |
| **$x_1$ / $\beta_1$** | Independent | $\text{kg/m}^3$ | **Cement Content**: Primary binder density[cite: 1, 2]. Expected $\beta_1 > 0$[cite: 2]. |
| **$x_2$ / $\beta_2$** | Independent | Decimal | **Water-Cement Ratio ($w/c$)**: Governed by Abrams' Law; excess water increases capillary porosity[cite: 1, 2]. Expected $\beta_2 < 0$[cite: 2]. |
| **$x_3$ / $\beta_3$** | Independent | Days | **Curing Age**: Hydration progression over time[cite: 1, 2]. Expected $\beta_3 > 0$[cite: 2]. |
| **$y$** | Dependent | $\text{MPa}$ | **Compressive Strength**: Measured ultimate failure load under UTM compression[cite: 1, 2]. |

---

## ✨ Features

* **Custom Gauss-Jordan Engine (`gauss_jordan.py`):**
* **Partial Pivoting:** Swaps rows to place the largest magnitude entry on the diagonal at each column step, preventing division by zero and minimizing floating-point roundoff errors[cite: 2, 3].
* **Scale-Aware Singularity Check:** Rejects ill-conditioned or near-singular matrices dynamically based on matrix scale rather than a static absolute tolerance[cite: 2, 3].
* **Step-by-Step Logging:** Generates audit logs for every row swap, row normalization, and row elimination step[cite: 1, 3].


* **Machine Learning & Statistical Diagnostics (`linear_regression.py`):**
* **OLS Normal Equations:** Solves $(X^T X)\beta = X^T y$ directly via Gauss-Jordan elimination[cite: 2, 4].
* **Multicollinearity Diagnostic:** Computes Variance Inflation Factors (VIF) and pairwise correlation matrices prior to model fitting to check for predictor collinearity (e.g., between cement and $w/c$ ratio)[cite: 1, 2].
* **Coefficient Uncertainty:** Inverts $X^T X$ using the custom solver to calculate Standard Errors, $t$-statistics, and $p$-values for hypothesis testing[cite: 1, 2, 4].
* **Leave-One-Out Cross-Validation (LOOCV):** Refits the model $N$ times while holding out one batch at a time to derive unbiased out-of-sample accuracy metrics ($R^2$, MAE, RMSE)[cite: 1, 2, 4].


* **Interactive Streamlit Web Interface (`app.py`):**
* **Tab 1 (Linear Systems Solver):** Solves editable $n \times n$ systems ($2 \times 2$ to $8 \times 8$) with step-by-step row operation expanders and subscript variable rendering[cite: 1].
* **Tab 2 (Concrete Strength Predictor):** Interactive training batch grid, automated Normal Equations summation breakdown table, in-sample vs. LOOCV comparison, and real-time mix design sliders[cite: 1].



---

## 📂 Project Structure

```text
├── app.py                 # Streamlit application UI, dashboards, VIF check, and sliders
├── gauss_jordan.py        # Core Gauss-Jordan solver (partial pivoting & scale-aware checks)
├── linear_regression.py   # OLS ML engine, (XᵀX)⁻¹ inversion, significance stats, & LOOCV
├── requirements.txt       # Dependencies (numpy, pandas, streamlit, scipy)
└── README.md              # Project documentation

```

### Module Responsibilities

| File | Primary Responsibility | Key Functions / Methods |
| --- | --- | --- |
| `gauss_jordan.py` | Solves $Ax = b$ via Gauss-Jordan elimination[cite: 3]. | `solve_gauss_jordan(A, b, return_steps, tol)`[cite: 3] |
| `linear_regression.py` | Builds and solves the Normal Equations[cite: 4]. | `fit()`, `predict()`, `inverse_XtX()`, `coefficient_stats()`, `loocv()`[cite: 4] |
| `app.py` | Streamlit dashboard interface[cite: 1]. | `compute_vif()`, tab views, interactive UI components[cite: 1] |

---

## 🚀 Installation & Local Execution

### Prerequisites

* Python 3.9 or higher installed.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/concrete-strength-predictor.git
cd concrete-strength-predictor

```

### 2. Set Up a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Application

```bash
streamlit run app.py

```

---

## 🔬 Validation & Results Summary

* **Solver Verification:** The custom Gauss-Jordan solver was verified against `numpy.linalg.solve` across multiple test systems, matching results to 6 decimal places[cite: 2].
* **In-Sample Fit vs. Cross-Validation:** The model achieved an in-sample $R^2 = 0.977$ on 8 training batches[cite: 2]. Leave-One-Out Cross-Validation yielded a held-out $R^2 = 0.942$ across valid folds, highlighting true predictive capacity while flagging ill-conditioned leave-one-out subsets[cite: 2].
* **Statistical Significance:** At current sample sizes ($N=8$), curing age ($\beta_3$, $p = 0.002$) is statistically significant ($p < 0.05$), while cement content ($\beta_1$) and $w/c$ ratio ($\beta_2$) align with physical expectations but require larger sample sizes to narrow their standard errors[cite: 2].
