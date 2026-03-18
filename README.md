# 🔬 Employee Attrition Intelligence Suite

A comprehensive, interactive Streamlit dashboard performing **Descriptive, Diagnostic, Predictive, and Prescriptive analysis** on employee attrition data — answering the central question:

> **Why do employees stay or leave the organisation?**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?logo=plotly)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Models-F7931E?logo=scikit-learn)

---

## 📌 Table of Contents

- [Objective](#-objective)
- [Dataset Overview](#-dataset-overview)
- [Quick Start](#-quick-start)
- [Deploy on Streamlit Cloud](#-deploy-on-streamlit-cloud)
- [Sidebar Filters](#-sidebar-filters)
- [KPI Dashboard](#-kpi-dashboard-6-cards)
- [Tab 1 — Descriptive Analysis](#-tab-1--descriptive-analysis-what-happened)
- [Tab 2 — Diagnostic Analysis](#-tab-2--diagnostic-analysis-why-did-it-happen)
- [Tab 3 — Predictive Analysis](#-tab-3--predictive-analysis-what-will-happen)
- [Tab 4 — Prescriptive Analysis](#-tab-4--prescriptive-analysis-what-should-we-do)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🎯 Objective

Every chart, test, and model in this dashboard serves a single objective: **understanding the drivers of employee attrition**. The target variable is the `Attrition` column (Yes/No), and every visualisation cross-references it to uncover patterns, validate hypotheses, predict outcomes, and prescribe actions.

| Analysis Layer | Core Question | # Charts |
|---|---|---|
| **Descriptive** | What happened? | 18 charts |
| **Diagnostic** | Why did it happen? | 10 charts + statistical tables |
| **Predictive** | What will happen? | 5 charts + 3 ML models |
| **Prescriptive** | What should we do? | 2 charts + risk simulator + 8 recommendations |

---

## 📦 Dataset Overview

| Property | Value |
|---|---|
| **Rows** | 1,470 employees |
| **Columns** | 35 features |
| **Target Variable** | `Attrition` (Yes: 237, No: 1,233) |
| **Missing Values** | 0 |
| **Attrition Rate** | 16.1% |

**Feature categories covered:**

- **Demographics:** Age, Gender, MaritalStatus, Education, EducationField
- **Compensation:** MonthlyIncome, DailyRate, HourlyRate, MonthlyRate, PercentSalaryHike, StockOptionLevel
- **Job Characteristics:** Department, JobRole, JobLevel, JobInvolvement, BusinessTravel, OverTime
- **Satisfaction Scores:** EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction, WorkLifeBalance (all 1–4 scale)
- **Tenure & Growth:** YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, TotalWorkingYears, NumCompaniesWorked, TrainingTimesLastYear
- **Other:** DistanceFromHome, PerformanceRating, EmployeeCount (constant), StandardHours (constant), Over18 (constant)

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/employee-attrition-dashboard.git
cd employee-attrition-dashboard

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## ☁️ Deploy on Streamlit Cloud

1. Push this entire folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → connect your GitHub account
4. Select the repo, branch (`main`), and set main file path to `app.py`
5. Click **Deploy** — Streamlit Cloud auto-installs from `requirements.txt`

Your live dashboard URL will be: `https://your-app-name.streamlit.app`

---

## 🎛️ Sidebar Filters

The sidebar provides **6 dynamic filters** that update every chart, KPI, and analysis in real-time:

| Filter | Type | Description |
|---|---|---|
| **Department** | Multi-select | Sales, R&D, Human Resources |
| **Gender** | Multi-select | Male, Female |
| **Job Role** | Multi-select | All 9 roles (Sales Executive, Research Scientist, Lab Technician, etc.) |
| **OverTime** | Multi-select | Yes, No |
| **Age Range** | Slider | Min–Max age range |
| **Monthly Income** | Slider | Min–Max income range |

All filters are applied globally — every chart, KPI, and statistical test recalculates based on the filtered subset.

---

## 📟 KPI Dashboard (6 Cards)

Six headline KPI cards appear at the top of every tab:

| # | KPI | What It Shows |
|---|---|---|
| 1 | **Total Employees** | Count of employees in the current filtered view |
| 2 | **Left Organisation** | Number who left + overall attrition rate percentage |
| 3 | **Still Active** | Number retained + retention rate percentage |
| 4 | **Avg Income (Left)** | Mean monthly income of employees who left, compared to those who stayed |
| 5 | **Avg Tenure (Left)** | Mean years at company for employees who left |
| 6 | **OT Attrition Rate** | Attrition rate specifically among employees working overtime |

---

## 📊 Tab 1 — Descriptive Analysis (What Happened?)

This tab contains **18 charts** organised into 8 thematic sections. Each chart breaks down workforce characteristics by the `Attrition` column to reveal **what patterns exist** in the data.

---

### Chart 1: Overall Attrition Split (Donut Chart)

- **Type:** Plotly Pie with `hole=0.65` (donut)
- **What it shows:** The binary split of employees into Yes (left) vs No (stayed), with the overall attrition rate percentage displayed in the centre of the donut
- **How to read it:** The red segment represents leavers, the green segment represents stayers. The centre annotation shows the headline attrition rate
- **Why it matters:** Establishes the baseline attrition rate — the number every other chart tries to explain

---

### Chart 2: Drill-Down Sunburst — Department → Job Role → Attrition

- **Type:** Plotly Sunburst (3-level hierarchical)
- **What it shows:** Employees nested as Department (inner ring) → Job Role (middle ring) → Attrition Yes/No (outer ring)
- **Interactive feature:** Click any segment to drill into it. Clicking "Sales" zooms into Sales job roles and their attrition splits. Click the centre to zoom back out
- **Why it matters:** Reveals which specific department-role combinations suffer the highest attrition. For example, Sales Representatives within the Sales department may show a disproportionately high red (Yes) segment compared to other roles

---

### Chart 3: Age Distribution by Attrition (Overlaid Histogram)

- **Type:** Plotly histogram with `barmode='overlay'`
- **What it shows:** The age distribution of employees who left (red) overlaid on those who stayed (green), with 30 bins
- **How to read it:** Where the red bars are relatively taller compared to the green, that age group has disproportionate attrition
- **Why it matters:** Identifies whether attrition is age-dependent — typically younger employees (25–35) show higher attrition spikes

---

### Chart 4: Gender vs Attrition (Grouped Bar)

- **Type:** Plotly grouped bar chart
- **What it shows:** Count of employees who left vs stayed, grouped by Male and Female
- **How to read it:** Compare the relative height of red vs green bars within each gender to assess if attrition is gender-biased
- **Why it matters:** Tests whether there's a gender-specific attrition pattern or if it's roughly proportional

---

### Chart 5: Marital Status vs Attrition (Grouped Bar)

- **Type:** Plotly grouped bar chart
- **What it shows:** Count of leavers vs stayers across Single, Married, and Divorced categories
- **How to read it:** Single employees typically show a higher red-to-green ratio, suggesting they leave at higher rates
- **Why it matters:** Marital status acts as a proxy for organisational attachment — single employees may have fewer anchoring commitments

---

### Chart 6: Attrition Rate by Age Group (Dual-Axis Combo)

- **Type:** Overlaid bar chart + line chart with secondary Y-axis
- **What it shows:** Total employees and leavers as bars (left axis), with the attrition rate percentage as a yellow line with data labels (right axis). Age groups: 18–25, 26–35, 36–45, 46–55, 56–60
- **How to read it:** The bars show absolute volumes; the line shows the rate. Even if a group has few leavers in absolute terms, the rate might be high (e.g., 56–60 age group)
- **Why it matters:** Separates volume from rate — a group can have low absolute attrition but a dangerously high rate

---

### Chart 7: Attrition Rate by Income Group (Dual-Axis Combo)

- **Type:** Overlaid bar chart + line chart with secondary Y-axis
- **What it shows:** Same dual-axis format as Chart 6 but segmented by monthly income brackets: <$3K, $3K–$6K, $6K–$10K, $10K+
- **How to read it:** Low-income brackets (<$3K) typically show the steepest attrition rates, with the rate declining as income increases
- **Why it matters:** Directly quantifies the relationship between compensation and attrition — critical for prescriptive salary interventions

---

### Chart 8: Satisfaction Radar — Left vs Stayed (Polar/Spider Chart)

- **Type:** Plotly Scatterpolar with filled areas
- **What it shows:** Five satisfaction dimensions plotted on a radar: Environment Satisfaction, Job Satisfaction, Job Involvement, Work-Life Balance, and Relationship Satisfaction. Two overlapping polygons compare average scores for employees who left (red area) vs stayed (green area)
- **How to read it:** Where the red polygon is significantly smaller than the green, that dimension has a strong gap — meaning employees who left scored notably lower
- **Why it matters:** Provides a single-glance comparison across all satisfaction metrics. The dimensions where the gap is widest are the strongest candidates for intervention

---

### Chart 9: Satisfaction Level Distribution — Employees Who Left (Stacked Bar)

- **Type:** Plotly stacked bar chart
- **What it shows:** For each of the 5 satisfaction factors, the percentage breakdown of satisfaction levels (1=Low to 4=Very High) among employees who left only
- **How to read it:** A factor where Level 1 (Low) dominates the stack means most leavers were deeply dissatisfied on that dimension
- **Why it matters:** Complements the radar chart by showing the full distribution rather than just the mean — a factor with a high mean could still have a significant cluster of deeply dissatisfied leavers

---

### Chart 10: Tenure Distribution (Box Plot)

- **Type:** Plotly box plot with outliers
- **What it shows:** The spread (median, IQR, whiskers, outliers) of Years at Company for employees who left vs stayed
- **How to read it:** A lower median and tighter IQR for leavers indicates they tend to leave early. Outliers show long-tenured leavers
- **Why it matters:** Identifies the critical tenure window where attrition risk is highest — typically the first 2–3 years

---

### Chart 11: Years in Current Role (Box Plot)

- **Type:** Plotly box plot
- **What it shows:** Distribution of years spent in the current role, split by attrition status
- **How to read it:** If leavers have significantly lower years in their current role, it suggests role stagnation or lack of role fit drives departures
- **Why it matters:** Short role tenure among leavers may indicate poor initial role matching or inadequate onboarding

---

### Chart 12: Years Since Last Promotion (Box Plot)

- **Type:** Plotly box plot
- **What it shows:** Distribution of years since the employee's last promotion, split by attrition
- **How to read it:** Counter-intuitively, leavers may show either very short or very long promotion gaps — recently promoted employees who still leave, vs those stuck without advancement
- **Why it matters:** Tests the hypothesis that lack of career progression drives attrition

---

### Chart 13: Attrition Rate by Job Role (Horizontal Bar with Color Gradient)

- **Type:** Horizontal bar chart with gradient colour encoding (green→yellow→red based on rate)
- **What it shows:** Attrition rate percentage for each of the 9 job roles, sorted from lowest to highest. Hover reveals average income and average tenure for each role
- **How to read it:** Roles at the top (red) are the highest-risk roles. The hover tooltip provides diagnostic context — low-income roles with high attrition suggest compensation-driven departures
- **Why it matters:** Pinpoints exactly which roles to target with retention programmes. Sales Representatives and Laboratory Technicians typically top this chart

---

### Chart 14: Attrition Rate by Travel Frequency (Bar Chart)

- **Type:** Single bar chart with 3 colour-coded bars
- **What it shows:** Attrition rate (%) for Non-Travel, Travel_Rarely, and Travel_Frequently groups
- **How to read it:** Frequent travellers typically show 2–3× the attrition rate of non-travellers
- **Why it matters:** Quantifies the toll of business travel and supports policies around travel caps or compensatory benefits

---

### Chart 15: OverTime vs Attrition (Grouped Bar)

- **Type:** Grouped bar chart
- **What it shows:** Count of leavers vs stayers, split by OverTime Yes/No
- **How to read it:** The Yes-OverTime group should show a much higher red bar relative to its green bar compared to No-OverTime
- **Why it matters:** OverTime is consistently one of the top 1–2 attrition predictors — this chart makes that visually undeniable

---

### Chart 16: Distance from Home Distribution (Overlaid Histogram)

- **Type:** Histogram with overlay, 20 bins
- **What it shows:** Distribution of commute distance (km) for leavers vs stayers
- **How to read it:** If the red distribution skews higher, longer commutes correlate with higher attrition
- **Why it matters:** Tests whether commute distance is a meaningful attrition driver or a non-factor

---

### Chart 17: Monthly Income Distribution (Violin Plot)

- **Type:** Plotly violin plot with embedded box plot
- **What it shows:** Full income distribution shape for leavers vs stayers, with median and quartiles visible via the box plot overlay
- **How to read it:** The violin width shows density — if the leaver violin is fat at lower incomes and thin at higher ones, low pay drives departures
- **Why it matters:** More informative than a simple box plot — reveals bimodality or skewness in income distributions

---

### Chart 18: Income vs Experience Scatter (Bubble Chart)

- **Type:** Scatter plot with size encoding
- **What it shows:** Each employee plotted by Total Working Years (X) vs Monthly Income (Y), coloured by attrition status, with bubble size representing Job Level
- **How to read it:** Red dots (leavers) clustered at low income + low experience = early-career attrition. Scattered red dots at higher experience levels = different attrition drivers
- **Why it matters:** Reveals whether attrition is concentrated in a specific experience-income band or distributed across the workforce

---

## 🔍 Tab 2 — Diagnostic Analysis (Why Did It Happen?)

This tab contains **10 charts and 2 statistical tables** that go beyond description to test hypotheses, measure effect sizes, and identify the root causes of attrition.

---

### Chart 19: Correlation with Attrition — Point-Biserial (Horizontal Bar)

- **Type:** Horizontal bar chart with diverging colour scale (green for negative, red for positive correlation)
- **What it shows:** The point-biserial correlation coefficient between each numeric variable and the binary Attrition flag (0/1). Variables are sorted from most negatively correlated (protective) to most positively correlated (risk-increasing)
- **How to read it:** Variables on the right/red side (positive correlation) increase attrition risk. Variables on the left/green side (negative correlation) are protective. The magnitude matters — values above ±0.15 are noteworthy
- **Why it matters:** This is the single most important diagnostic chart. It quantifies the direction and strength of every numeric variable's relationship with attrition, guiding which factors to prioritise in interventions

---

### Chart 20: Feature Correlation Matrix — Heatmap

- **Type:** Plotly `imshow` heatmap with annotated correlation coefficients
- **What it shows:** Pairwise Pearson correlations between the most important variables (including Attrition_Flag), displayed as a colour-coded matrix with RdBu diverging scale
- **How to read it:** Dark red = strong positive correlation, dark blue = strong negative. Look for clusters of high inter-correlation (e.g., YearsAtCompany, YearsInCurrentRole, and YearsWithCurrManager tend to cluster)
- **Why it matters:** Exposes multicollinearity between features (important for model building) and reveals structural relationships — e.g., Job Level and Monthly Income are highly correlated, so both appearing as attrition predictors is partially redundant

---

### Chart 21: Cramér's V — Categorical Effect Sizes (Horizontal Bar)

- **Type:** Horizontal bar with gradient colour encoding
- **What it shows:** Cramér's V statistic (0–1 effect size measure) for each categorical variable's association with Attrition, derived from Chi-Square tests
- **How to read it:** Higher bars = stronger association. Values above 0.1 indicate meaningful relationships. OverTime typically ranks highest
- **Why it matters:** Complements the numeric correlations (Chart 19) by covering categorical variables that can't use correlation coefficients. Together they provide a complete picture

---

### Statistical Table: Chi-Square Test Results

- **Type:** Dataframe table
- **What it shows:** For each categorical variable: Chi² statistic, p-value, Cramér's V, and whether the association is statistically significant at α=0.05
- **How to read it:** Variables with p < 0.05 and ✅ have a statistically significant relationship with attrition. Those with ❌ are not significantly different between leavers and stayers
- **Why it matters:** Statistical rigour — ensures we don't act on patterns that could be due to random chance

---

### Chart 22: Drill-Down Sunburst — Department → OverTime → Attrition

- **Type:** 3-level Plotly Sunburst
- **What it shows:** Employees nested as Department → OverTime (Yes/No) → Attrition (Yes/No)
- **Interactive feature:** Click to drill into any segment
- **Why it matters:** Tests the hypothesis that overtime's impact on attrition varies by department. Some departments may absorb overtime better than others

---

### Chart 23: Drill-Down Sunburst — Marital Status → Gender → Attrition

- **Type:** 3-level Plotly Sunburst
- **What it shows:** Employees nested as Marital Status → Gender → Attrition
- **Interactive feature:** Click to drill in
- **Why it matters:** Examines whether the higher attrition among single employees is gender-specific or universal

---

### Chart 24: Drill-Down Treemap — Education Level → Field → Attrition

- **Type:** Plotly Treemap (3-level hierarchy)
- **What it shows:** Employees organised by Education Level (Below College to Doctor) → Education Field (Life Sciences, Medical, Marketing, etc.) → Attrition status. Area size represents count
- **Interactive feature:** Click any rectangle to zoom in
- **Why it matters:** Tests whether specific education backgrounds correlate with attrition — e.g., do Life Sciences graduates in non-science roles leave more?

---

### Chart 25: Drill-Down Sunburst — Job Level → Work-Life Balance → Attrition

- **Type:** 3-level Plotly Sunburst
- **What it shows:** Employees nested as Job Level (1–5) → Work-Life Balance rating (Bad/Good/Better/Best) → Attrition
- **Interactive feature:** Click to drill in
- **Why it matters:** Tests whether work-life balance issues are concentrated at specific job levels — entry-level employees with "Bad" WLB may leave at alarming rates while senior employees tolerate it

---

### Chart 26: Top 15 Risk Factor Combinations (Horizontal Bar)

- **Type:** Horizontal bar chart with colour gradient and sample size annotations
- **What it shows:** The 15 combinations of OverTime × Marital Status × Department with the highest attrition rates. Each bar shows the rate % and sample size (n=)
- **How to read it:** Combinations at the top are "deadly combinations" — e.g., "Yes OT | Single | Sales" might show 45%+ attrition. The sample size annotation ensures you don't over-react to small groups
- **Why it matters:** Moves from single-variable analysis to multi-factor risk profiling. These combinations directly identify which employee segments to target with retention interventions

---

### Chart 27: Average Satisfaction Scores — Stayed vs Left (Grouped Bar)

- **Type:** Grouped bar chart (green=Stayed, red=Left)
- **What it shows:** Average scores on all 5 satisfaction dimensions for employees who stayed vs left, displayed side by side
- **How to read it:** The gap between green and red bars shows the "satisfaction deficit" among leavers. Larger gaps = stronger drivers
- **Why it matters:** Translates radar chart insights into precise numbers and is accompanied by t-test results

---

### Statistical Table: Satisfaction Gap Analysis with T-Tests

- **Type:** Dataframe table
- **What it shows:** For each satisfaction factor: Left average, Stayed average, Gap, t-test p-value, and significance flag
- **How to read it:** Factors with significant p-values and large gaps are confirmed diagnostic drivers of attrition
- **Why it matters:** Validates whether observed satisfaction differences are statistically meaningful or within random variation

---

## 🤖 Tab 3 — Predictive Analysis (What Will Happen?)

This tab builds **3 machine learning models** and contains **5 charts** focused on predicting which employees are likely to leave.

---

### Chart 28: Model Comparison — Cross-Validated AUC (Bar with Error Bars)

- **Type:** Bar chart with error bars (±1 standard deviation)
- **What it shows:** Mean AUC-ROC score from 5-fold cross-validation for Logistic Regression, Random Forest, and Gradient Boosting. Class weighting is applied to handle the imbalanced target (16% attrition)
- **How to read it:** Higher bars = better discrimination between leavers and stayers. Error bars show stability across folds — tight bars = consistent model
- **Why it matters:** Establishes which model generalises best and whether we can trust predictions

---

### Chart 29: ROC Curves (Multi-Line)

- **Type:** Multi-line plot with diagonal reference
- **What it shows:** The ROC (Receiver Operating Characteristic) curve for each model — plotting True Positive Rate vs False Positive Rate at all classification thresholds. The dashed diagonal represents random guessing (AUC=0.5)
- **How to read it:** Curves closer to the top-left corner are better. The AUC value in the legend quantifies overall performance (1.0 = perfect, 0.5 = random)
- **Why it matters:** Visualises the trade-off between catching more leavers (sensitivity) and false alarms (1-specificity) at every threshold

---

### Chart 30: Top 20 Feature Importances (Horizontal Bar)

- **Type:** Horizontal bar with gradient colour encoding
- **What it shows:** The top 20 most important features for the selected model (user can switch between all 3 via dropdown). For tree-based models, this is Gini importance. For Logistic Regression, this is absolute coefficient magnitude
- **How to read it:** Top features are the strongest predictors. Compare across models using the dropdown to see which features are consistently important
- **Why it matters:** Identifies actionable levers — if Monthly Income ranks #1, compensation changes will have the biggest impact on reducing predicted attrition

---

### Chart 31: Consensus Feature Ranking — All Models (Grouped Horizontal Bar)

- **Type:** Grouped horizontal bar chart with 3 model colours
- **What it shows:** Normalised (0–1) feature importance for the top 15 features across all 3 models simultaneously. Each feature has 3 bars showing its importance in each model
- **How to read it:** Features where all 3 bars are tall are "consensus predictors" — reliable across methods. Features where only 1 bar is tall may be method-specific artefacts
- **Why it matters:** Eliminates model-specific bias. Only features validated across multiple algorithms should drive strategic decisions

---

## 💊 Tab 4 — Prescriptive Analysis (What Should We Do?)

This tab translates all findings into **actionable recommendations**, containing **1 interactive simulator**, **1 strategic matrix chart**, and **8 detailed recommendation cards**.

---

### Interactive Widget: Employee Risk Score Simulator (Gauge)

- **Type:** Plotly Indicator gauge with 8 input sliders/dropdowns
- **Inputs:** OverTime (Yes/No), Job Satisfaction (1–4), Monthly Income ($1K–$20K), Job Involvement (1–4), Work-Life Balance (1–4), Environment Satisfaction (1–4), Years at Company (0–40), Age (18–60)
- **What it shows:** A real-time risk score (0–100%) calculated from a weighted formula based on all diagnostic findings. The gauge is colour-coded: Green (0–30% Low Risk), Yellow (30–60% Medium Risk), Red (60–100% High Risk)
- **How to use it:** Adjust the sliders to simulate different employee profiles. For example, set OverTime=Yes, Income=$2,000, Satisfaction=1 to see how risk spikes. Use it to identify which factor changes would reduce risk the most
- **Why it matters:** Makes analysis actionable at the individual level — HR managers can input a specific employee's attributes to assess their flight risk

---

### 8 Strategic Recommendation Cards

Each card is colour-coded by priority (HIGH/MEDIUM/LOW) and contains data-backed rationale:

| # | Recommendation | Priority | Evidence Basis |
|---|---|---|---|
| 1 | **Overtime Management Program** | 🔴 HIGH | OT shows highest Cramér's V + top feature importance + highest correlation |
| 2 | **Compensation Realignment** | 🔴 HIGH | Income <$3K bracket shows extreme attrition rates in Chart 7 |
| 3 | **Job Involvement & Enrichment** | 🟡 MEDIUM | Top 3 predictor across all 3 ML models (Chart 31) |
| 4 | **Early Career Retention (Ages 18–35)** | 🔴 HIGH | Age group analysis (Chart 6) + tenure box plots (Chart 10) |
| 5 | **Environment & Culture Enhancement** | 🟡 MEDIUM | Significant satisfaction gap confirmed by t-tests |
| 6 | **Single Employee Engagement** | 🟢 LOW | Marital status shows elevated rates in Chart 5 + sunburst |
| 7 | **Proactive Risk Monitoring** | 🔴 HIGH | Deploy predictive model as quarterly early-warning system |
| 8 | **Stock Option & Long-Term Incentives** | 🟡 MEDIUM | StockOptionLevel=0 correlates with attrition in Chart 19 |

Each card dynamically calculates specific attrition rates from the current filtered data to justify its recommendation.

---

### Chart 32: Impact vs Cost Matrix (Bubble Scatter)

- **Type:** Scatter plot with size encoding
- **What it shows:** Each intervention plotted by Implementation Cost (X, scale 1–5) vs Estimated Attrition Rate Reduction (Y, percentage points). Bubble size represents Time to Impact in months
- **How to read it:** Top-left quadrant = high impact, low cost (prioritise these). Bottom-right = low impact, high cost (deprioritise). Smaller bubbles = faster results
- **Why it matters:** Enables evidence-based prioritisation of HR investments. Not all interventions are equal — this chart helps leadership allocate budget to the highest-ROI actions

---

## 📁 Project Structure

```
employee-attrition-dashboard/
├── app.py                    # Main Streamlit application (~500 lines)
├── EA.csv                    # Employee Attrition dataset (1,470 × 35)
├── requirements.txt          # Python dependencies (pinned versions)
├── .streamlit/
│   └── config.toml           # Dark theme + server configuration
├── .gitignore                # Python/OS excludes
└── README.md                 # This documentation
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Framework** | Streamlit 1.41 | Dashboard UI, sidebar filters, tabs, layout |
| **Visualisation** | Plotly 5.24 | All 32 interactive charts (sunburst, radar, violin, gauges, treemaps, etc.) |
| **ML Models** | scikit-learn 1.5 | Logistic Regression, Random Forest, Gradient Boosting (with class weighting) |
| **Statistics** | SciPy 1.14 | Chi-Square tests, independent t-tests, point-biserial correlation |
| **Data Processing** | Pandas 2.2, NumPy 1.26 | Data loading, aggregation, feature engineering, label mapping |
| **Styling** | Custom CSS | Dark theme, KPI cards, insight boxes, recommendation cards, section headers |

---

## 🎨 Theme Configuration

The `.streamlit/config.toml` file enforces a dark theme:

```toml
[theme]
primaryColor = "#818cf8"              # Indigo accent
backgroundColor = "#0a0e17"           # Near-black background
secondaryBackgroundColor = "#1e293b"  # Sidebar and card backgrounds
textColor = "#e0e6ed"                 # Light grey text
```

To switch to a light theme, replace these values or delete the config file entirely to use Streamlit defaults.

---

## 🔄 How the Analysis Layers Connect

The four tabs build on each other in a logical progression:

```
DESCRIPTIVE (Tab 1)           DIAGNOSTIC (Tab 2)
"What happened?"       →      "Why did it happen?"
18 charts showing              Correlation + Chi-Square +
patterns & distributions       risk combos + significance tests
         ↓                              ↓
PREDICTIVE (Tab 3)            PRESCRIPTIVE (Tab 4)
"What will happen?"    →      "What should we do?"
3 ML models + feature          Risk simulator + 8 recommendations
importance rankings            + impact-cost prioritisation
```

Each descriptive pattern is validated diagnostically, confirmed predictively, and translated into a prescriptive action. For example:

1. **Descriptive:** Chart 15 shows overtime employees leave more
2. **Diagnostic:** Cramér's V confirms OverTime has the strongest categorical association (Chart 21)
3. **Predictive:** OverTime ranks in the top 3 features across all models (Chart 31)
4. **Prescriptive:** Recommendation #1 proposes an Overtime Management Program with estimated impact (Chart 32)

---

## 📄 License

MIT License — free to use, modify, and distribute.
