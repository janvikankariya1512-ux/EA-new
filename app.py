
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Universal Bank AI Marketing Dashboard", layout="wide")

st.title("🏦 Universal Bank Personal Loan AI Intelligence Dashboard")

df = pd.read_csv("UniversalBank.csv")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
"📊 Descriptive Analytics",
"🔎 Diagnostic Insights",
"🤖 Predictive Models",
"🎯 Prescriptive Strategy",
"📂 Batch Prediction"
])

# ---------- DESCRIPTIVE ----------
with tab1:
    st.subheader("Customer Profile Overview")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="Income", color="Personal Loan",
        color_discrete_sequence=["#FF6B6B","#1B9AAA"],
        title="Income Distribution vs Loan Acceptance")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Higher income customers show significantly higher loan acceptance rates.")

    with col2:
        fig = px.box(df, x="Education", y="Income", color="Personal Loan",
        color_discrete_sequence=["#FF6B6B","#1B9AAA"],
        title="Education Level vs Income vs Loan Acceptance")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Higher education groups tend to have greater purchasing power and loan uptake.")

    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(df, x="CCAvg", color="Personal Loan",
        title="Credit Card Spending vs Loan Acceptance")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.bar(df.groupby("Family")["Personal Loan"].mean().reset_index(),
        x="Family", y="Personal Loan",
        title="Loan Acceptance Rate by Family Size")
        st.plotly_chart(fig, use_container_width=True)

# ---------- DIAGNOSTIC ----------
with tab2:
    st.subheader("Correlation Analysis")

    corr = df.drop(columns=["ID"]).corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Insight:** Income, CCAvg and CD Account show strong positive relationships with Personal Loan acceptance.")

# ---------- MODEL TRAINING ----------
X = df.drop(columns=["Personal Loan","ID"])
y = df["Personal Loan"]

X_train, X_test, y_train, y_test = train_test_split(
X,y,test_size=0.25,random_state=42,stratify=y)

models = {
"Decision Tree":DecisionTreeClassifier(max_depth=6),
"Random Forest":RandomForestClassifier(n_estimators=200),
"Gradient Boosted Tree":GradientBoostingClassifier()
}

metrics = []
roc_data = {}

for name,model in models.items():
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    metrics.append({
    "Model":name,
    "Accuracy":accuracy_score(y_test,pred),
    "Precision":precision_score(y_test,pred),
    "Recall":recall_score(y_test,pred),
    "F1 Score":f1_score(y_test,pred)
    })

    fpr,tpr,_ = roc_curve(y_test,prob)
    roc_data[name]=(fpr,tpr)

# ---------- PREDICTIVE ----------
with tab3:

    st.subheader("Model Performance Comparison")

    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df)

    st.subheader("ROC Curve Comparison")

    fig = go.Figure()

    for model in roc_data:
        fpr,tpr = roc_data[model]
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=model))

    fig.add_shape(type="line",line=dict(dash="dash"),
    x0=0,x1=1,y0=0,y1=1)

    fig.update_layout(
    title="ROC Curve for Classification Models",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrices")

    for name,model in models.items():
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test,pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

# ---------- PRESCRIPTIVE ----------
with tab4:

    st.subheader("Feature Importance for Marketing Targeting")

    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X.columns)

    fig = px.bar(importances.sort_values(ascending=False),
    title="Top Features Influencing Loan Acceptance")

    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    **Marketing Strategy Recommendation**
    - Target high-income professionals
    - Customers with high credit card spending
    - CD account holders
    - Customers with higher education levels
    """)

# ---------- PREDICTION ----------
with tab5:

    st.subheader("Upload Customer Dataset for Prediction")

    file = st.file_uploader("Upload CSV file")

    if file:
        data = pd.read_csv(file)
        rf = models["Random Forest"]

        preds = rf.predict(data.drop(columns=["ID"],errors="ignore"))
        probs = rf.predict_proba(data.drop(columns=["ID"],errors="ignore"))[:,1]

        data["Predicted Personal Loan"] = preds
        data["Acceptance Probability"] = probs

        st.dataframe(data)

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions",csv,"loan_predictions.csv")
