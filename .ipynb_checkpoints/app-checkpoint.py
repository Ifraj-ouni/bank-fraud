import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

model = joblib.load('xgb_modelFraud.pkl')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
y_test_series = y_test.iloc[:,0]  # prendre la première colonne comme Series
st.sidebar.title("Menu")
page = st.sidebar.radio("Choisir une page", ["Évaluation modèle", "Vue transactions", "Ajouter transaction"])

if page == "Évaluation modèle":
    st.title("Évaluation du modèle XGBoost")
    
    # Probabilités
    y_pred_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    st.subheader("ROC Curve")
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, color='red', label=f'AUC={roc_auc:.3f}')
    ax.plot([0,1],[0,1], color='gray', linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Scatter plot
    st.subheader("Probabilités prédites vs valeurs réelles")
    fig2, ax2 = plt.subplots(figsize=(6,5))
    sns.scatterplot(x=y_test.values.flatten(), y=y_pred_prob, alpha=0.3, ax=ax2)
    ax2.set_xlabel('Valeur réelle')
    ax2.set_ylabel('Probabilité prédite')
    st.pyplot(fig2)

    # Feature importance
    st.subheader("Top 10 features importance")
    fig3, ax3 = plt.subplots(figsize=(6,5))
    xgb.plot_importance(model, max_num_features=10, importance_type='weight', ax=ax3)
    st.pyplot(fig3)

    # Metrics
    st.subheader("Metrics détaillés")
    st.text(f"Accuracy : {np.round((y_test_series==y_pred).mean(), 4)}")
    st.text(f"Classification report:\n{classification_report(y_test, y_pred)}")
    st.text(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    
elif page == "Vue transactions":
    st.title("Vue globale des transactions")
    st.dataframe(pd.concat([X_test, y_test], axis=1).head(50))  # montrer les 50 premières
    st.subheader("Statistiques globales")
    y_pred_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    df_stats = pd.DataFrame({
        "Total transactions": [len(X_test)],
        "Fraudes détectées": [sum(y_pred)],
        "Non-fraudes": [len(y_pred)-sum(y_pred)],
        "Taux fraude (%)": [100*sum(y_pred)/len(y_pred)]
    })
    st.table(df_stats)
elif page == "Ajouter transaction":
    st.title("Ajouter une transaction et prédire")
    input_data = {}
    st.subheader("Entrer les valeurs de la transaction")
    for col in X_test.columns:
        input_data[col] = st.number_input(col, value=0.0)

    if st.button("Prédire"):
        df_input = pd.DataFrame([input_data])
        prob = model.predict_proba(df_input)[:,1][0]
        pred_class = model.predict(df_input)[0]

        st.write(f"**Probabilité de fraude :** {prob:.4f}")
        if pred_class == 1:
            st.error("⚠️ Cette transaction est probablement une FRAUDE")
        else:
            st.success("✅ Cette transaction semble NON FRAUDE")

   