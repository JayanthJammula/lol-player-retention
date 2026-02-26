"""
app.py
Streamlit dashboard for League of Legends Player Churn Prediction.
Uses temporal-split features: train on historical gameplay, predict future churn.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

from feature_engineering import load_and_clean, get_training_features

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LoL Player Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data & model loading (cached)
# ---------------------------------------------------------------------------
MODEL_DIR = './models'


@st.cache_data
def load_data():
    return load_and_clean()


@st.cache_resource
def load_models():
    models = {}
    models['scaler'] = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    models['logistic_regression'] = joblib.load(
        os.path.join(MODEL_DIR, 'logistic_regression.joblib'))
    models['random_forest'] = joblib.load(
        os.path.join(MODEL_DIR, 'random_forest.joblib'))
    models['xgboost'] = joblib.load(
        os.path.join(MODEL_DIR, 'xgboost_model.joblib'))

    from tensorflow.keras.models import load_model
    models['neural_network'] = load_model(
        os.path.join(MODEL_DIR, 'neural_network.keras'))
    models['lstm'] = load_model(
        os.path.join(MODEL_DIR, 'lstm_model.keras'))

    with open(os.path.join(MODEL_DIR, 'model_comparison.json')) as f:
        models['comparison'] = json.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_importance.json')) as f:
        models['feature_importance'] = json.load(f)
    with open(os.path.join(MODEL_DIR, 'training_features.json')) as f:
        models['feature_names'] = json.load(f)

    for key, fname in [('nn_history', 'nn_history.json'),
                       ('lstm_history', 'lstm_history.json'),
                       ('cv_results', 'cv_results.json'),
                       ('learning_curves', 'learning_curves.json'),
                       ('leakage_check', 'leakage_check.json'),
                       ('data_quality', 'data_quality.json')]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                models[key] = json.load(f)

    return models


df = load_data()
models = load_models()
training_features = get_training_features()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Feature Analysis",
    "Model Comparison",
    "Live Prediction",
    "Player Explorer",
])

# ---------------------------------------------------------------------------
# Helper: predict with tabular models (not LSTM — needs sequences)
# ---------------------------------------------------------------------------
TABULAR_MODELS = {
    'Logistic Regression': 'logistic_regression',
    'Random Forest': 'random_forest',
    'XGBoost': 'xgboost',
    'Dense Neural Network': 'neural_network',
}


def predict_proba_tabular(model_key, X_scaled):
    """Return churn probability for scaled tabular input."""
    model = models[model_key]
    if model_key == 'neural_network':
        return model.predict(X_scaled, verbose=0).flatten()
    else:
        return model.predict_proba(X_scaled)[:, 1]


# ===================================================================
# PAGE 1: Overview
# ===================================================================
if page == "Overview":
    st.title("League of Legends — Player Churn Prediction")

    st.markdown("""
    This dashboard predicts whether a League of Legends player will **churn**
    (not return within 6 hours) based on their **historical gameplay patterns**.
    The pipeline uses a **temporal split**: features are computed from a player's
    first 70% of games, and the churn label is determined by whether they return
    for the remaining 30%.
    """)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Players", f"{len(df):,}")
    col2.metric("Churn Rate", f"{df['churn'].mean():.1%}")
    col3.metric("Features", len(training_features))
    col4.metric("Models", 5)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Class Distribution")
        dist = df['churn'].value_counts().reset_index()
        dist.columns = ['Status', 'Count']
        dist['Status'] = dist['Status'].map({0: 'Active', 1: 'Churned'})
        fig = px.bar(
            dist, x='Status', y='Count', color='Status',
            color_discrete_map={'Active': '#2ecc71', 'Churned': '#e74c3c'},
            text_auto=True,
        )
        fig.update_layout(showlegend=False, yaxis_title='Number of Players')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Dataset Summary")
        summary = df[training_features + ['churn']].describe().T
        summary = summary[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)

    with st.expander("Methodology: Temporal Split"):
        st.markdown("""
        **Why temporal split?** Traditional churn models risk data leakage when
        features and labels come from the same time window. Our approach:

        1. **Feature window** (first 70% of games): All 20 features are computed
           only from historical gameplay: KDA, win rate, champion diversity,
           play frequency, and performance trends.
        2. **Label window** (remaining 30%): If the gap between the last
           feature-window game and the next game exceeds **6 hours**, the player
           is labeled as churned.
        3. **Minimum 10 games** per player ensures enough data for meaningful
           features.

        This eliminates circular reasoning and produces honest, generalizable
        predictions.
        """)

    # Data quality checks
    dq = models.get('data_quality')
    if dq:
        st.divider()
        st.subheader("Data Quality Checks")
        dq_rows = []
        for check in dq['checks']:
            dq_rows.append({
                'Check': check['name'],
                'Status': 'PASS' if check['passed'] else 'FAIL',
            })
        dq_df = pd.DataFrame(dq_rows)

        def style_status(val):
            color = '#2ecc71' if val == 'PASS' else '#e74c3c'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            dq_df.style.map(style_status, subset=['Status']),
            use_container_width=True, hide_index=True,
        )
        n_pass = sum(1 for c in dq['checks'] if c['passed'])
        st.caption(f"{n_pass}/{len(dq['checks'])} checks passed — "
                   f"{dq['total_players']} players validated")


# ===================================================================
# PAGE 2: Feature Analysis
# ===================================================================
elif page == "Feature Analysis":
    st.title("Feature Analysis")

    # Feature grouped selector
    feature_groups = {
        'Performance': ['win_rate', 'kda', 'kill_death_ratio', 'avg_damage',
                        'avg_cs', 'avg_vision_score'],
        'Game Context': ['avg_game_duration', 'avg_champion_level',
                         'avg_gold_earned', 'gold_efficiency', 'unique_champions'],
        'Engagement': ['total_games_played', 'unique_play_days',
                       'avg_time_between_games_hrs', 'median_time_between_games_hrs',
                       'play_frequency', 'feature_window_days'],
        'Trends': ['kda_trend', 'winrate_trend', 'last_gap_days'],
    }

    group = st.selectbox("Feature Group", list(feature_groups.keys()))
    feature = st.selectbox("Feature", feature_groups[group])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution by Class")
        plot_df = df[[feature, 'churn']].copy()
        plot_df['Status'] = plot_df['churn'].map({0: 'Active', 1: 'Churned'})
        fig = px.histogram(
            plot_df, x=feature, color='Status', barmode='overlay',
            opacity=0.7, marginal='box',
            color_discrete_map={'Active': '#2ecc71', 'Churned': '#e74c3c'},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Box Plot Comparison")
        fig = px.box(
            plot_df, x='Status', y=feature, color='Status',
            color_discrete_map={'Active': '#2ecc71', 'Churned': '#e74c3c'},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Correlation Heatmap")
        corr = df[training_features + ['churn']].corr()
        fig = px.imshow(
            corr, text_auto='.2f',
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            aspect='auto',
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Feature Importance (Random Forest)")
        fi = models['feature_importance']
        fi_df = pd.DataFrame({
            'Feature': list(fi.keys()),
            'Importance': list(fi.values()),
        }).sort_values('Importance', ascending=True)
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues')
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # T-test results
    st.divider()
    st.subheader("Statistical Tests (T-Test: Active vs Churned)")
    from scipy.stats import ttest_ind
    churn_df = df[df['churn'] == 1]
    active_df = df[df['churn'] == 0]
    ttest_rows = []
    for feat in training_features:
        t_stat, p_val = ttest_ind(
            churn_df[feat].dropna(), active_df[feat].dropna()
        )
        ttest_rows.append({
            'Feature': feat,
            'T-Statistic': round(t_stat, 4),
            'P-Value': f"{p_val:.2e}",
            'Significant (p<0.05)': 'Yes' if p_val < 0.05 else 'No',
        })
    st.dataframe(pd.DataFrame(ttest_rows), use_container_width=True,
                 hide_index=True)

    with st.expander("About the Temporal Split"):
        st.markdown("""
        All features are computed from the **first 70% of each player's games**
        (the feature window). The churn label is determined from the **remaining
        30%** (the label window). This means:

        - `avg_time_between_games_hrs` is **safe to use** — it measures gaps
          within the historical window, not the gap that defines churn.
        - Trend features (`kda_trend`, `winrate_trend`) capture whether a player's
          performance was improving or declining before the prediction point.
        - There is **zero data leakage** between features and labels.
        """)

        # Leakage check results
        lc = models.get('leakage_check')
        if lc:
            st.markdown("---")
            st.markdown("**Programmatic Leakage Verification**")
            if lc['passed']:
                st.success(f"All {lc['n_players_checked']} sampled players "
                           f"passed integrity checks.")
            else:
                st.error("Some integrity checks failed!")

            lc_rows = []
            for check in lc['checks']:
                lc_rows.append({
                    'Player': check['puuid_prefix'],
                    'Games': check['total_games'],
                    'Feature Games': check['feature_games'],
                    'Monotonic': 'PASS' if check['monotonic_timestamps'] else 'FAIL',
                    'No Overlap': 'PASS' if check['no_temporal_overlap'] else 'FAIL',
                    'Churn Correct': 'PASS' if check['churn_label_correct'] else 'FAIL',
                    'Count Correct': 'PASS' if check['game_count_correct'] else 'FAIL',
                })
            st.dataframe(pd.DataFrame(lc_rows), use_container_width=True,
                         hide_index=True)


# ===================================================================
# PAGE 3: Model Comparison
# ===================================================================
elif page == "Model Comparison":
    st.title("Model Comparison")

    comparison = models['comparison']

    # Metrics table
    st.subheader("Performance Metrics")
    rows = []
    for name, m in comparison.items():
        rows.append({
            'Model': name,
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1 Score': m['f1'],
            'ROC-AUC': m['roc_auc'],
        })
    metrics_df = pd.DataFrame(rows).set_index('Model')

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #d4efdf' if v else '' for v in is_max]

    st.dataframe(
        metrics_df.style.apply(highlight_max).format("{:.4f}"),
        use_container_width=True,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curves")
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, (name, m) in enumerate(comparison.items()):
            fig.add_trace(go.Scatter(
                x=m['fpr'], y=m['tpr'], mode='lines',
                name=f"{name} (AUC={m['roc_auc']:.3f})",
                line=dict(color=colors[i % len(colors)]),
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random', line=dict(dash='dash', color='gray'),
        ))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confusion Matrices")
        model_select = st.selectbox(
            "Select model", list(comparison.keys()))
        cm = np.array(comparison[model_select]['confusion_matrix'])
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale='Blues',
            x=['Predicted Active', 'Predicted Churned'],
            y=['Actual Active', 'Actual Churned'],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Confidence intervals
    st.subheader("Bootstrap 95% Confidence Intervals")
    ci_rows = []
    for name, m in comparison.items():
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            ci = m.get(f'{metric}_ci', [0, 0])
            ci_rows.append({
                'Model': name,
                'Metric': metric.capitalize(),
                'Value': m[metric],
                'CI Lower': ci[0],
                'CI Upper': ci[1],
            })
    ci_df = pd.DataFrame(ci_rows)
    fig = px.scatter(
        ci_df, x='Value', y='Model', color='Metric',
        error_x_minus=ci_df['Value'] - ci_df['CI Lower'],
        error_x=ci_df['CI Upper'] - ci_df['Value'],
        height=400,
    )
    fig.update_layout(xaxis_title='Score', xaxis_range=[0.0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    # Training history
    st.divider()
    st.subheader("Neural Network Training History")
    hist_tab1, hist_tab2 = st.tabs(["Dense NN", "LSTM (Sequential)"])

    for tab, key, label in [
        (hist_tab1, 'nn_history', 'Dense NN'),
        (hist_tab2, 'lstm_history', 'LSTM (Sequential)'),
    ]:
        with tab:
            if key in models:
                hist = models[key]
                epochs = list(range(1, len(hist['accuracy']) + 1))
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=epochs, y=hist['accuracy'],
                        name='Train', mode='lines'))
                    fig.add_trace(go.Scatter(
                        x=epochs, y=hist['val_accuracy'],
                        name='Validation', mode='lines'))
                    fig.update_layout(
                        title=f'{label} — Accuracy',
                        xaxis_title='Epoch', yaxis_title='Accuracy')
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=epochs, y=hist['loss'],
                        name='Train', mode='lines'))
                    fig.add_trace(go.Scatter(
                        x=epochs, y=hist['val_loss'],
                        name='Validation', mode='lines'))
                    fig.update_layout(
                        title=f'{label} — Loss',
                        xaxis_title='Epoch', yaxis_title='Loss')
                    st.plotly_chart(fig, use_container_width=True)

    with st.expander("About the LSTM Architecture"):
        st.markdown("""
        The LSTM receives **proper sequential input**: each player's last 20
        games from their feature window, with 11 per-game features (kills,
        deaths, assists, gold, damage, vision, etc.).

        Input shape: **(N, 20, 11)** — 20 timesteps of 11 features per player.
        This allows the LSTM to learn **temporal patterns**: declining
        performance over recent games, increasing gaps, or changing playstyles.

        The model uses a Bidirectional LSTM with masking for variable-length
        sequences (zero-padded for players with fewer than 20 games in their
        feature window).
        """)

    # --- Cross-Validation Results ---
    cv = models.get('cv_results')
    if cv:
        st.divider()
        st.subheader("Cross-Validation Results (5-Fold Stratified)")
        st.markdown("Proves generalization — each player appears in the test fold "
                     "exactly once. Mean ± std across 5 folds.")

        cv_rows = []
        for name, res in cv.items():
            cv_rows.append({
                'Model': name,
                'Accuracy': f"{res['mean']['accuracy']:.3f} ± {res['std']['accuracy']:.3f}",
                'Precision': f"{res['mean']['precision']:.3f} ± {res['std']['precision']:.3f}",
                'Recall': f"{res['mean']['recall']:.3f} ± {res['std']['recall']:.3f}",
                'F1': f"{res['mean']['f1']:.3f} ± {res['std']['f1']:.3f}",
                'ROC-AUC': f"{res['mean']['roc_auc']:.3f} ± {res['std']['roc_auc']:.3f}",
            })
        st.dataframe(pd.DataFrame(cv_rows).set_index('Model'),
                      use_container_width=True)

        # Per-fold detail chart
        fold_data = []
        for name, res in cv.items():
            for fold in res['per_fold']:
                fold_data.append({
                    'Model': name,
                    'Fold': fold['fold'],
                    'F1': fold['f1'],
                    'ROC-AUC': fold['roc_auc'],
                })
        fold_df = pd.DataFrame(fold_data)

        col_cv1, col_cv2 = st.columns(2)
        with col_cv1:
            fig = px.bar(fold_df, x='Model', y='F1', color='Model',
                         barmode='group',
                         title='F1 Score per Fold',
                         hover_data=['Fold'])
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col_cv2:
            fig = px.bar(fold_df, x='Model', y='ROC-AUC', color='Model',
                         barmode='group',
                         title='ROC-AUC per Fold',
                         hover_data=['Fold'])
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    # --- Precision-Recall Curves ---
    has_pr = any('pr_precision' in m for m in comparison.values())
    if has_pr:
        st.divider()
        st.subheader("Precision-Recall Curves")
        col_pr, col_pr_info = st.columns([2, 1])
        with col_pr:
            fig = go.Figure()
            colors_pr = px.colors.qualitative.Set2
            for i, (name, m) in enumerate(comparison.items()):
                if 'pr_precision' in m:
                    ap = m.get('avg_precision', 0)
                    fig.add_trace(go.Scatter(
                        x=m['pr_recall'], y=m['pr_precision'], mode='lines',
                        name=f"{name} (AP={ap:.3f})",
                        line=dict(color=colors_pr[i % len(colors_pr)]),
                    ))
            churn_rate = df['churn'].mean()
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[churn_rate, churn_rate], mode='lines',
                name=f'No Skill (AP={churn_rate:.3f})',
                line=dict(dash='dash', color='gray'),
            ))
            fig.update_layout(
                xaxis_title='Recall', yaxis_title='Precision',
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_pr_info:
            st.markdown("""
            **Why Precision-Recall?**

            With a **22.6% churn rate**, ROC curves can overstate performance.
            PR curves focus specifically on the minority class (churned players).

            The dashed line is the no-skill baseline always predicting churn
            at the class prevalence rate.

            **Average Precision (AP)** summarizes the PR curve as a single number.
            Higher is better.
            """)

    # --- Learning Curves ---
    lc_data = models.get('learning_curves')
    if lc_data:
        st.divider()
        st.subheader("Learning Curves")
        st.markdown("Shows how F1 score changes with training set size. "
                     "A **converging gap** suggests more data may help; "
                     "a **flat test curve** suggests the model needs better features.")

        lc_cols = st.columns(len(lc_data))
        for col, (name, lc) in zip(lc_cols, lc_data.items()):
            with col:
                fig = go.Figure()
                sizes = lc['train_sizes']

                # Train score line with shading
                fig.add_trace(go.Scatter(
                    x=sizes, y=lc['train_mean'], mode='lines+markers',
                    name='Train F1', line=dict(color='#3498db'),
                ))
                fig.add_trace(go.Scatter(
                    x=sizes + sizes[::-1],
                    y=[m + s for m, s in zip(lc['train_mean'], lc['train_std'])] +
                      [m - s for m, s in zip(lc['train_mean'][::-1], lc['train_std'][::-1])],
                    fill='toself', fillcolor='rgba(52,152,219,0.15)',
                    line=dict(width=0), showlegend=False,
                ))

                # Test score line with shading
                fig.add_trace(go.Scatter(
                    x=sizes, y=lc['test_mean'], mode='lines+markers',
                    name='CV Test F1', line=dict(color='#e74c3c'),
                ))
                fig.add_trace(go.Scatter(
                    x=sizes + sizes[::-1],
                    y=[m + s for m, s in zip(lc['test_mean'], lc['test_std'])] +
                      [m - s for m, s in zip(lc['test_mean'][::-1], lc['test_std'][::-1])],
                    fill='toself', fillcolor='rgba(231,76,60,0.15)',
                    line=dict(width=0), showlegend=False,
                ))

                fig.update_layout(
                    title=name,
                    xaxis_title='Training Samples',
                    yaxis_title='F1 Score',
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE 4: Live Prediction
# ===================================================================
elif page == "Live Prediction":
    st.title("Live Prediction Demo")
    st.markdown("Adjust the sliders to simulate a player's historical stats "
                "and see the predicted churn probability.")

    # Model selector (tabular models only — LSTM needs sequences)
    model_choice = st.selectbox("Select Model", list(TABULAR_MODELS.keys()))

    st.divider()

    # Feature sliders organized by group
    feature_values = {}

    tab_perf, tab_ctx, tab_eng, tab_trend = st.tabs([
        "Performance", "Game Context", "Engagement", "Trends"
    ])

    feature_groups_sliders = {
        tab_perf: ['win_rate', 'kda', 'kill_death_ratio', 'avg_damage',
                   'avg_cs', 'avg_vision_score'],
        tab_ctx: ['avg_game_duration', 'avg_champion_level',
                  'avg_gold_earned', 'gold_efficiency', 'unique_champions'],
        tab_eng: ['total_games_played', 'unique_play_days',
                  'avg_time_between_games_hrs', 'median_time_between_games_hrs',
                  'play_frequency', 'feature_window_days'],
        tab_trend: ['kda_trend', 'winrate_trend', 'last_gap_days'],
    }

    for tab, feats in feature_groups_sliders.items():
        with tab:
            col1, col2 = st.columns(2)
            for i, feat in enumerate(feats):
                col = col1 if i % 2 == 0 else col2
                with col:
                    fmin = float(df[feat].min())
                    fmax = float(df[feat].max())
                    fmed = float(df[feat].median())
                    label = feat.replace('_', ' ').title()
                    feature_values[feat] = st.slider(
                        label, min_value=fmin, max_value=fmax, value=fmed,
                        key=f"slider_{feat}",
                    )

    st.divider()

    # Prediction
    input_arr = np.array([[feature_values[f] for f in training_features]])
    scaled = models['scaler'].transform(input_arr)
    model_key = TABULAR_MODELS[model_choice]
    prob = predict_proba_tabular(model_key, scaled)[0]
    prediction = "Churned" if prob >= 0.5 else "Active"

    col_gauge, col_info = st.columns([2, 1])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': '%'},
            title={'text': "Churn Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#e74c3c' if prob >= 0.5 else '#2ecc71'},
                'steps': [
                    {'range': [0, 30], 'color': '#d5f5e3'},
                    {'range': [30, 70], 'color': '#fdebd0'},
                    {'range': [70, 100], 'color': '#fadbd8'},
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 3},
                    'thickness': 0.75,
                    'value': 50,
                },
            },
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("### Prediction")
        if prediction == "Churned":
            st.error(f"**{prediction}** — This player profile indicates a "
                     f"high likelihood of churn.")
        else:
            st.success(f"**{prediction}** — This player profile indicates "
                       f"continued engagement.")

        st.markdown("---")
        st.markdown("**Input Summary**")
        for feat in training_features:
            pct = (df[feat] < feature_values[feat]).mean() * 100
            st.caption(f"{feat.replace('_', ' ').title()}: "
                       f"{feature_values[feat]:.2f} "
                       f"(Percentile: {pct:.0f}%)")


# ===================================================================
# PAGE 5: Player Explorer
# ===================================================================
elif page == "Player Explorer":
    st.title("Player Explorer")

    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.selectbox(
            "Churn Status", ["All", "Active", "Churned"])
    with col2:
        min_games = st.number_input(
            "Min Games Played", min_value=1,
            max_value=int(df['total_games_played'].max()),
            value=1)
    with col3:
        search = st.text_input("Search PUUID (substring)")

    filtered = df.copy()
    if status_filter == "Active":
        filtered = filtered[filtered['churn'] == 0]
    elif status_filter == "Churned":
        filtered = filtered[filtered['churn'] == 1]
    filtered = filtered[filtered['total_games_played'] >= min_games]
    if search:
        filtered = filtered[
            filtered['puuid'].str.contains(search, case=False, na=False)]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} players")

    # Format numeric columns
    fmt = {c: "{:.2f}" for c in training_features}
    st.dataframe(
        filtered.head(200).style.format(fmt, na_rep='—'),
        use_container_width=True,
        height=350,
    )

    st.divider()

    if len(filtered) > 0:
        st.subheader("Player Detail")
        idx = st.number_input(
            "Select row index (from table above)", min_value=0,
            max_value=max(0, min(199, len(filtered) - 1)), value=0)
        player = filtered.iloc[idx]

        col_chart, col_preds = st.columns(2)

        with col_chart:
            st.markdown("**Top Feature Percentiles**")
            # Show top 10 most important features as horizontal bar chart
            fi = models['feature_importance']
            top_feats = list(fi.keys())[:10]
            pct_data = []
            for feat in top_feats:
                pct = (df[feat] < player[feat]).mean()
                pct_data.append({
                    'Feature': feat.replace('_', ' ').title(),
                    'Percentile': round(pct, 3),
                })
            pct_df = pd.DataFrame(pct_data)
            fig = px.bar(
                pct_df, x='Percentile', y='Feature', orientation='h',
                color='Percentile', color_continuous_scale='RdYlGn',
                range_x=[0, 1],
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_preds:
            st.markdown("**Per-Model Predictions**")
            player_input = np.array(
                [[player[f] for f in training_features]])
            player_scaled = models['scaler'].transform(player_input)

            pred_rows = []
            for display_name, key in TABULAR_MODELS.items():
                prob = predict_proba_tabular(key, player_scaled)[0]
                pred_rows.append({
                    'Model': display_name,
                    'Probability': f"{prob:.1%}",
                    'Prediction': 'Churned' if prob >= 0.5 else 'Active',
                })
            st.dataframe(
                pd.DataFrame(pred_rows), use_container_width=True,
                hide_index=True)

            st.markdown("---")
            actual = "Churned" if player['churn'] == 1 else "Active"
            st.markdown(f"**Actual Status:** {actual}")
            st.markdown(f"**Games Played:** {int(player['total_games_played'])}")
            st.markdown(f"**PUUID:** `{player['puuid'][:20]}...`")
    else:
        st.info("No players match the current filters.")
