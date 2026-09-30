"""ClimateTwin interactive Streamlit dashboard.

The dashboard is a *view* over already-trained artifacts. It never trains a
model: it loads the fitted scaler, the trained ``ClimateLSTM`` checkpoint, the
processed test split, and the evaluation metrics, then drives the existing
``src.scenario`` and ``src.impact`` engines to produce forecasts, what-if
scenarios, and human-meaningful impact indicators.

Entry point: ``dashboard/app.py`` (run with ``streamlit run dashboard/app.py``).
"""
