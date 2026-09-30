"""Page implementations for the ClimateTwin dashboard.

Named ``pages_impl`` rather than ``pages`` on purpose: Streamlit treats a
top-level ``pages/`` directory as automatic multipage navigation. We drive
navigation ourselves from ``dashboard/app.py``, so these modules are plain
render functions invoked by the entrypoint.
"""
