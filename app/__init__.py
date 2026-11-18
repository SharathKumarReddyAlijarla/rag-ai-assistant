"""Package marker for the manual assistant backend.

This file tells Python that the app folder is an importable package. That makes
imports such as app.config, app.rag_pipeline, and app.main work from the
backend, scripts, and command-line tools. It does not need runtime code because
the actual behavior lives in the other modules.
"""
