"""Industrial Predictive Maintenance MLOps platform.

A stage-portable ML system for turbofan remaining-useful-life (RUL)
prediction and failure classification. The same application code runs
unmodified across three deployment stages (dev CSV -> office SQL/MongoDB
-> fully offline IIOT server); only YAML configuration changes between
stages. See docs/STAGES.md for the promotion procedure.
"""

__version__ = "0.1.0"
