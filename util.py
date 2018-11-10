"""
This houses all general purpose objects.

These values can be set as environment variables.
"""
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_DIR = os.path.join(ROOT_DIR, "jupy_note", "data.csv")
METRIC_SCORING = "roc_auc"
CV = 10

SAVE_MODEL = True
MODEL_PATH = os.path.join(ROOT_DIR, "model.joblib")
