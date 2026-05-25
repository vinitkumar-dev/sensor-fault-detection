import os

# Base project directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------
# Data Paths
# ---------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed.csv")

# ---------------------------
# Artifact Paths (Model outputs)
# ---------------------------
ARTIFACTS_DIR =  "artifacts"


# ---------------------------
# Logs
# ---------------------------
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE_NAME = "app.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# ---------------------------
# Model Training Config
# ---------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET_COLUMN = "target"

# ---------------------------
# General Config
# ---------------------------
FILE_ENCODING = "utf-8"

#DATABASE CONNECTION

MONGO_DATABASE_NAME = 'pwskills'
MONGO_COLLECTION_NAME ='waferfault'
TARGET_COLUMN = 'quality'
MONGO_DB_URL = 'mongodb+srv://pwskills:aBWLq7IA53wjxMpc@cluster0.o3qp0iv.mongodb.net/?appName=Cluster0'