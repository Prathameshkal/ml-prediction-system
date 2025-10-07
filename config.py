import os

class Config:
    # File size limits (in bytes)
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_TRAINING_SIZE = 500 * 1024 * 1024  # 500MB for training
    MAX_MEMORY_USAGE = 2 * 1024 * 1024 * 1024  # 2GB RAM limit
    
    # Performance settings
    CHUNK_SIZE = 10000  # Rows per chunk for large files
    MAX_ROWS = 1000000  # 1 million rows max
    MAX_COLUMNS = 1000  # 1000 columns max
    
    # Model settings
    MAX_TRAINING_TIME = 3600  # 1 hour max training time
    BATCH_SIZE = 1000
    
    # Storage paths
    UPLOAD_FOLDER = 'uploads'
    MODEL_FOLDER = 'models'
    TEMP_FOLDER = 'temp'
    LOG_FOLDER = 'logs'

config = Config()