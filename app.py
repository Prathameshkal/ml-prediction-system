from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from ml_model import MLPredictor
from data_processor import DataProcessor
from config import config
import psutil
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = config.MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE



# Initialize ML components
ml_predictor = MLPredictor()
data_processor = DataProcessor()

# Global training status
training_status = {}

def get_system_resources():
    """Get current system resource usage"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    cpu = psutil.cpu_percent(interval=1)
    
    return {
        'memory_used_gb': memory.used / (1024**3),
        'memory_total_gb': memory.total / (1024**3),
        'memory_percent': memory.percent,
        'disk_free_gb': disk.free / (1024**3),
        'cpu_percent': cpu
    }

def validate_file_size(file):
    """Validate file size before processing"""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if file_size > config.MAX_UPLOAD_SIZE:
        return False, f"File too large: {file_size/(1024*1024):.1f}MB. Maximum allowed: {config.MAX_UPLOAD_SIZE/(1024*1024)}MB"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, "OK"

def validate_dataset_size(df):
    """Validate dataset dimensions"""
    if len(df) > config.MAX_ROWS:
        return False, f"Too many rows: {len(df)}. Maximum allowed: {config.MAX_ROWS}"
    
    if len(df.columns) > config.MAX_COLUMNS:
        return False, f"Too many columns: {len(df.columns)}. Maximum allowed: {config.MAX_COLUMNS}"
    
    # Estimate memory usage
    estimated_memory = df.memory_usage(deep=True).sum()
    if estimated_memory > config.MAX_MEMORY_USAGE:
        return False, f"Dataset too large for memory: {estimated_memory/(1024**3):.1f}GB. Maximum: {config.MAX_MEMORY_USAGE/(1024**3)}GB"
    
    return True, "OK"

def process_large_file(filepath, chunk_size=config.CHUNK_SIZE):
    """Process large files in chunks"""
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # Basic preprocessing for each chunk
        chunk = chunk.dropna(axis=1, how='all')  # Remove completely empty columns
        chunks.append(chunk)
    
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ml-prediction')
def ml_prediction_page():
    return render_template('index.html')

@app.route('/results')
def results_page():
    return render_template('index.html')


@app.route('/system-status')
def system_status():
    """Get current system resource status"""
    resources = get_system_resources()
    return jsonify({
        'resources': resources,
        'limits': {
            'max_upload_mb': config.MAX_UPLOAD_SIZE / (1024 * 1024),
            'max_rows': config.MAX_ROWS,
            'max_columns': config.MAX_COLUMNS,
            'max_memory_gb': config.MAX_MEMORY_USAGE / (1024**3)
        },
        'active_trainings': len([t for t in training_status.values() if t['status'] == 'training'])
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload endpoint called")

    if 'file' not in request.files:
        print("No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    print(f"File  received:{file.filename}")
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file size
    is_valid, message = validate_file_size(file)
    if not is_valid:
        print(f"File validation failed: {message}")
        return jsonify({'error': message}), 400
    
    # Check system resources
    resources = get_system_resources()
    if resources['memory_percent'] > 90:
        return jsonify({'error': 'System memory too high. Please try again later.'}), 400
    
    try:
        # Save uploaded file
        filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {filepath}") 
        file.save(filepath)

        #check file was saved
        if not os.path.exists(filepath):
            print("file not saved properly")
            return jsonify({'error': 'File could not be saved'}), 500
        print(f" File savved successfully. Size:{os.path.getsize(filepath)} bytes") 
        
        # Determine if we need chunked processing
        file_size = os.path.getsize(filepath)
        use_chunks = file_size > 50 * 1024 * 1024  # 50MB threshold
        
        if use_chunks:
            df = process_large_file(filepath)
        else:
            df = pd.read_csv(filepath)
        
        # Validate dataset size
        is_valid, message = validate_dataset_size(df)
        if not is_valid:
            print(f"Dataset validation failed: {message}")
            os.remove(filepath)  # Clean up
            return jsonify({'error': message}), 400
        
        # Process and analyze data
        print("analyze dataset...") 
        analysis_result = data_processor.analyze_dataset(filepath)
        print("analyze Complete") 
        
        return jsonify({
            'success': True,
            'filename': filename,
            'analysis': analysis_result,
            'preview': df.head(10).to_dict('records'),
            'columns': df.columns.tolist(),
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'size_mb': os.path.getsize(filepath) / (1024 * 1024),
                'processing_mode': 'full'
            }
        })
        
    except Exception as e:
        print(f"Unexpected Error: {str(e)}") 
        # Clean up on error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    filename = data.get('filename')
    target_column = data.get('target_column')
    model_type = data.get('model_type', 'random_forest')
    
    # Check if training already in progress
    if filename in training_status and training_status[filename]['status'] == 'training':
        return jsonify({'error': 'Training already in progress for this dataset'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Dataset file not found'}), 404
    
    # Check system resources
    resources = get_system_resources()
    if resources['memory_percent'] > 85:
        return jsonify({'error': 'System memory too high for training. Please try again later.'}), 400
    
    # Start training in background thread
    training_status[filename] = {
        'status': 'training',
        'start_time': datetime.now(),
        'progress': 0
    }
    
    def training_thread():
        try:
            # Train the model
            training_result = ml_predictor.train_model(
                filepath, 
                target_column, 
                model_type
            )
            
            training_status[filename] = {
                'status': 'completed',
                'result': training_result,
                'end_time': datetime.now()
            }
            
        except Exception as e:
            training_status[filename] = {
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now()
            }
    
    thread = threading.Thread(target=training_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Training started in background',
        'training_id': filename
    })

@app.route('/training-status/<training_id>')
def get_training_status(training_id):
    """Check training progress"""
    if training_id not in training_status:
        return jsonify({'error': 'Training not found'}), 404
    
    status = training_status[training_id]
    return jsonify({'status': status})

@app.route('/cancel-training/<training_id>', methods=['POST'])
def cancel_training(training_id):
    """Cancel ongoing training"""
    if training_id in training_status and training_status[training_id]['status'] == 'training':
        training_status[training_id]['status'] = 'cancelled'
        return jsonify({'success': True, 'message': 'Training cancelled'})
    
    return jsonify({'error': 'Training not found or not running'}), 404


# ... (other routes remain similar)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True) 
    os.makedirs(config.TEMP_FOLDER, exist_ok=True)
    os.makedirs(config.LOG_FOLDER, exist_ok=True)
    
    print(f"🚀 ML System Started with Limits:")
    print(f"📁 Max Upload Size: {config.MAX_UPLOAD_SIZE/(1024*1024)}MB")
    print(f"📊 Max Rows: {config.MAX_ROWS:,}")
    print(f"📈 Max Columns: {config.MAX_COLUMNS}")
    print(f"💾 Max Memory: {config.MAX_MEMORY_USAGE/(1024**3)}GB")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)


