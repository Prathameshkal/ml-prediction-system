// ML Prediction functionality
let currentDataset = null;
let currentModel = null;
let trainingInterval = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeMLPrediction();
});

function initializeMLPrediction() {
    // File upload handling
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    
    if (fileInput && uploadArea) {
        fileInput.addEventListener('change', handleFileUpload);
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('drop', handleFileDrop);
    }
    
    // Check system status on load
    checkSystemStatus();
}

function handleDragOver(e) {
    e.preventDefault();
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.style.background = 'rgba(0, 200, 255, 0.1)';
    }
}

function handleFileDrop(e) {
    e.preventDefault();
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.style.background = '';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            processFile(files[0]);
        }
    }
}

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

function processFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a CSV file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    showUploadProgress();

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentDataset = data.filename;
            displayDataAnalysis(data.analysis, data.preview, data.dataset_info);
            populateTargetColumn(data.analysis.basic_info.columns);
            hideUploadProgress();
        } else {
            throw new Error(data.error);
        }
    })
    .catch(error => {
        hideUploadProgress();
        alert('Error uploading file: ' + error.message);
    });
}

function showUploadProgress() {
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadArea = document.getElementById('uploadArea');
    
    if (uploadProgress) uploadProgress.style.display = 'block';
    if (uploadArea) uploadArea.style.display = 'none';
}

function hideUploadProgress() {
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadArea = document.getElementById('uploadArea');
    
    if (uploadProgress) uploadProgress.style.display = 'none';
    if (uploadArea) uploadArea.style.display = 'block';
}

function displayDataAnalysis(analysis, preview, datasetInfo) {
    const resultsDiv = document.getElementById('analysisResults');
    if (!resultsDiv) return;
    
    let html = `
        <div class="analysis-section">
            <h4>📊 Dataset Overview</h4>
            <p><strong>Rows:</strong> ${analysis.basic_info.shape.rows.toLocaleString()}</p>
            <p><strong>Columns:</strong> ${analysis.basic_info.shape.columns}</p>
            <p><strong>Size:</strong> ${datasetInfo.size_mb.toFixed(2)} MB</p>
            <p><strong>Data Quality Score:</strong> ${analysis.data_quality.overall}% (${analysis.data_quality.quality_grade})</p>
        </div>
        
        <div class="analysis-section">
            <h4>📈 Data Quality</h4>
            <p><strong>Completeness:</strong> ${analysis.data_quality.completeness}%</p>
            <p><strong>Missing Values:</strong> ${analysis.missing_values.total_missing.toLocaleString()}</p>
            <p><strong>Duplicate Rows:</strong> ${analysis.basic_info.duplicate_rows}</p>
        </div>
        
        <div class="analysis-section">
            <h4>🔗 Data Types</h4>
    `;
    
    const numericalCount = Object.keys(analysis.statistical_summary.numerical).length;
    const categoricalCount = Object.keys(analysis.statistical_summary.categorical).length;
    
    html += `
        <p><strong>Numerical Columns:</strong> ${numericalCount}</p>
        <p><strong>Categorical Columns:</strong> ${categoricalCount}</p>
    `;
    
    // Show first few columns with their types
    const columns = analysis.basic_info.columns.slice(0, 5);
    columns.forEach(col => {
        const colInfo = analysis.data_types.column_details[col];
        html += `<p><strong>${col}:</strong> ${colInfo.type} (${colInfo.dtype})</p>`;
    });
    
    if (analysis.basic_info.columns.length > 5) {
        html += `<p>... and ${analysis.basic_info.columns.length - 5} more columns</p>`;
    }
    
    html += `</div>`;
    
    // Add data preview
    if (preview && preview.length > 0) {
        html += `
            <div class="analysis-section">
                <h4>👀 Data Preview</h4>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem;">
                        <thead>
                            <tr>
                                ${Object.keys(preview[0]).map(key => `<th style="border: 1px solid rgba(255,255,255,0.2); padding: 8px; text-align: left;">${key}</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody>
                            ${preview.slice(0, 5).map(row => `
                                <tr>
                                    ${Object.values(row).map(value => `<td style="border: 1px solid rgba(255,255,255,0.1); padding: 8px;">${value}</td>`).join('')}
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }
    
    resultsDiv.innerHTML = html;
    
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) analyzeBtn.disabled = false;
}

function populateTargetColumn(columns) {
    const select = document.getElementById('targetColumn');
    if (!select) return;
    
    select.innerHTML = '<option value="">Select target column</option>';
    
    columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        select.appendChild(option);
    });
    
    select.disabled = false;
}

function loadSampleDataset(datasetName) {
    fetch('/load-sample-data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            dataset_name: datasetName
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentDataset = data.filename;
            displayDataAnalysis(data.analysis, data.preview, {
                size_mb: 0.1,
                processing_mode: 'sample'
            });
            populateTargetColumn(data.analysis.basic_info.columns);
            
            // Auto-select the target column for sample datasets
            const targetMap = {
                'titanic': 'Survived',
                'housing': 'price',
                'churn': 'churn'
            };
            
            const targetSelect = document.getElementById('targetColumn');
            if (targetSelect && targetMap[datasetName]) {
                targetSelect.value = targetMap[datasetName];
            }
        } else {
            throw new Error(data.error);
        }
    })
    .catch(error => {
        alert('Error loading sample dataset: ' + error.message);
    });
}

function analyzeData() {
    if (!currentDataset) return;

    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.innerHTML = '<span class="loading-spinner"></span>Analyzing...';
        analyzeBtn.disabled = true;
    }

    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: currentDataset
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayComprehensiveAnalysis(data.analysis);
        } else {
            throw new Error(data.error);
        }
    })
    .catch(error => {
        alert('Error analyzing data: ' + error.message);
    })
    .finally(() => {
        if (analyzeBtn) {
            analyzeBtn.innerHTML = 'Run Analysis';
            analyzeBtn.disabled = false;
        }
    });
}

function displayComprehensiveAnalysis(analysis) {
    const resultsDiv = document.getElementById('analysisResults');
    if (!resultsDiv) return;
    
    let html = resultsDiv.innerHTML;
    
    // Add statistical summary
    html += `
        <div class="analysis-section">
            <h4>📊 Statistical Summary</h4>
            <div class="stats-grid">
    `;
    
    const numericalCols = Object.keys(analysis.statistical_summary.numerical).slice(0, 6);
    numericalCols.forEach(col => {
        const stats = analysis.statistical_summary.numerical[col];
        html += `
            <div class="stat-card">
                <h5>${col}</h5>
                <p>Mean: ${stats.mean?.toFixed(2) || 'N/A'}</p>
                <p>Std: ${stats.std?.toFixed(2) || 'N/A'}</p>
                <p>Min: ${stats.min?.toFixed(2) || 'N/A'}</p>
                <p>Max: ${stats.max?.toFixed(2) || 'N/A'}</p>
            </div>
        `;
    });
    
    html += `</div></div>`;
    
    // Add correlation insights
    if (analysis.correlation_analysis.high_correlations.length > 0) {
        html += `
            <div class="analysis-section">
                <h4>🔗 Strong Correlations</h4>
        `;
        
        analysis.correlation_analysis.high_correlations.slice(0, 3).forEach(corr => {
            html += `<p><strong>${corr.feature1}</strong> ↔ <strong>${corr.feature2}</strong>: ${corr.correlation}</p>`;
        });
        
        html += `</div>`;
    }
    
    resultsDiv.innerHTML = html;
}

function trainModel() {
    const targetColumn = document.getElementById('targetColumn')?.value;
    const modelType = document.getElementById('modelType')?.value;
    
    if (!targetColumn) {
        alert('Please select a target column');
        return;
    }

    const trainBtn = document.getElementById('trainBtn');
    if (trainBtn) {
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<span class="loading-spinner"></span>Training...';
    }

    fetch('/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: currentDataset,
            target_column: targetColumn,
            model_type: modelType
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentModel = modelType;
            startTrainingStatusCheck(currentDataset);
            displayTrainingStart();
        } else {
            throw new Error(data.error);
        }
    })
    .catch(error => {
        alert('Error training model: ' + error.message);
        if (trainBtn) {
            trainBtn.disabled = false;
            trainBtn.innerHTML = 'Train Model';
        }
    });
}

function startTrainingStatusCheck(trainingId) {
    // Check training status every 2 seconds
    trainingInterval = setInterval(() => {
        fetch(`/training-status/${trainingId}`)
            .then(response => response.json())
            .then(data => {
                if (data.status.status === 'completed') {
                    clearInterval(trainingInterval);
                    displayTrainingResults(data.status.result);
                    const trainBtn = document.getElementById('trainBtn');
                    if (trainBtn) {
                        trainBtn.disabled = false;
                        trainBtn.innerHTML = 'Train Model';
                    }
                } else if (data.status.status === 'failed') {
                    clearInterval(trainingInterval);
                    alert('Training failed: ' + data.status.error);
                    const trainBtn = document.getElementById('trainBtn');
                    if (trainBtn) {
                        trainBtn.disabled = false;
                        trainBtn.innerHTML = 'Train Model';
                    }
                }
            })
            .catch(error => {
                console.error('Error checking training status:', error);
            });
    }, 2000);
}

function displayTrainingStart() {
    const resultsDiv = document.getElementById('trainingResults');
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = `
        <div class="training-success">
            <h4>⏳ Training Started</h4>
            <p>Model training is in progress. This may take a few minutes...</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 30%"></div>
            </div>
        </div>
    `;
}

function displayTrainingResults(results) {
    const resultsDiv = document.getElementById('trainingResults');
    if (!resultsDiv) return;
    
    let html = `
        <div class="training-success">
            <h4>✅ Model Training Complete</h4>
            <p><strong>Problem Type:</strong> ${results.problem_type}</p>
            <p><strong>Model Type:</strong> ${results.model_type}</p>
    `;
    
    if (results.problem_type === 'classification') {
        html += `<p><strong>Accuracy:</strong> ${(results.metrics.accuracy * 100).toFixed(2)}%</p>`;
        if (results.metrics.precision) {
            html += `<p><strong>Precision:</strong> ${(results.metrics.precision * 100).toFixed(2)}%</p>`;
        }
        if (results.metrics.recall) {
            html += `<p><strong>Recall:</strong> ${(results.metrics.recall * 100).toFixed(2)}%</p>`;
        }
    } else {
        html += `
            <p><strong>R² Score:</strong> ${results.metrics.r2?.toFixed(3) || 'N/A'}</p>
            <p><strong>RMSE:</strong> ${results.metrics.rmse?.toFixed(3) || 'N/A'}</p>
        `;
    }
    
    if (Object.keys(results.feature_importance).length > 0) {
        html += `<h5>🔍 Top Features:</h5>`;
        let count = 0;
        for (const [feature, importance] of Object.entries(results.feature_importance)) {
            if (count < 5) {
                html += `<p>${feature}: ${importance.toFixed(2)}%</p>`;
                count++;
            }
        }
    }
    
    html += `</div>`;
    
    resultsDiv.innerHTML = html;
    
    // Create prediction inputs based on feature importance
    createPredictionInputs(results.feature_importance);
    
    const predictBtn = document.getElementById('predictBtn');
    if (predictBtn) predictBtn.disabled = false;
}

function createPredictionInputs(featureImportance) {
    const inputsDiv = document.getElementById('predictionInputs');
    if (!inputsDiv) return;
    
    inputsDiv.innerHTML = '<h4>Enter Feature Values:</h4>';
    
    let count = 0;
    for (const feature of Object.keys(featureImportance)) {
        if (count < 8) { // Limit to top 8 features for UI
            inputsDiv.innerHTML += `
                <div class="input-group">
                    <label>${feature}:</label>
                    <input type="text" id="input_${feature}" placeholder="Enter value">
                </div>
            `;
            count++;
        }
    }
}

function makePrediction() {
    if (!currentModel) {
        alert('Please train a model first');
        return;
    }

    const inputData = {};
    const inputs = document.querySelectorAll('#predictionInputs input');
    
    for (const input of inputs) {
        const feature = input.id.replace('input_', '');
        inputData[feature] = input.value;
    }

    // Validate inputs
    let isValid = true;
    for (const input of inputs) {
        if (!input.value.trim()) {
            isValid = false;
            input.style.borderColor = 'var(--error)';
        } else {
            input.style.borderColor = '';
        }
    }

    if (!isValid) {
        alert('Please fill in all feature values');
        return;
    }

    const predictBtn = document.getElementById('predictBtn');
    if (predictBtn) {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="loading-spinner"></span>Predicting...';
    }

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            input_data: inputData,
            model_type: currentModel
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayPredictionResults(data.prediction);
        } else {
            throw new Error(data.error);
        }
    })
    .catch(error => {
        alert('Error making prediction: ' + error.message);
    })
    .finally(() => {
        if (predictBtn) {
            predictBtn.disabled = false;
            predictBtn.innerHTML = 'Predict';
        }
    });
}

function displayPredictionResults(prediction) {
    const resultsDiv = document.getElementById('predictionResults');
    if (!resultsDiv) return;
    
    let resultHTML = `
        <div class="prediction-result">
            <h4>🎯 Prediction Result</h4>
            <div class="prediction-value">${prediction.prediction[0]}</div>
    `;
    
    if (prediction.confidence) {
        const confidencePercent = (prediction.confidence[0] * 100).toFixed(1);
        resultHTML += `<div class="confidence-score">Confidence: ${confidencePercent}%</div>`;
    }
    
    resultHTML += `
            <p>Based on the trained ${currentModel} model</p>
        </div>
    `;
    
    resultsDiv.innerHTML = resultHTML;
}

function checkSystemStatus() {
    fetch('/system-status')
        .then(response => response.json())
        .then(data => {
            updateSystemStatusDisplay(data);
        })
        .catch(error => {
            console.error('Error checking system status:', error);
        });
}

function updateSystemStatusDisplay(status) {
    const currentMemory = document.getElementById('currentMemory');
    const activeTrainings = document.getElementById('activeTrainings');
    
    if (currentMemory && status.resources) {
        currentMemory.textContent = `${status.resources.memory_used_gb.toFixed(1)}GB / ${status.resources.memory_total_gb.toFixed(1)}GB`;
    }
    
    if (activeTrainings && status.active_trainings !== undefined) {
        activeTrainings.textContent = status.active_trainings;
    }
}

// Check system status every 30 seconds
setInterval(checkSystemStatus, 30000);
