// Dashboard functionality
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

function initializeDashboard() {
    // Page Navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Update active nav link
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Show corresponding page
            const pageId = this.getAttribute('data-page');
            document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
            document.getElementById(pageId).classList.add('active');
            
            // Update page title
            updatePageTitle(pageId);
        });
    });

    // Initialize chart
    generateChart();
    
    // Add animation to cards on load
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });
}

function updatePageTitle(pageId) {
    const pageTitle = document.getElementById('page-title');
    const titles = {
        'dashboard': 'Welcome, Operator',
        'ml-prediction': 'ML Prediction System',
        'ml-results': 'ML Prediction Results',
        'ai-control': 'AI Control Panel',
        'models': 'AI Models Management',
        'data-sets': 'Data Sets Management',
        'analytics': 'System Analytics',
        'settings': 'System Settings',
        'support': 'Support & Help',
        'documentation': 'Documentation'
    };
    pageTitle.textContent = titles[pageId] || 'AI Frost Console';
}

// Dark mode toggle
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const button = document.querySelector('button[onclick="toggleDarkMode()"]');
    if (document.body.classList.contains('dark-mode')) {
        button.innerHTML = '☀️ Light Mode';
        localStorage.setItem('darkMode', 'enabled');
    } else {
        button.innerHTML = '🌙 Dark Mode';
        localStorage.setItem('darkMode', 'disabled');
    }
}

// Check for saved dark mode preference
if (localStorage.getItem('darkMode') === 'enabled') {
    document.body.classList.add('dark-mode');
    const button = document.querySelector('button[onclick="toggleDarkMode()"]');
    if (button) button.innerHTML = '☀️ Light Mode';
}

// AI Message function
function sendAIMessage() {
    const input = document.getElementById("aiInput");
    const log = document.getElementById("logConsole");
    if (input.value.trim()) {
        const time = new Date().toLocaleTimeString();
        log.innerHTML += `
            <div class="log-entry">
                <div class="log-time">${time}</div>
                <div class="log-message info">User: ${input.value}</div>
            </div>
        `;
        
        // Simulate AI response after a short delay
        setTimeout(() => {
            const responses = [
                "I've analyzed your query and found the relevant data in the performance metrics.",
                "Based on current system status, I recommend checking the memory usage alerts.",
                "The model training completed successfully at 14:23. Would you like more details?",
                "I've detected an issue with the backup system. Maintenance is scheduled for 16:00."
            ];
            const response = responses[Math.floor(Math.random() * responses.length)];
            const responseTime = new Date().toLocaleTimeString();
            log.innerHTML += `
                <div class="log-entry">
                    <div class="log-time">${responseTime}</div>
                    <div class="log-message success">AI: ${response}</div>
                </div>
            `;
            log.scrollTop = log.scrollHeight;
        }, 1000);
        
        log.scrollTop = log.scrollHeight;
        input.value = "";
    }
}

// Allow sending message with Enter key
document.addEventListener('DOMContentLoaded', function() {
    const aiInput = document.getElementById("aiInput");
    if (aiInput) {
        aiInput.addEventListener("keypress", function(event) {
            if (event.key === "Enter") {
                sendAIMessage();
            }
        });
    }
});

// System notifications
function showNotifications() {
    alert("You have 3 notifications:\n- Model training completed\n- High memory usage\n- Backup system offline");
}

// Refresh data function
function refreshData() {
    const cards = document.querySelectorAll('.card .value');
    cards.forEach(card => {
        card.style.transform = 'scale(1.1)';
        setTimeout(() => {
            card.style.transform = 'scale(1)';
        }, 300);
    });
    
    // Add a refresh entry to the log
    const log = document.getElementById("logConsole");
    if (log) {
        const time = new Date().toLocaleTimeString();
        log.innerHTML += `
            <div class="log-entry">
                <div class="log-time">${time}</div>
                <div class="log-message info">Data refresh requested</div>
            </div>
        `;
        log.scrollTop = log.scrollHeight;
    }
}

// Chart functionality
function generateChart() {
    const chart = document.getElementById('performanceChart');
    const grid = document.getElementById('chartGrid');
    
    if (!chart || !grid) return;
    
    // Clear existing content
    chart.innerHTML = '';
    grid.innerHTML = '';
    
    // Generate grid
    for (let i = 0; i < 100; i++) {
        const gridLine = document.createElement('div');
        gridLine.className = 'chart-grid-line';
        grid.appendChild(gridLine);
    }
    
    // Generate bars with random heights
    const barCount = 12;
    for (let i = 0; i < barCount; i++) {
        const bar = document.createElement('div');
        const height = Math.floor(Math.random() * 80) + 20;
        bar.className = 'chart-bar';
        bar.style.height = `${height}%`;
        bar.setAttribute('data-value', `${height}%`);
        chart.appendChild(bar);
    }
}

// Change chart type
function changeChartType(type) {
    const buttons = document.querySelectorAll('.chart-actions button');
    buttons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    // In a real app, this would fetch new data based on the selected timeframe
    generateChart();
}

// Simulate system events
setInterval(() => {
    const log = document.getElementById("logConsole");
    if (!log) return;
    
    const events = [
        {type: "info", message: "AI Model Sync Complete"},
        {type: "warning", message: "High memory usage detected in sector 4"},
        {type: "info", message: "Data pipeline refreshed successfully"},
        {type: "error", message: "Stream timeout, retrying connection..."},
        {type: "success", message: "All systems operating within normal parameters"},
        {type: "info", message: "New data batch processed (1.2GB)"},
        {type: "warning", message: "CPU temperature approaching threshold"}
    ];
    const event = events[Math.floor(Math.random() * events.length)];
    const time = new Date().toLocaleTimeString();
    
    log.innerHTML += `
        <div class="log-entry">
            <div class="log-time">${time}</div>
            <div class="log-message ${event.type}">${event.message}</div>
        </div>
    `;
    log.scrollTop = log.scrollHeight;
}, 15000);


