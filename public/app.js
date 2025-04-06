// API URL - Change this if your server is on a different host/port
const API_BASE_URL = '/';  // Using relative path for same-origin requests

// DOM Elements
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');
const loadingOverlay = document.getElementById('loading-overlay');

// Tab elements
const askButton = document.getElementById('ask-button');
const questionInput = document.getElementById('question');
const includeSourcesCheckbox = document.getElementById('include-sources');
const askResults = document.getElementById('ask-results');

const searchButton = document.getElementById('search-button');
const searchQuery = document.getElementById('search-query');
const resultLimit = document.getElementById('result-limit');
const detailedResultsCheckbox = document.getElementById('detailed-results');
const searchResults = document.getElementById('search-results');

const refreshStatusButton = document.getElementById('refresh-status');
const detailedStatusCheckbox = document.getElementById('detailed-status');
const statusResults = document.getElementById('status-results');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Set up tab switching
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.getAttribute('data-tab');
            switchTab(tabName);
        });
    });

    // Set up event listeners
    askButton.addEventListener('click', handleAskQuestion);
    searchButton.addEventListener('click', handleSearch);
    refreshStatusButton.addEventListener('click', handleRefreshStatus);

    // Enter key in search field
    searchQuery.addEventListener('keyup', (e) => {
        if (e.key === 'Enter') handleSearch();
    });

    // Enter key in question field (Ctrl+Enter)
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) handleAskQuestion();
    });

    // Initial status check
    handleRefreshStatus();
});

// Tab switching
function switchTab(tabName) {
    // Update active tab
    tabs.forEach(tab => {
        if (tab.getAttribute('data-tab') === tabName) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    // Update active content
    tabContents.forEach(content => {
        if (content.id === `${tabName}-tab`) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });
}

// Loading state
function setLoading(isLoading) {
    if (isLoading) {
        loadingOverlay.classList.add('visible');
    } else {
        loadingOverlay.classList.remove('visible');
    }
}

// API Handlers
async function handleAskQuestion() {
    const question = questionInput.value.trim();
    if (!question) {
        alert('Please enter a question');
        return;
    }

    setLoading(true);
    try {
        const response = await fetch(`${API_BASE_URL}api/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                include_sources: includeSourcesCheckbox.checked
            })
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        displayAnswer(data);
    } catch (error) {
        console.error('Error asking question:', error);
        askResults.innerHTML = `
            <div class="error">
                <h3>Error</h3>
                <p>${error.message || 'Failed to get answer'}</p>
            </div>
        `;
    } finally {
        setLoading(false);
    }
}

async function handleSearch() {
    const query = searchQuery.value.trim();
    if (!query) {
        alert('Please enter a search query');
        return;
    }

    setLoading(true);
    try {
        const response = await fetch(`${API_BASE_URL}api/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                limit: parseInt(resultLimit.value),
                detailed: detailedResultsCheckbox.checked
            })
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        displaySearchResults(data);
    } catch (error) {
        console.error('Error searching:', error);
        searchResults.innerHTML = `
            <div class="error">
                <h3>Error</h3>
                <p>${error.message || 'Failed to search code'}</p>
            </div>
        `;
    } finally {
        setLoading(false);
    }
}

async function handleRefreshStatus() {
    setLoading(true);
    try {
        const detailed = detailedStatusCheckbox.checked ? '?detailed=true' : '';
        const response = await fetch(`${API_BASE_URL}api/system/status${detailed}`);

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        displaySystemStatus(data);
    } catch (error) {
        console.error('Error getting status:', error);
        statusResults.innerHTML = `
            <div class="error">
                <h3>Error</h3>
                <p>${error.message || 'Failed to get system status'}</p>
            </div>
        `;
    } finally {
        setLoading(false);
    }
}

// UI Rendering Functions
function displayAnswer(data) {
    let html = `<div class="answer-content">${formatText(data.answer)}</div>`;

    if (data.sources && data.sources.length > 0) {
        html += `
            <div class="sources-list">
                <h3 class="sources-header">Sources (${data.sources.length})</h3>
        `;

        data.sources.forEach(source => {
            html += `
                <div class="source-item">
                    <div class="source-path">${escapeHtml(source.file_path)}</div>
                    <div class="source-repo">Repository: ${escapeHtml(source.repo_name)}</div>
                    <pre class="source-content">${escapeHtml(source.content)}</pre>
                </div>
            `;
        });

        html += `</div>`;
    }

    askResults.innerHTML = html;
}

function displaySearchResults(data) {
    if (!data.results || data.results.length === 0) {
        searchResults.innerHTML = `<div class="placeholder">No results found for "${escapeHtml(data.query)}"</div>`;
        return;
    }

    let html = `
        <div class="search-header">
            <h3>Results for "${escapeHtml(data.query)}" (${data.count})</h3>
            ${data.query_time ? `<div class="search-time">Query time: ${data.query_time}</div>` : ''}
        </div>
    `;

    data.results.forEach(result => {
        const relevanceScore = result.relevance ? (result.relevance * 100).toFixed(0) + '%' : 'N/A';
        
        html += `
            <div class="search-item">
                <div class="search-item-header">
                    <div class="search-path">${escapeHtml(result.file_path)}</div>
                    <div class="search-score">Relevance: ${relevanceScore}</div>
                </div>
                <div class="search-repo">Repository: ${escapeHtml(result.repo_name)}</div>
                <pre class="search-snippet">${escapeHtml(result.snippet)}</pre>
            </div>
        `;
    });

    searchResults.innerHTML = html;
}

function displaySystemStatus(data) {
    let html = '';

    // API Status
    html += `
        <div class="status-card">
            <h3>API Status</h3>
            <div class="status-section">
                <div class="status-item">
                    <div class="status-label">Version:</div>
                    <div class="status-value">${data.api?.version || 'Unknown'}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Initialized:</div>
                    <div class="status-value ${data.api?.initialized ? 'good' : 'bad'}">
                        ${data.api?.initialized ? 'Yes' : 'No'}
                    </div>
                </div>
                ${data.api?.uptime ? `
                <div class="status-item">
                    <div class="status-label">Uptime:</div>
                    <div class="status-value">${formatUptime(data.api.uptime)}</div>
                </div>` : ''}
            </div>
        </div>
    `;

    // Database Status
    html += `
        <div class="status-card">
            <h3>Database</h3>
            <div class="status-section">
                <div class="status-item">
                    <div class="status-label">Status:</div>
                    <div class="status-value ${data.database?.exists ? 'good' : 'bad'}">
                        ${data.database?.exists ? 'Available' : 'Not Found'}
                    </div>
                </div>
                <div class="status-item">
                    <div class="status-label">Size:</div>
                    <div class="status-value">${data.database?.size || 'Unknown'}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Collection:</div>
                    <div class="status-value">${data.database?.collection?.name || 'Not available'}</div>
                </div>
                ${data.database?.collection?.doc_count ? `
                <div class="status-item">
                    <div class="status-label">Documents:</div>
                    <div class="status-value">${data.database.collection.doc_count}</div>
                </div>` : ''}
            </div>
        </div>
    `;

    // Repositories Status
    html += `
        <div class="status-card">
            <h3>Repositories</h3>
            <div class="status-section">
                <div class="status-item">
                    <div class="status-label">Total Repos:</div>
                    <div class="status-value">${data.repositories?.total || 0}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Active Repos:</div>
                    <div class="status-value">${data.repositories?.active || 0}</div>
                </div>
                ${data.repositories?.names ? `
                <div class="status-item">
                    <div class="status-label">Active Names:</div>
                    <div class="status-value">${data.repositories.names.join(', ')}</div>
                </div>` : ''}
            </div>
            
            ${renderDetailedRepos(data.repositories?.details)}
        </div>
    `;

    // Models Status
    html += `
        <div class="status-card">
            <h3>Models</h3>
            <div class="status-section">
                <div class="status-item">
                    <div class="status-label">Status:</div>
                    <div class="status-value ${data.models?.exists ? 'good' : 'bad'}">
                        ${data.models?.exists ? 'Available' : 'Not Found'}
                    </div>
                </div>
                <div class="status-item">
                    <div class="status-label">Size:</div>
                    <div class="status-value">${data.models?.size || 'Unknown'}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">LLM:</div>
                    <div class="status-value">${data.models?.llm || 'Not specified'}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Embedding:</div>
                    <div class="status-value">${data.models?.embedding || 'Not specified'}</div>
                </div>
            </div>
        </div>
    `;

    statusResults.innerHTML = html;
}

// Helper Functions
function renderDetailedRepos(details) {
    if (!details || details.length === 0) return '';
    
    let html = '<div class="repo-details">';
    
    details.forEach(repo => {
        html += `
            <div class="repo-detail-item">
                <h4>${escapeHtml(repo.name)}</h4>
                <div class="status-item">
                    <div class="status-label">Key:</div>
                    <div class="status-value">${escapeHtml(repo.key)}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Status:</div>
                    <div class="status-value ${repo.exists ? 'good' : 'bad'}">
                        ${repo.exists ? 'Available' : 'Not Found'}
                    </div>
                </div>
                ${repo.stats ? `
                <div class="status-item">
                    <div class="status-label">Files:</div>
                    <div class="status-value">
                        Total: ${repo.stats.total_files || 0} |
                        TS: ${repo.stats.typescript_files || 0} |
                        JS: ${repo.stats.javascript_files || 0} |
                        PY: ${repo.stats.python_files || 0}
                    </div>
                </div>` : ''}
            </div>
        `;
    });
    
    html += '</div>';
    return html;
}

function formatUptime(seconds) {
    if (!seconds) return 'Unknown';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    return `${hours}h ${minutes}m ${secs}s`;
}

function formatText(text) {
    if (!text) return '';
    
    // Replace newlines with <br>
    let formatted = text.replace(/\n/g, '<br>');
    
    // Wrap code blocks
    formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre class="code-block">$1</pre>');
    
    // Handle inline code
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    return formatted;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}