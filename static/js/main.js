// --- SocketIO Setup ---
const socket = io();

// --- DOM Elements ---
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const sessionTime = document.getElementById('session-time');
const fpsCounter = document.getElementById('fps-counter');
const integrityScore = document.getElementById('integrity-score');
const facesDetected = document.getElementById('faces-detected');
const brightnessLevel = document.getElementById('brightness-level');
const totalAlerts = document.getElementById('total-alerts');
const sessionDuration = document.getElementById('session-duration');
const alertsContainer = document.getElementById('alerts-container');
const noAlerts = document.getElementById('no-alerts');
const clearAlertsBtn = document.getElementById('clear-alerts');

let statsInterval = null;
let alertsInterval = null;
let sessionTimer = null;
let sessionStart = null;



// (Loading overlay removed as requested)

// --- Button Actions ---
startBtn.addEventListener('click', () => {
	fetch('/start_proctoring', { method: 'POST' })
		.then(res => res.json())
		.then(data => {
			startBtn.style.display = 'none';
			stopBtn.style.display = '';
			statusDot.style.background = '#4caf50';
			statusText.textContent = 'Proctoring';
			sessionStart = Date.now();
			startStatsPolling();
			startAlertsPolling();
			startSessionTimer();
			// Immediately update stats after session starts
			updateStats();
		});
});

stopBtn.addEventListener('click', () => {
	fetch('/stop_proctoring', { method: 'POST' })
		.then(res => res.json())
		.then(data => {
			stopBtn.style.display = 'none';
			startBtn.style.display = '';
			statusDot.style.background = '#f44336';
			statusText.textContent = 'Stopped';
			stopStatsPolling();
			stopAlertsPolling();
			stopSessionTimer();
			showSessionReport(data.report);
		});
});

clearAlertsBtn.addEventListener('click', () => {
	alertsContainer.innerHTML = '';
	noAlerts.style.display = '';
});

// --- Polling Functions ---
function startStatsPolling() {
	updateStats();
	statsInterval = setInterval(updateStats, 1000);
}
function stopStatsPolling() {
	clearInterval(statsInterval);
}
function updateStats() {
	fetch('/get_stats')
		.then(res => res.json())
		.then(stats => {
			facesDetected.textContent = stats.face_detections;
			brightnessLevel.textContent = Math.round(stats.brightness_avg);
			totalAlerts.textContent = stats.total_alerts;
			sessionDuration.textContent = Math.floor(stats.session_duration / 60) + 'm';
			fpsCounter.textContent = stats.fps + ' FPS';
			integrityScore.textContent = stats.integrity_score + '%';
		});
}

function startAlertsPolling() {
	updateAlerts();
	alertsInterval = setInterval(updateAlerts, 2000);
}
function stopAlertsPolling() {
	clearInterval(alertsInterval);
}
function updateAlerts() {
	fetch('/get_alerts')
		.then(res => res.json())
		.then(data => {
			if (data.alerts.length === 0) {
				noAlerts.style.display = '';
				alertsContainer.innerHTML = '';
			} else {
				noAlerts.style.display = 'none';
				alertsContainer.innerHTML = '';
				data.alerts.forEach(alert => {
					const div = document.createElement('div');
					div.className = 'alert-item ' + alert.type;
					div.innerHTML = `<span class="alert-time">[${alert.timestamp}]</span> <span class="alert-msg">${alert.message}</span>`;
					alertsContainer.appendChild(div);
				});
			}
		});
}

// --- Session Timer ---
function startSessionTimer() {
	stopSessionTimer();
	sessionTimer = setInterval(() => {
		if (sessionStart) {
			const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
			const min = String(Math.floor(elapsed / 60)).padStart(2, '0');
			const sec = String(elapsed % 60).padStart(2, '0');
			sessionTime.textContent = `${min}:${sec}`;
		}
	}, 1000);
}
function stopSessionTimer() {
	clearInterval(sessionTimer);
	sessionTime.textContent = '00:00';
}

// --- SocketIO Events ---
socket.on('new_alert', function(alert) {
	updateAlerts();
});
socket.on('proctoring_started', function(data) {
	statusDot.style.background = '#4caf50';
	statusText.textContent = 'Proctoring';
});
socket.on('proctoring_stopped', function(report) {
	statusDot.style.background = '#f44336';
	statusText.textContent = 'Stopped';
	showSessionReport(report);
});

// --- Session Report Modal ---
function showSessionReport(report) {
	const modal = document.getElementById('report-modal');
	const body = document.getElementById('modal-body');
	if (!report) return;
	body.innerHTML = `
		<p><b>Session Duration:</b> ${Math.floor(report.session_duration/60)}m ${Math.floor(report.session_duration%60)}s</p>
		<p><b>Total Alerts:</b> ${report.total_alerts}</p>
		<p><b>Integrity Score:</b> ${report.integrity_score}%</p>
		<h4>Recent Alerts:</h4>
		<ul>${report.alerts.map(a => `<li>[${a.timestamp}] ${a.message}</li>`).join('')}</ul>
	`;
	modal.style.display = 'block';
	document.querySelector('.modal .close').onclick = () => { modal.style.display = 'none'; };
	window.onclick = (e) => { if (e.target == modal) modal.style.display = 'none'; };
}

// --- Download Report ---
function downloadReport() {
	const body = document.getElementById('modal-body');
	const text = body.innerText;
	const blob = new Blob([text], {type: 'text/plain'});
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = 'session_report.txt';
	a.click();
	URL.revokeObjectURL(url);
}