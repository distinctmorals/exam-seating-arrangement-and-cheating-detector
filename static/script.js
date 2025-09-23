const videoGrid = document.querySelector('.video-grid');
const status = document.getElementById('status');
const statusText = document.getElementById('statusText');
const statusIcon = document.getElementById('statusIcon');
const incidentList = document.getElementById('incident-list');
const toggleIncidents = document.getElementById('toggleIncidents');
const cameraSelect = document.getElementById('cameraSelect');
const seatingForm = document.getElementById('seatingForm');
const seatingResult = document.getElementById('seatingResult');
const sessionIdInput = document.getElementById('sessionId');
const logoutLink = document.getElementById('logoutLink');
const proctoringSection = document.getElementById('proctoringSection');

const examId = 'exam_' + Math.random().toString(36).substr(2, 9);
const streams = [
    { userId: 'user_1', streamId: 'stream_1' }
];

// Retrieve session_id from URL or localStorage
const urlParams = new URLSearchParams(window.location.search);
let sessionId = urlParams.get('session_id') || localStorage.getItem('session_id');
console.log('Client: Retrieved session_id:', sessionId);

// Store session_id in localStorage if present
if (sessionId) {
    localStorage.setItem('session_id', sessionId);
    sessionIdInput.value = sessionId;
    logoutLink.href = `/logout?session_id=${encodeURIComponent(sessionId)}`;
} else {
    console.warn('Client: No session_id found in URL or localStorage');
    seatingResult.innerHTML = `<p class="text-red-600">Error: No session ID. Please log in again.</p>`;
    setTimeout(() => { window.location.href = '/'; }, 2000);
}

async function getVideoDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(device => device.kind === 'videoinput');
}

async function populateCameraSelect() {
    const devices = await getVideoDevices();
    cameraSelect.innerHTML = '<option value="">Select a camera</option>';
    devices.forEach(device => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text = device.label || `Camera ${cameraSelect.options.length}`;
        cameraSelect.appendChild(option);
    });
    return devices;
}

async function selectUSBWebcam(devices) {
    const usbWebcam = devices.find(device => 
        !device.label.toLowerCase().includes('droidcam') && 
        (device.label.toLowerCase().includes('webcam') || device.label.toLowerCase().includes('usb'))
    ) || devices[0];
    return usbWebcam ? usbWebcam.deviceId : null;
}

async function setupCamera(streamId, deviceId = null) {
    const container = document.getElementById(`container-${streamId}`) || document.createElement('div');
    container.className = 'video-container';
    container.id = `container-${streamId}`;
    container.style.position = 'relative';
    
    let video = document.getElementById(`video-${streamId}`);
    if (!video) {
        video = document.createElement('video');
        video.id = `video-${streamId}`;
        video.autoplay = true;
    }
    
    let canvas = document.getElementById(`canvas-${streamId}`);
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = `canvas-${streamId}`;
        canvas.style.display = 'none';
    }
    
    let streamStatus = document.getElementById(`status-${streamId}`);
    if (!streamStatus) {
        streamStatus = document.createElement('div');
        streamStatus.id = `status-${streamId}`;
        streamStatus.className = 'stream-status';
        streamStatus.style.position = 'absolute';
        streamStatus.style.top = '10px';
        streamStatus.style.left = '10px';
        streamStatus.style.padding = '5px 10px';
        streamStatus.style.borderRadius = '5px';
        streamStatus.style.color = 'white';
        streamStatus.style.fontWeight = 'bold';
        streamStatus.style.fontSize = '14px';
        streamStatus.style.zIndex = '10';
    }
    
    container.innerHTML = '';
    container.appendChild(video);
    container.appendChild(canvas);
    container.appendChild(streamStatus);
    videoGrid.innerHTML = '';
    videoGrid.appendChild(container);
    
    if (!deviceId) {
        const devices = await getVideoDevices();
        deviceId = await selectUSBWebcam(devices);
        cameraSelect.value = deviceId || '';
    }
    
    if (!deviceId) {
        streamStatus.textContent = `No USB webcam found`;
        streamStatus.style.backgroundColor = '#dc2626';
        return { video, canvas };
    }
    
    try {
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { deviceId: { exact: deviceId }, width: 192, height: 192 },
            audio: false
        });
        video.srcObject = stream;
        video.play();
        
        canvas.width = 192;
        canvas.height = 192;
        
        const devices = await getVideoDevices();
        const selectedDevice = devices.find(device => device.deviceId === deviceId);
        console.log(`Stream ${streamId}: Selected device ID ${deviceId}, Label: ${selectedDevice ? selectedDevice.label : 'unknown'}`);
        streamStatus.textContent = 'Normal';
        streamStatus.style.backgroundColor = '#16a34a';
        
        return { video, canvas };
    } catch (error) {
        console.error(`Error setting up camera for stream ${streamId}:`, error);
        streamStatus.textContent = `Failed to access camera`;
        streamStatus.style.backgroundColor = '#dc2626';
        return { video, canvas };
    }
}

async function detectFaces() {
    const streamData = [];
    
    for (const stream of streams) {
        const video = document.getElementById(`video-${stream.streamId}`);
        const canvas = document.getElementById(`canvas-${stream.streamId}`);
        const ctx = canvas.getContext('2d');
        
        if (!video.srcObject) {
            statusText.textContent = 'No active video stream';
            status.className = 'status suspicious flex items-center justify-center p-4 rounded-md mb-4';
            statusIcon.innerHTML = '❌';
            return;
        }
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg', 0.7);
        
        if (!imageData.startsWith('data:image/jpeg;base64,')) {
            console.error(`Invalid image data for stream ${stream.streamId}`);
            statusText.textContent = `Invalid image data`;
            status.className = 'status suspicious flex items-center justify-center p-4 rounded-md mb-4';
            statusIcon.innerHTML = '❌';
            return;
        }
        
        streamData.push({
            image: imageData,
            user_id: stream.userId,
            exam_id: examId,
            stream_id: stream.streamId
        });
    }
    
    try {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ streams: streamData })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}: ${await response.text()}`);
        }
        
        const results = await response.json();
        let hasSuspicious = false;
        
        results.forEach(result => {
            const streamStatus = document.getElementById(`status-${result.stream_id}`);
            console.log(`Stream ${result.stream_id}: Status: ${result.status}, Primary: ${result.primary_incident}`);
            if (result.status === 'suspicious') {
                hasSuspicious = true;
                streamStatus.textContent = result.primary_incident;
                streamStatus.style.backgroundColor = '#dc2626';
            } else {
                streamStatus.textContent = 'Normal';
                streamStatus.style.backgroundColor = '#16a34a';
            }
            const li = document.createElement('li');
            li.textContent = `Stream ${result.stream_id}: ${result.primary_incident} at ${new Date().toLocaleString('en-US', { timeZone: 'Africa/Lagos' })}`;
            incidentList.prepend(li);
        });
        
        statusText.textContent = `Overall Status: ${hasSuspicious ? 'Suspicious' : 'Normal'}`;
        status.className = `status ${hasSuspicious ? 'suspicious' : 'normal'} flex items-center justify-center p-4 rounded-md mb-4`;
        statusIcon.innerHTML = hasSuspicious ? '⚠️' : '✅';
    } catch (error) {
        console.error('Error in detectFaces:', error);
        statusText.textContent = `Error: ${error.message}`;
        status.className = 'status suspicious flex items-center justify-center p-4 rounded-md mb-4';
        statusIcon.innerHTML = '❌';
    }
}

async function loadIncidents() {
    try {
        const response = await fetch(`/incidents/${examId}`);
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        const incidents = await response.json();
        incidentList.innerHTML = '';
        incidents.forEach(incident => {
            const li = document.createElement('li');
            li.textContent = `Stream ${incident.stream_id}: ${incident.incident_type} at ${new Date(incident.timestamp).toLocaleString('en-US', { timeZone: 'Africa/Lagos' })}`;
            incidentList.appendChild(li);
        });
    } catch (error) {
        console.error('Error loading incidents:', error);
    }
}

cameraSelect.addEventListener('change', async () => {
    const deviceId = cameraSelect.value;
    if (deviceId) {
        await setupCamera(streams[0].streamId, deviceId);
        detectFaces();
    }
});

toggleIncidents.addEventListener('click', () => {
    incidentList.classList.toggle('hidden');
    toggleIncidents.textContent = incidentList.classList.contains('hidden') ? 'Show' : 'Hide';
});

seatingForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!sessionId) {
        console.error('Client: No session_id available. Redirecting to login.');
        seatingResult.innerHTML = `<p class="text-red-600">Error: No session ID. Please log in again.</p>`;
        window.location.href = '/';
        return;
    }
    console.log('Client: Submitting seating form with session_id:', sessionId);
    const formData = new FormData(seatingForm);
    try {
        const response = await fetch(`/generate_seating?session_id=${encodeURIComponent(sessionId)}`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`HTTP error ${response.status}: ${errorData.detail || 'Unknown error'}`);
        }
        const data = await response.json();
        console.log('Client: Seating arrangement received:', data);
        displaySeating(data.seating);
        // Start proctoring after seating is generated
        proctoringSection.style.display = 'block';
        init();
    } catch (error) {
        console.error('Client: Error generating seating:', error);
        seatingResult.innerHTML = `<p class="text-red-600">Error: ${error.message}</p>`;
    }
});

const courseColors = {};

function getColor(course) {
    if (!courseColors[course]) {
        let hash = 0;
        for (let i = 0; i < course.length; i++) {
            hash = course.charCodeAt(i) + ((hash << 5) - hash);
        }
        const r = (hash & 0xFF);
        const g = ((hash >> 8) & 0xFF);
        const b = ((hash >> 16) & 0xFF);
        courseColors[course] = `rgb(${r}, ${g}, ${b})`;
    }
    return courseColors[course];
}

function displaySeating(seating) {
    let html = '<table class="w-full border-collapse border border-gray-300">';
    seating.forEach(row => {
        html += '<tr>';
        row.forEach(cell => {
            const bgColor = cell ? getColor(cell.course) : 'white';
            html += `<td class="border border-gray-300 p-2" style="background-color: ${bgColor};">`;
            if (cell) {
                html += `${cell.name} (${cell.course})`;
            } else {
                html += '&nbsp;';
            }
            html += '</td>';
        });
        html += '</tr>';
    });
    html += '</table>';
    seatingResult.innerHTML = html;
}

async function init() {
    if (!sessionId) {
        console.error('Client: No session_id found. Redirecting to login.');
        window.location.href = '/';
        return;
    }
    console.log('Client: Initializing with session_id:', sessionId);
    const devices = await getVideoDevices();
    console.log('Client: Available video devices:', devices);
    await populateCameraSelect();
    
    for (const stream of streams) {
        await setupCamera(stream.streamId);
    }
    setInterval(detectFaces, 1000);
    setInterval(loadIncidents, 5000);
}

// Do not call init() here; it will be called after seating generation