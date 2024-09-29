function updateStatus() {
    fetch('/status')  // Call the endpoint to get the current status
        .then(response => response.json())
        .then(data => {
            document.getElementById('name').innerText = data.name;  // Update name
            document.getElementById('status').innerText = data.status;  // Update status
        });
}

// Update status every second
setInterval(updateStatus, 500);
