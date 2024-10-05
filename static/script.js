function updateAttendanceTable() {
    fetch('/attendance_data')  // Fetch the data from the attendance_data route
        .then(response => response.json())
        .then(data => {
            const tableBody = document.getElementById('attendace-table');
            tableBody.innerHTML = '';  // Clear the existing rows

            data.forEach(row => {
                // Create a new row
                const tr = document.createElement('tr');

                // Create and populate cells for each column
                const nameCell = document.createElement('td');
                nameCell.textContent = row.name;
                tr.appendChild(nameCell);

                const statusCell = document.createElement('td');
                statusCell.textContent = row.status;
                tr.appendChild(statusCell);

                const dateCell = document.createElement('td');
                dateCell.textContent = row.tanggal;  // Use 'tanggal' instead of 'datetime_detected'
                tr.appendChild(dateCell);

                // Add the row to the table
                tableBody.appendChild(tr);
            });
        });
}

// Update the table every 2 seconds (2000 milliseconds)
setInterval(updateAttendanceTable, 2000);
