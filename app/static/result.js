// Function to toggle the visibility of suggestions
function toggleSuggestions(index) {
    const suggestionsList = document.getElementById('suggestions-' + index);

    // Toggle the display of suggestions
    if (suggestionsList.style.display === "none") {
        suggestionsList.style.display = "block";
    } else {
        suggestionsList.style.display = "none";
    }

    // Optionally toggle the aria-expanded attribute for accessibility
    const button = event.target;
    const isExpanded = button.getAttribute("aria-expanded") === "true";
    button.setAttribute("aria-expanded", !isExpanded);
}

// Function to download the ranking results as a PDF
function downloadResults() {
    const { jsPDF } = window.jspdf;  // Import jsPDF
    const doc = new jsPDF();         // Create a new PDF document

    let yPosition = 10;  // Starting y position for the text in PDF

    // Add title to the PDF
    doc.setFontSize(18);
    doc.text('Resume Ranking Results', 10, yPosition);
    yPosition += 15;

    // Get all rows in the table (excluding the header)
    const rows = document.querySelectorAll('table tbody tr');

    rows.forEach(row => {
        const filename = row.querySelector('td:first-child').textContent.trim();
        const score = row.querySelector('td:nth-child(2)').textContent.trim();
        const jobPosition = row.querySelector('td:nth-child(3)').textContent.trim();

        // Add the resume data to the PDF
        doc.setFontSize(12);
        doc.text(`Resume: ${filename}`, 10, yPosition);
        yPosition += 10;
        doc.text(`Job Position: ${jobPosition}`, 10, yPosition);
        yPosition += 10;
        doc.text(`Ranking Score: ${score}`, 10, yPosition);
        yPosition += 15;  // Space between each entry

        // If there are suggestions, add them to the PDF as well
        const suggestionsList = row.querySelector('ul');
        if (suggestionsList) {
            doc.text('Suggestions:', 10, yPosition);
            yPosition += 10;
            suggestionsList.querySelectorAll('li').forEach(suggestion => {
                const suggestionText = suggestion.textContent.trim();
                doc.text(`- ${suggestionText}`, 10, yPosition);
                yPosition += 10;
            });
            yPosition += 10;  // Space after suggestions
        }

        // Add a page break if content exceeds the page limit
        if (yPosition > 250) {
            doc.addPage();
            yPosition = 10;  // Reset position for the new page
        }
    });

    // Save the PDF with a filename
    doc.save('ranking_result.pdf');

    // Optionally, you can add a success message
    alert("Your results are ready to be downloaded as a PDF!");
}
