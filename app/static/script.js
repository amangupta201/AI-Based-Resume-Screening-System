document.addEventListener("DOMContentLoaded", function () {
    const uploadTypeSelect = document.getElementById("upload_type");
    const resumeSection = document.getElementById("resume_section");
    const jobDescSection = document.getElementById("job_desc_section");
    const form = document.querySelector("form");

    // Populate inputs based on the selected mode
    function updateFileInputs(selectedType) {
        resumeSection.innerHTML = '';
        jobDescSection.innerHTML = '';

        if (selectedType === '1') {
            resumeSection.innerHTML = `<label for="resume_input">Resume (PDF):</label>
                <input type="file" id="resume_input" name="resume" accept="application/pdf" class="form-control" required>`;
            jobDescSection.innerHTML = `<label for="job_desc_input">Job Description (PDF):</label>
                <input type="file" id="job_desc_input" name="job_desc" accept="application/pdf" class="form-control" required>`;
        } else if (selectedType === '2') {
            resumeSection.innerHTML = `<label for="resume_input">Resumes (PDFs):</label>
                <input type="file" id="resume_input" name="resume" accept="application/pdf" class="form-control" multiple required>`;
            jobDescSection.innerHTML = `<label for="job_desc_input">Job Description (PDF):</label>
                <input type="file" id="job_desc_input" name="job_desc" accept="application/pdf" class="form-control" required>`;
        } else if (selectedType === '3') {
            resumeSection.innerHTML = `<label for="resume_input">Resume (PDF):</label>
                <input type="file" id="resume_input" name="resume" accept="application/pdf" class="form-control" required>`;
            jobDescSection.innerHTML = `<label for="job_desc_input">Job Descriptions (PDFs):</label>
                <input type="file" id="job_desc_input" name="job_desc" accept="application/pdf" class="form-control" multiple required>`;
        }
    }

    // Add event listener to update file inputs on selection change
    uploadTypeSelect.addEventListener("change", function () {
        updateFileInputs(this.value);
    });

    // Initialize with default mode
    updateFileInputs(uploadTypeSelect.value);

    // Form submission validation
    form.addEventListener("submit", function (event) {
        const resumeInput = document.querySelector('input[name="resume"]');
        const jobDescInput = document.querySelector('input[name="job_desc"]');

        if (!resumeInput || !jobDescInput || resumeInput.files.length === 0 || jobDescInput.files.length === 0) {
            alert("Please upload the required resume and job description file(s) based on the selected mode.");
            event.preventDefault();
            return;
        }

        console.log("Form is valid for submission.");
    });
});
