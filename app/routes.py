import re

import spacy

# Load spaCy model for English
nlp = spacy.load("en_core_web_sm")
from flask import Blueprint, render_template, request
import os
from werkzeug.utils import secure_filename
from rank import rank_resume
from utils import extract_text_from_pdf
import io
import json
from flask import Response

# Create a Blueprint for routing
main = Blueprint('main', __name__)


@main.route('/')
def home():
    return render_template('index.html')


def extract_job_position(job_desc_text):
    title_pattern = r"(?i)(?:position|title):\s*(.+)"
    title_match = re.search(title_pattern, job_desc_text)

    if title_match:
        return title_match.group(1).strip()

    lines = job_desc_text.splitlines()
    for line in lines[:5]:
        if any(keyword in line.lower() for keyword in ["engineer", "developer", "manager", "scientist", "analyst"]):
            return line.strip()

    return "Unknown Position"


@main.route('/upload', methods=['POST'])
def upload():
    upload_type = request.form['upload_type']  # Get the upload type from the form

    # Initialize the results list to store all comparisons
    results = []

    # Check which type of upload is selected and process accordingly
    if upload_type == '1':  # One Resume, One Job Description
        resume = request.files['resume']
        job_desc_file = request.files['job_desc']

        if resume and job_desc_file:
            # Save the job description file
            job_desc_filename = secure_filename(job_desc_file.filename)
            job_desc_filepath = os.path.join('uploads', job_desc_filename)
            job_desc_file.save(job_desc_filepath)

            # Extract text from the job description
            job_desc_text = extract_text_from_pdf(job_desc_filepath)

            # Extract job position dynamically
            job_position = extract_job_position(job_desc_text)

            # Save and process the resume
            if resume.filename.endswith('.pdf'):
                resume_filename = secure_filename(resume.filename)
                resume_filepath = os.path.join('uploads', resume_filename)
                resume.save(resume_filepath)

                # Extract text from resume
                resume_text = extract_text_from_pdf(resume_filepath)

                # Inside each resume processing block
                candidate_type = categorize_candidate(resume_text)

                # Rank the resume based on the job description
                score = rank_resume(resume_text, job_desc_text)

                # Adjust feedback and ranking based on candidate type
                feedback = get_improvement_suggestions(resume_text, job_desc_text, candidate_type)
                results.append({
                    'filename': resume_filename,
                    'score': score,
                    'job_position': job_position,
                    'candidate_type': candidate_type,
                    'feedback': feedback
                })

    elif upload_type == '2':  # Multiple Resumes, One Job Description
        resumes = request.files.getlist('resume')
        job_desc_file = request.files['job_desc']

        if resumes and job_desc_file:
            # Save the job description file
            job_desc_filename = secure_filename(job_desc_file.filename)
            job_desc_filepath = os.path.join('uploads', job_desc_filename)
            job_desc_file.save(job_desc_filepath)

            # Extract text from the job description
            job_desc_text = extract_text_from_pdf(job_desc_filepath)

            # Extract job position dynamically
            job_position = extract_job_position(job_desc_text)

            # Process each resume
            for resume_file in resumes:
                if resume_file.filename.endswith('.pdf'):
                    resume_filename = secure_filename(resume_file.filename)
                    resume_filepath = os.path.join('uploads', resume_filename)
                    resume_file.save(resume_filepath)

                    # Extract text from resume
                    resume_text = extract_text_from_pdf(resume_filepath)

                    # Inside each resume processing block
                    candidate_type = categorize_candidate(resume_text)

                    # Rank the resume based on the job description
                    score = rank_resume(resume_text, job_desc_text)

                    # Append score, filename, and feedback to results
                    results.append({
                        'filename': resume_filename,
                        'score': score,
                        'job_position': job_position,
                        'candidate_type': candidate_type,
                        'feedback': get_improvement_suggestions(resume_text, job_desc_text, candidate_type)
                    })

    elif upload_type == '3':  # One Resume, Multiple Job Descriptions
        resume = request.files['resume']
        job_desc_files = request.files.getlist('job_desc')

        if resume and job_desc_files:
            # Save the resume
            resume_filename = secure_filename(resume.filename)
            resume_filepath = os.path.join('uploads', resume_filename)
            resume.save(resume_filepath)

            # Extract text from the resume
            resume_text = extract_text_from_pdf(resume_filepath)

            # Process each job description
            for job_desc_file in job_desc_files:
                if job_desc_file.filename.endswith('.pdf'):
                    # Save the job description file
                    job_desc_filename = secure_filename(job_desc_file.filename)
                    job_desc_filepath = os.path.join('uploads', job_desc_filename)
                    job_desc_file.save(job_desc_filepath)

                    # Extract text from the job description
                    job_desc_text = extract_text_from_pdf(job_desc_filepath)

                    # Extract job position dynamically
                    job_position = extract_job_position(job_desc_text)

                    # Inside each resume processing block
                    candidate_type = categorize_candidate(resume_text)

                    # Rank the resume based on the job description
                    score = rank_resume(resume_text, job_desc_text)

                    # Append score, filename, and feedback to results
                    results.append({
                        'filename': resume_filename,
                        'job_desc_filename': job_desc_filename,
                        'score': score,
                        'job_position': job_position,
                        'candidate_type': candidate_type,
                        'feedback': get_improvement_suggestions(resume_text, job_desc_text, candidate_type)
                    })



    return render_template('result.html', results=results)


def get_improvement_suggestions(resume_text, job_desc_text, candidate_type):
    """Generate improvement suggestions based on resume and job description comparison."""
    suggestions = []

    # Display the candidate type in the suggestions
    suggestions.append(f"Candidate Type: {candidate_type}")

    # Check for missing keywords in resume
    missing_keywords = check_missing_keywords(resume_text, job_desc_text)
    if missing_keywords:
        missing_keywords_str = ', '.join(missing_keywords)  # Convert list of missing keywords to a string
        suggestions.append(
            f"Your resume could benefit from incorporating more specific terms related to the job description. For example, consider adding the following missing keywords: {missing_keywords_str}.")


    return suggestions


def check_missing_keywords(resume_text, job_desc_text):
    """Compare keywords between resume and job description using advanced tokenization."""
    # Process text with spaCy to tokenize and process words
    resume_doc = nlp(resume_text.lower())  # Process resume text (lowercase for uniformity)
    job_desc_doc = nlp(job_desc_text.lower())  # Process job description text (lowercase)

    # Extract tokens (words) from the resume and job description
    resume_tokens = set([token.text for token in resume_doc if not token.is_stop and not token.is_punct])
    job_desc_tokens = set([token.text for token in job_desc_doc if not token.is_stop and not token.is_punct])

    # Find missing keywords in the resume that are in the job description
    missing_keywords = job_desc_tokens - resume_tokens
    return missing_keywords


@main.route('/download', methods=['GET'])
def download():
    # Retrieve and parse the 'results' query parameter
    results = request.args.get('results', '[]')  # Default to an empty JSON array if not provided
    try:
        results_list = json.loads(results)  # Parse the JSON string into a list
    except json.JSONDecodeError:
        return "Invalid 'results' format", 400  # Handle invalid JSON gracefully

    # Prepare the results in a CSV format
    output = io.StringIO()
    output.write("Resume,Ranking Score,Suggestions\n")
    for result in results_list:
        # Ensure the result is a dictionary with expected keys
        filename = result.get('filename', 'Unknown')
        score = result.get('score', 'N/A')
        feedback = result.get('feedback', [])
        suggestions = "; ".join(feedback) if isinstance(feedback, list) else feedback  # Handle list or string

        output.write(f"{filename},{score},{suggestions}\n")

    # Create a response to return the file as a download
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=results.csv"
    return response


def categorize_candidate(resume_text):
    """
    Categorize a candidate as a fresher or experienced based on resume content.
    Args:
        resume_text (str): Text extracted from the resume.
    Returns:
        str: 'Fresher' or 'Experienced'
    """
    experience_keywords = ["years of experience", "worked at", "employment history", "professional experience"]
    education_keywords = ["recent graduate", "intern", "fresher", "no experience"]

    for keyword in experience_keywords:
        if keyword in resume_text.lower():
            return "Experienced"

    for keyword in education_keywords:
        if keyword in resume_text.lower():
            return "Fresher"

    return "Fresher"  # Default to fresher if unclear
