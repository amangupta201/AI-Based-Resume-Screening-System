# AI-Based-Resume-Screening-System
# Overview
The AI-Based Resume Screening System is designed to streamline the process of screening resumes for job applications. The system ranks resumes based on their relevance to the job description by utilizing various AI techniques, including TF-IDF, Word2Vec, and BERT embeddings, alongside a Q-learning model to personalize the ranking. This tool allows users to upload one or multiple resumes and job descriptions, providing a ranking for each resume based on its compatibility with the job description.

# Key Features
One Resume, One Job Description: Upload a single resume and a single job description to get a ranking based on the relevance of the resume to the job description.
Multiple Resumes, One Job Description: Upload multiple resumes and get ranked results for each, based on their match with the job description.
One Resume, Multiple Job Descriptions: Upload a single resume and multiple job descriptions, and receive a ranking for each job description, showing the best matching ones.
# Tech Stack
Python
TensorFlow / PyTorch
Transformers (Hugging Face)
Gensim (for Word2Vec)
Scikit-learn (for TF-IDF and cosine similarity)
Q-Learning (Reinforcement Learning)
# How It Works
The system works by comparing the content of resumes and job descriptions using multiple AI models and algorithms. Here’s a breakdown of the process:

Preprocessing: Texts are cleaned by converting them to lowercase and removing unnecessary characters.
Feature Extraction:
TF-IDF: The system uses Term Frequency-Inverse Document Frequency (TF-IDF) to transform the resume and job description into feature vectors.
Word2Vec: Google’s pre-trained Word2Vec model is used to get embeddings for the words in both the resume and job description.
BERT Embeddings: BERT embeddings are extracted from both the resume and job description for a more contextually aware representation of the text.
Cosine Similarity: Cosine similarity is used to measure the similarity between the vectors.
Reinforcement Learning (Q-learning): The system utilizes a Q-learning model to personalize the ranking of resumes based on historical interactions, continually improving its recommendations.
Ranking Logic
Cosine Similarity: Calculates the similarity between resume and job description using TF-IDF and embeddings.
Q-learning Model: Incorporates reinforcement learning to improve ranking by evaluating the quality of the matches (using states and actions for ranking suggestions).

The ranking output is divided into four categories:

Excellent Match
Good Match
Average Match
Poor Match
# Installation
To run this system locally, you need to set up your environment with the required dependencies:

1. Clone the repository
git clone https://github.com/your-username/ai-resume-screening.git
cd ai-resume-screening
2. Install Dependencies
Create a virtual environment and install the required libraries:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Pre-trained Models
The system uses Google’s Word2Vec model for word embeddings (model.bin). You can download it from here or use your own pre-trained Word2Vec model.

Place the model.bin file in the project directory.
# Usage
Once the system is set up, you can use it in the following modes:

Mode 1: One Resume, One Job Description
Upload a single resume and a single job description. The system will output a ranking based on how well the resume matches the job description.

Mode 2: Multiple Resumes, One Job Description
Upload multiple resumes and one job description. The system will output the ranking of each resume in relation to the job description.

Mode 3: One Resume, Multiple Job Descriptions
Upload one resume and multiple job descriptions. The system will rank how well the resume matches each job description.

Example
Here's an example of how the system works using Python:

# Initialize the ranking system
resume = "Your resume content here."
job_desc = "Job description content here."

# Rank the resume for one job description
result = rank_resume(resume, job_desc)

# Output the result
print(result)
The output will provide a score and a suggested rank (e.g., "Excellent Match", "Good Match", etc.).

Contributing
Feel free to fork the repository, make improvements, or add new features. Pull requests are always welcome.

License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgements
Google Word2Vec for pre-trained embeddings.
BERT from Hugging Face for contextual text embeddings.
PyTorch for deep learning models.
Scikit-learn for feature extraction and similarity calculations.
Let me know if you'd like to add any other sections or features to the README!







