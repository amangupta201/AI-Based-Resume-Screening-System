import random

import torch
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch import cosine_similarity as torch_cosine_similarity
from transformers import BertTokenizer, BertModel


# Preprocessing function (example, modify according to your needs)
def preprocess_text(text):
    # Perform your text preprocessing here (e.g., lowercasing, removing stopwords, etc.)
    text = text.lower()
    return text

# Q-Learning model for RL to suggest personalized rankings
class RLModel:
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2  # Exploration vs Exploitation

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)

    def update_q_value(self, state, action, reward):
        q_value = self.get_q_value(state, action)
        new_q_value = q_value + self.learning_rate * (reward + self.discount_factor * max(self.q_table.get((state, a), 0) for a in ["rank_1", "rank_2", "rank_3", "rank_4"]) - q_value)
        self.q_table[(state, action)] = new_q_value

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            # Exploration: Choose a random action
            return random.choice(["rank_1", "rank_2", "rank_3", "rank_4"])
        else:
            # Exploitation: Choose the best action based on Q-values
            return max(["rank_1", "rank_2", "rank_3", "rank_4"], key=lambda action: self.get_q_value(state, action))

# Initialize RL model
rl_model = RLModel()

# Function to get TF-IDF matrix
def get_tfidf_matrix(resume, job_desc):
    """
    Get TF-IDF matrix for the given resume and job description.
    """
    vectorizer = TfidfVectorizer()
    corpus = [resume, job_desc]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix

# Function to get Word2Vec embedding
def get_word2vec_embedding(text, model_path='model.bin'):
    """
    Get Word2Vec embedding for the given text.
    """
    words = text.split()
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    embedding = sum([model[word] for word in words if word in model])
    return embedding

# Function to get BERT embedding
def get_bert_embedding(text):
    """
    Get BERT embedding for the given text.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
    return embedding

# Function to calculate cosine similarity based on TF-IDF
def calculate_cosine_similarity(tfidf_matrix):
    """
    Calculate cosine similarity between the resume and job description based on TF-IDF.
    """
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

# Function to rank resume using personalized RL-based approach
def rank_resume(resume, job_desc):
    # Preprocess the texts
    resume_cleaned = preprocess_text(resume)
    job_desc_cleaned = preprocess_text(job_desc)

    # Extract features (TF-IDF, Word2Vec, or BERT embeddings)
    tfidf_matrix = get_tfidf_matrix(resume_cleaned, job_desc_cleaned)
    tfidf_similarity = calculate_cosine_similarity(tfidf_matrix)

    # Get BERT embeddings
    resume_embedding = get_bert_embedding(resume_cleaned)
    job_desc_embedding = get_bert_embedding(job_desc_cleaned)

    # Convert embeddings to tensors (if they are numpy arrays)
    resume_embedding_tensor = torch.tensor(resume_embedding)
    job_desc_embedding_tensor = torch.tensor(job_desc_embedding)

    # Calculate the cosine similarity for embeddings (using torch_cosine_similarity)
    embedding_similarity = torch_cosine_similarity(resume_embedding_tensor, job_desc_embedding_tensor).item()

    # Calculate a final score combining all factors (you can tune the weights here)
    final_score = 0.5 * tfidf_similarity + 0.5 * embedding_similarity

    # Incorporating RL: Update Q-table based on internal reward function
    state = f"{resume_cleaned[:100]} | {job_desc_cleaned[:100]}"  # Simplified state representation
    action = rl_model.choose_action(state)

    # Define a reward based on the final score
    if final_score > 0.8:
        reward = 1  # Best match
        rank = "Excellent Match"
    elif final_score > 0.6:
        reward = 0.7  # Good match
        rank = "Good Match"
    elif final_score > 0.4:
        reward = 0.3  # Average match
        rank = "Average Match"
    else:
        reward = -1  # Poor match
        rank = "Poor Match"

    # Update the RL model with the internal reward
    rl_model.update_q_value(state, action, reward)

    # Format suggestions based on ranking
    suggestion = f"Score: {final_score:.2f}"
    suggestion += f"\nSuggested Rank: {rank}"

    return suggestion
