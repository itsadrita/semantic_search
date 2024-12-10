from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import json
import numpy as np
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from joblib import Parallel, delayed
import nltk
import multiprocessing

nltk.download('stopwords')

# Load bi-encoder models for ensemble
bi_encoder_models = {
    "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
    "multi-qa-mpnet-base-dot-v1": SentenceTransformer("multi-qa-mpnet-base-dot-v1"),
    "all-distilroberta-v1": SentenceTransformer("all-distilroberta-v1"),
    "all-MiniLM-L12-v2": SentenceTransformer("all-MiniLM-L12-v2"),
}

# Load cross-encoder model
cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Initialize stopwords and tokenizer
stop_words = set(stopwords.words('english')) - {"not"}
tokenizer = TweetTokenizer()

# Preprocessing helper function
def preprocess_text(text):
    words = tokenizer.tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha()]
    return " ".join(filtered_words)

# Batch preprocessing with parallelization
def preprocess_texts(texts):
    return Parallel(n_jobs=multiprocessing.cpu_count())(delayed(preprocess_text)(text) for text in texts)

# Function to compute and normalize embeddings
def get_embeddings(model, texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    return normalize(embeddings, axis=1)

# Function to compute similarity scores across bi-encoder models
def compute_ensemble_similarity(models, question_embeddings, answer_embeddings):
    scores = []
    for model_name, model in models.items():
        q_emb = question_embeddings[model_name]
        a_emb = answer_embeddings[model_name]
        score = np.dot(q_emb, a_emb.T)
        scores.append(score)
    return np.mean(scores, axis=0)

# Cross-encoder scoring for top-k candidates
def get_best_match_cross_encoder(cross_encoder, question, answers):
    pairs = [(question, answer) for answer in answers]
    scores = cross_encoder.predict(pairs)
    best_match_idx = np.argmax(scores)
    return best_match_idx

# Function to process one database
def process_database(database_path):
    with open(database_path, 'r') as file:
        data = json.load(file)

    q_indices = data['q_indices']
    a_indices = data['a_indices']
    texts = data['texts']

    questions = preprocess_texts([texts[idx] for idx in q_indices])
    answers = preprocess_texts([texts[idx] for idx in a_indices])

    # Compute bi-encoder similarity
    question_embeddings = {name: get_embeddings(model, questions) for name, model in bi_encoder_models.items()}
    answer_embeddings = {name: get_embeddings(model, answers) for name, model in bi_encoder_models.items()}
    bi_encoder_similarity = compute_ensemble_similarity(bi_encoder_models, question_embeddings, answer_embeddings)

    predicted_pairs = {}
    correct_matches = 0
    top_k = 5  # Number of top candidates to refine with cross-encoder

    for i, similarities in enumerate(bi_encoder_similarity):
        # Get the top-k most similar answers using bi-encoder
        top_candidates_idx = np.argsort(similarities)[-top_k:]
        top_candidates = [answers[idx] for idx in top_candidates_idx]

        # Refine using cross-encoder
        best_match_idx = top_candidates_idx[
            get_best_match_cross_encoder(cross_encoder_model, questions[i], top_candidates)
        ]
        predicted_a_idx = a_indices[best_match_idx]
        predicted_pairs[q_indices[i]] = predicted_a_idx

        # Check if the prediction is correct
        if predicted_a_idx == a_indices[i]:
            correct_matches += 1

    accuracy = correct_matches / len(q_indices)
    return predicted_pairs, accuracy

# Main function to process databases
def process_first_five_databases(data_folder):
    results = {}
    overall_accuracy = 0
    total_files_processed = 0

    for i in range(5):
        filename = f'database{i}.txt'
        database_path = os.path.join(data_folder, filename)
        if os.path.exists(database_path):
            print(f"Processing {filename}...")
            predicted_pairs, accuracy = process_database(database_path)
            results[filename] = {
                "predicted_pairs": predicted_pairs,
                "accuracy": accuracy
            }
            overall_accuracy += accuracy
            total_files_processed += 1
        else:
            print(f"File {filename} not found in {data_folder}!")

    overall_accuracy = overall_accuracy / total_files_processed if total_files_processed > 0 else 0
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    return results

# Save the results
def save_results(results, output_path):
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)

# Example usage
data_folder = 'data/data'  # Replace with your data folder path
output_path = 'results.json'

results = process_first_five_databases(data_folder)
save_results(results, output_path)
print(f"Results saved to {output_path}")
