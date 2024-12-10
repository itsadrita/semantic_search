# Semantic Similarity Prediction System for Question-Answer Matching 🚀  

---

## 📖 Overview  

This project implements a semantic similarity prediction system to accurately match 64 questions with their corresponding 64 answers from a mixed array of 128 passages using **NLP**, **transformer models**, and **ensemble learning techniques**. The system uses **bi-encoder models**, **cross-encoder scoring**, and **parallelized preprocessing** to optimize performance and accuracy.

---

## 🛠️ Technologies  

- **Programming Language**: Python  
- **Libraries & Frameworks**:  
  - **SentenceTransformers**  
  - **CrossEncoder**  
  - **NumPy**  
  - **scikit-learn**  
  - **nltk**, **TweetTokenizer**  
  - **joblib**  
  - **JSON**  
  - **multiprocessing**  
  - **Pandas**, **Matplotlib**, **Hugging Face Transformers**  

---

## 🔮 Objective  

To accurately match questions to answers by leveraging semantic similarity computations using pre-trained **transformer models** (bi-encoders and cross-encoders) and ensemble techniques.  

---

## 📊 How it Works  

1. **Extract & Preprocess Text**: Extracted and tokenized questions and answers using **nltk** and **TweetTokenizer** with stopword filtering. Parallelization is applied to optimize preprocessing.
2. **Generate Bi-Encoder Embeddings**: Used pre-trained Sentence Transformers models (e.g., `all-mpnet-base-v2`) to compute embeddings for both questions and answers.
3. **Compute Ensemble Similarity**: Integrated similarity scores from multiple bi-encoder models using ensemble learning to identify the most promising matches.
4. **Cross-Encoder Scoring**: Selected the top-k candidates from the bi-encoder similarity scores and used **cross-encoder scoring** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to refine results.
5. **Pipeline Execution**: Parallelized pipelines speed up computations. Results are validated and logged in **JSON** for transparency.

---

## 🖥️ Features  

- **Ensemble Learning Accuracy**: Accuracy was improved from **50% to 91.08%** by combining multiple bi-encoder models and refining predictions with a cross-encoder.
- **Parallelized Preprocessing**: Optimized preprocessing workflows with multiprocessing and **joblib** to handle large text data efficiently.
- **Model Integration**: Utilized state-of-the-art **SentenceTransformer** models and **cross-encoder scoring** for semantic matching accuracy.
- **Data Normalization**: Applied **scikit-learn's normalize** to standardize embeddings for consistent similarity calculations.
- **JSON Logging**: Results and computations are saved using **JSON** for transparency and reproducibility.

---

## ⚙️ Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/itsadrita/semantic_search.git
