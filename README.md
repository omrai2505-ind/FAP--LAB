# 📚 AI Book Recommender

An interactive web application built with Streamlit that recommends books based on their attributes using Machine Learning. 

## 📖 Overview
Finding your next great read is easy with the AI Book Recommender. This app uses a **K-Nearest Neighbors (KNN)** model with cosine similarity to find books with similar profiles based on their average rating, page count, and review metrics. 

## ✨ Features
* **Interactive UI:** A clean and user-friendly web interface powered by Streamlit.
* **Customizable Recommendations:** Use the sidebar slider to adjust the number of book recommendations you want to receive (between 3 and 12).
* **Machine Learning Engine:** Utilizes Scikit-learn's NearestNeighbors algorithm on scaled numerical features (`average_rating`, `num_pages`, `ratings_count`, `text_reviews_count`).
* **Similarity Scoring:** Calculates and displays a percentage-based similarity score for each recommended book.
* **Robust Data Handling:** Automatically cleans missing data and handles CSV parsing errors seamlessly.

## 🛠️ Tech Stack
* **[Streamlit](https://streamlit.io/):** For building the front-end web application.
* **[Scikit-learn](https://scikit-learn.org/):** For the K-Nearest Neighbors (KNN) machine learning model and data scaling.
* **[Pandas](https://pandas.pydata.org/):** For data loading, manipulation, and cleaning.
* **[NumPy](https://numpy.org/):** For numerical operations.

## 🚀 Installation and Setup

### Prerequisites
Make sure you have Python installed on your system. 

### 1. Clone the repository 
(If applicable, otherwise navigate to your project directory)
```bash
cd your-repository-folder
