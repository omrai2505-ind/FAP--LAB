# --- 1. IMPORT LIBRARIES ---
# Make sure you have these libraries installed in your environment:
# pip install streamlit pandas scikit-learn numpy
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# --- 2. CONFIGURE THE STREAMLIT PAGE ---
# This sets the browser tab title, icon, and layout
st.set_page_config(
    layout="wide",
    page_title="AI Book Recommender",
    page_icon="📚"
)

# --- 3. DATA LOADING AND CACHING ---
@st.cache_data
def load_data(csv_path='books_tags.csv'):
    """
    Loads, cleans, and processes the book data.
    This function is cached so it only runs once.
    """
    if not os.path.exists(csv_path):
        st.error(f"Error: The file '{csv_path}' was not found in the same directory.")
        return None

    try:
        # --- THIS IS THE FIX ---
        # Added on_bad_lines='skip' to ignore rows with parsing errors
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        # --- END OF FIX ---
        
    except pd.errors.EmptyDataError:
        st.error(f"Error: The file '{csv_path}' is empty.")
        return None
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Define the columns we need for the model and for display
    model_features = ['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count']
    display_cols = ['bookID', 'title', 'authors']
    cols_to_use = list(set(model_features + display_cols)) # Use set to avoid duplicates

    # Check if all required columns exist
    if not all(col in df.columns for col in cols_to_use):
        st.error(f"CSV file must contain the following columns: {', '.join(cols_to_use)}")
        return None

    df = df[cols_to_use]

    # Data cleaning and type conversion
    for col in model_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows that have missing values in our critical feature columns
    df = df.dropna(subset=model_features)
    
    # Handle potential missing authors for display
    df['authors'] = df['authors'].fillna('Unknown Author')
    
    # Remove duplicate titles to make the selectbox cleaner
    df = df.drop_duplicates(subset=['title'])

    # Reset index to ensure it aligns with the feature matrix
    df = df.reset_index(drop=True)
    return df

# --- 4. MODEL BUILDING AND CACHING ---
@st.cache_resource
def build_model(df, n_neighbors):
    """
    Builds the KNN model from the dataframe.
    This is cached as a "resource" (like a model)
    and will re-run if 'n_neighbors' changes.
    """
    features = ['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count']
    feature_matrix = df[features]

    # Scale the features (crucial for distance-based algorithms like KNN)
    scaler = MinMaxScaler()
    scaled_feature_matrix = scaler.fit_transform(feature_matrix)

    # Initialize and fit the KNN model
    # We ask for n_neighbors + 1 (to account for the book itself)
    # 'cosine' metric is excellent for finding items with similar "direction" or "profile"
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors + 1)
    model_knn.fit(scaled_feature_matrix)

    return model_knn, scaled_feature_matrix

# --- 5. MAIN APPLICATION UI ---
def run_app():
    
    # --- Sidebar for Controls ---
    with st.sidebar:
        st.title("Controls & Settings ⚙️")
        st.markdown("Adjust the recommendation engine parameters.")
        
        # Add a slider for number of recommendations
        num_recs = st.slider(
            "Number of Recommendations:",
            min_value=3,
            max_value=12,
            value=5,
            step=1
        )
        
        st.divider()
        st.markdown(
            "This app uses a **K-Nearest Neighbors (KNN)** model "
            "to find books with similar attributes."
        )

    # --- Main Page Title ---
    st.title("📚 AI Book Recommender")
    st.markdown(
        "Find your next great read! This app recommends books based on their attributes "
        "(rating, page count, review counts) using a machine learning model."
    )
    st.divider()

    # --- Load Data and Build Model ---
    data = load_data()
    
    if data is not None:
        if data.empty:
            st.error("No valid data could be loaded after processing. Please check your CSV file.")
        else:
            # Show a spinner while the model is being built (or retrieved from cache)
            with st.spinner("Building recommendation model... (this is fast!)"):
                model, features = build_model(data, num_recs)
            
            # --- User Input Section ---
            st.header("1. Select a Book You Love")
            
            # Get a sorted list of book titles for the dropdown
            book_list = np.sort(data['title'].unique())
            
            selected_book_title = st.selectbox(
                "Start typing to find a book:",
                book_list,
                index=None,  # Show a placeholder initially
                placeholder="Search for a book in the dataset..."
            )
            
            st.header("2. Get Your Recommendations")
            # Use columns to center the button and make it larger
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                find_button = st.button("Find Similar Books 🚀", type="primary", use_container_width=True)

            if find_button and selected_book_title:
                try:
                    # --- Recommendation Logic ---
                    # Find the index of the selected book
                    book_index = data[data['title'] == selected_book_title].index[0]
                    
                    # Get the feature vector for the selected book
                    query_vec = features[book_index].reshape(1, -1)

                    # Find the nearest neighbors
                    distances, indices = model.kneighbors(query_vec)
                    
                    # Get the indices and distances (skipping the first one, which is the book itself)
                    recommendation_indices = indices.flatten()[1:]
                    recommendation_distances = distances.flatten()[1:]

                    # --- Display Results ---
                    st.header(f"3. Your Top {num_recs} Recommendations")
                    st.subheader(f"Because you liked '{selected_book_title}':")
                    st.write("") # Add some space

                    # Iterate and display recommendations in a clean, card-like format
                    for i, rec_index in enumerate(recommendation_indices):
                        # Ensure rec_index is within bounds (it should be, but good to be safe)
                        if rec_index < len(data):
                            rec_book = data.iloc[rec_index]
                            rec_distance = recommendation_distances[i]
                            
                            # Calculate a "similarity score" (Cosine distance is 0 for identical, 1 for opposite)
                            # So, (1 - distance) * 100 is a nice percentage
                            similarity_score = (1 - rec_distance) * 100

                            with st.container(border=True):
                                col1, col2 = st.columns([1, 4])
                                
                                # Column 1: Similarity Score
                                with col1:
                                    st.metric(
                                        label=f"Rank {i+1}",
                                        value=f"{similarity_score:.1f}%",
                                        help="This score represents the 'similarity' of attributes to your selected book."
                                    )
                                
                                # Column 2: Book Details
                                with col2:
                                    st.subheader(f"{rec_book['title']}")
                                    st.caption(f"by {rec_book['authors']}")
                                
                                st.divider()
                                
                                # Show the key metrics used for comparison
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Average Rating", f"{rec_book['average_rating']:.2f} ⭐")
                                c2.metric("Page Count", f"{int(rec_book['num_pages'])} 📖")
                                c3.metric("Total Ratings", f"{int(rec_book['ratings_count']):,}")
                        else:
                            st.warning(f"Skipped an invalid recommendation index: {rec_index}")


                except IndexError:
                    st.error("Book not found in the processed data. This shouldn't happen, please try another.")
                except Exception as e:
                    st.error(f"An error occurred during recommendation: {e}")
            
            elif find_button and not selected_book_title:
                st.warning("Please select a book first!")

# --- Main execution guard ---
if __name__ == "__main__":
    run_app()

