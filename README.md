# NLP Text Classification Project

## Project Overview

This project focuses on text classification using Natural Language Processing (NLP) techniques. The goal is to classify news articles into one of five categories: Politics, Technology, Sports, Business, and Entertainment. The project demonstrates the end-to-end pipeline for processing text data and applying machine learning models to automate the categorization process.

## Problem Statement

The objective is to develop a solution that can classify a collection of news articles into predefined categories based on their content. The categories are:

- Politics
- Technology
- Sports
- Business
- Entertainment

**Attributes**:
- **Article**: The text of the news article.
- **Category**: The corresponding category of the article (Politics, Technology, Sports, Business, Entertainment).

## Approach

1. **Data Preprocessing**:  
   - Text cleaning by removing non-alphabetical characters.
   - Tokenization of the text.
   - Stopword removal to eliminate common non-informative words.
   - Lemmatization to reduce words to their base forms.

2. **Feature Extraction**:  
   - Applied two feature extraction methods:
     - **Bag of Words (BoW)**
     - **TF-IDF (Term Frequency - Inverse Document Frequency)**

3. **Model Training**:  
   - Performed a 75:25 train-test split of the dataset.
   - Trained and compared the following classification models:
     - Naive Bayes
     - Decision Tree
     - Random Forest
     - K-Nearest Neighbors (KNN)
   
4. **Model Evaluation**:  
   - Used metrics to evaluate model performance:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - Confusion Matrix
     - ROC and Precision-Recall Curves

## Key Insights

- **Text Preprocessing**: Implemented an efficient text preprocessing pipeline with tokenization, stopword removal, and lemmatization to prepare the data for classification.
- **Best Performing Model**: The Random Forest model performed the best, achieving the highest accuracy and balanced precision-recall scores.
- **Category Distribution**: Addressed skewed data distribution with class balancing techniques to improve model performance for underrepresented categories.

## Results

- **Best Model**: Random Forest  
  - **Accuracy**: 0.85  
  - **Precision**: 0.78  
  - **Recall**: 0.80  
  - **F1-Score**: 0.79  
  - **ROC-AUC**: 0.92  

## Recommendations

- **Random Forest for Classification**: Based on the results, Random Forest is recommended for automating the classification of news articles into their respective categories.
- **Continuous Model Retraining**: Regular model retraining with updated data will help adapt the model to evolving article content.
- **Feature Importance Monitoring**: Analyze feature importance scores to understand the most informative words contributing to classification.

## Future Enhancements

- **Deep Learning Models**: Explore the use of advanced deep learning models like LSTM, GRU, or transformer-based models (e.g., BERT) to further improve classification performance.
- **Data Augmentation**: Consider techniques for data augmentation to improve class balance, especially for underrepresented categories.

## Technology Stack

- **Programming Language**: Python
- **Libraries Used**:
  - Numpy
  - Pandas
  - Matplotlib
  - Seaborn
  - scikit-learn
- **Algorithms**: Naive Bayes, Decision Tree, Random Forest, K-Nearest Neighbors (KNN)

## Author

Arjun Shrivas  
*Data Science Enthusiast*


