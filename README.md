# Sentiment-Analysis
Objective
To develop a deep learning algorithm to detect various types of sentiment in a collection of English sentences within a large paragraph. The goal is to accurately predict the overall sentiment of the paragraph.

Introduction / Description of Internship
This project focused on developing a deep learning algorithm to automate sentiment analysis of textual comments and feedback. The goal was to accurately detect different types of sentiment in a collection of English sentences and predict the overall sentiment of paragraphs. This represents a significant step towards enhancing the efficiency and accuracy of sentiment analysis in large datasets. In today's digital era, vast amounts of textual data are generated daily. Understanding the sentiment expressed in this data is crucial for businesses and organizations to make informed decisions. This project addresses the need for automated sentiment analysis by developing a deep learning algorithm that can detect different types of sentiment, such as positive, negative, and neutral, in English sentences. This project aims to provide a reliable and scalable solution for analyzing sentiment in textual comments and feedback.

Activities
1.	Dataset Selection: Used the Women's Clothing E-Commerce Reviews dataset for sentiment analysis.

2.	Data Collection and Preprocessing: Imported and preprocessed the dataset using google colab. This included data cleaning, handling missing values, and text. 
preprocessing tasks such as tokenization, lemmatization, and removing stop words.

3.	Sentiment Analysis: Employed the VADER lexicon for sentiment analysis, assigning sentiment scores and labels (Positive, Negative, Neutral) to each review text based on the compound scores.

4.	Data Visualization: Created various visualizations using Matplotlib and Plotly to illustrate sentiment distribution, sentiment scores by rating, review length analysis, and word clouds representing sentiment categories.

5.	Machine Learning Models: Implemented Logistic Regression and LSTM models for sentiment prediction and classification. Trained, evaluated, and used these models to predict sentiment for new reviews.


6.	Report and Analysis: Prepared a comprehensive project report detailing methodology, assumptions, exceptions, algorithms, challenges, opportunities, reflections, recommendations, outcomes, conclusions, and enhancement scope.

   Approach / Methodology
1.	Data Collection:
•	Obtained the "Women's Clothing E-Commerce Reviews" dataset using Pandas.
•	imported the dataset into a Pandas DataFrame for analysis.

2.	Data Preprocessing:
•	Checked the first 10 rows of the dataset to understand its structure and contents.
•	Examined the data summary using df.info() to understand data types, missing values, and other information
•	Checked for missing values using df.isnull().sum() and handled them by replacing NaN values in the 'Review Text' column with empty strings
•	Cleaned the dataset by dropping unnecessary columns like 'Unnamed: 0' and 'Clothing ID'.
•	•Merged relevant text columns ('Title' and 'Review Text') into a single column named 'Text'
•	Calculated and added the length of each review text as a new column 'Text_Length' for analysis.

3.	Sentiment Analysis
•	Used the VADER lexicon for sentiment analysis.
•	Imported the SentimentIntensityAnalyzer from NLTK and created a sentiment analyzer object.
•	Calculated sentiment scores for each review using sid.polarity_scores(text).
•	Added sentiment scores as a new column, 'Sentiment Score', in the DataFrame.
•	Mapped the compound sentiment scores to sentiment labels ('Positive', 'Negative', 'Neutral') using a custom function.

4.	Data Visualization
•	Visualized the sentiment distribution using a pie chart to show the percentage of positive, negative, and neutral sentiments in the dataset.
•	Created pie charts and bar plots to analyze sentiment labels based on ratings (1 to 5) to understand sentiment variations across different ratings.
•	Plotted a box plot to visualize the distribution of ratings by sentiment categories ('Positive', 'Negative', 'Neutral').
•	Generated histograms to visualize the distribution of ratings and review lengths in the dataset.

5.	Correlation Analysis:
Calculated the correlation matrix between 'Review Length' and 'Sentiment Score' to explore the relationship between review length and sentiment intensity.

6.	Data Insights:
Used visualizations and analysis to understand the distribution of sentiment labels, their relationship with ratings, and patterns in review length and sentiment scores.
Identified key words and phrases for positive, negative, and neutral sentiments using word clouds.

7.	Machine Learning / Deep Learning Models:
•	Built a sentiment prediction model using Logistic Regression and evaluated its    performance using classification metrics such as precision, recall, and F1-score.
•	Pre-processed the text data for the sentiment prediction model using TF-IDF vectorization.
•	Trained and tested a logistic regression model to predict sentiment labels ('Positive', 'Negative', 'Neutral') based on review texts.

8.	Evaluation:
•	Evaluated the logistic regression model's performance using classification report to get accuracy, precision, recall, F1-score, and AUC metrics.
•	Plotted AUC and loss curves to visualize model training and validation performance over time.


9.	Assumptions:
•	The dataset accurately represents customer sentiments about women's clothing products.
•	The VADER lexicon provides reliable sentiment analysis for English text.
•	The machine learning models' predictions reflect actual sentiments expressed in the reviews.

10.	 Exceptions:
•	The analysis focuses solely on English textual comments within the provided dataset.
•	External factors influencing sentiments, such as context or cultural nuances, are not considered in this analysis.

Algorithms 
1	VADER Sentiment Analysis: VADER is a lexicon-based method used for sentiment analysis. It assigns sentiment scores by examining text for words with predefined sentiment polarities. This approach is effective for quickly assessing sentiment in text.
2	Logistic Regression: Logistic Regression is a type of supervised learning algorithm used for sentiment prediction tasks. It analyzes features and their relationship to sentiment labels, providing results that are easy to interpret. It performs well with linearly separable data.
3	LSTM (Long Short-Term Memory): LSTM is a deep learning model used for sequence-based sentiment analysis. It belongs to the family of recurrent neural networks (RNNs) and is particularly effective for understanding context and nuances in sentiment expression over long. 

Outcome / Conclusion
The project successfully automated sentiment analysis of textual comments and feedback, providing actionable insights into customer sentiments regarding women's clothing products. The deep learning models demonstrated good accuracy in predicting sentiment labels, contributing to effective feedback analysis and decision-making processes.
