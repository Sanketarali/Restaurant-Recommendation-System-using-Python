# Restaurant-Recommendation-System-using-Python
A restaurant recommendation system is an application that recommends similar restaurants to a customer according to the customer’s taste

# How did I do?<br>
let’s import the necessary Python libraries and the dataset we need for this task:<br><br>
import numpy as np<br>
import pandas as pd<br>
from sklearn.feature_extraction import text<br>
from sklearn.metrics.pairwise import cosine_similarity<br>

data=pd.read_csv('TripAdvisor_Restauarant.csv')<br>
data.head()<br>

![result](https://github.com/Sanketarali/Restaurant-Recommendation-System-using-Python/blob/main/Screenshot%20(3001).png)

<h3>I will select two columns from the dataset for the rest of the task (Name, Type):</h3><br><br>
data=data[['Name','Type']]<br>
data<br>

![result](https://github.com/Sanketarali/Restaurant-Recommendation-System-using-Python/blob/main/Screenshot%20(3004).png)

<h3>Before moving forward, let’s have a look at whether the data contains any null values or not:</h3><br>
data.isnull().sum()<br>

Name     0<br>
Type    13<br>
dtype: int64<br>

<h3>So the data has some null values in the Type column. I will delete the rows containing null values before moving forward:</h3><br><br>
data = data.dropna()<br>

<h3>The type of restaurant is a valuable feature in the data to build a recommendation system. The type column here represents the category of restaurants. For example, if a customer likes vegetarian-friendly restaurants, he will only look at the recommendations if they are vegetarian friendly too. So I will use the Type column as the feature to recommend similar restaurants to the customer</h3><br>

feature = data["Type"].tolist()<br>
tfidf = text.TfidfVectorizer(input=feature, stop_words="english")<br>
tfidf_matrix = tfidf.fit_transform(feature)<br>
similarity = cosine_similarity(tfidf_matrix)<br>

<h3>Now I will set the name of the restaurant as an index so that we can find similar restaurants by giving the name of the restaurant as an input:</h3><br>
indices = pd.Series(data.index, index=data['Name']).drop_duplicates()<br>

<h3>function to recommend similar restaurants:</h3><br>

def restaurant_recommendation(name, similarity = similarity):<br>
    index = indices[name]<br>
    similarity_scores = list(enumerate(similarity[index]))<br>
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)<br>
    similarity_scores = similarity_scores[0:10]<br>
    restaurantindices = [i[0] for i in similarity_scores]<br>
    return data['Name'].iloc[restaurantindices]<br>
print(restaurant_recommendation("Market Grill"))<br>

![result](https://github.com/Sanketarali/Restaurant-Recommendation-System-using-Python/blob/main/Screenshot%20(3003).png)
