# Sentiment Analysis

The goal is to develop a ML algorithm to perform a sentiment analysis after creating a corpus, and prepare it (preprocessing) and transform it in matrix terms-document.
Movies_comments.csv has 2 columns.
- Text for the comments
- Sentiment for the sentiments 1 being positive and 0 negative.

We will use this dataset ot build an alogrithm which will be able later to ddetermine positive or negative sentiment on any new phrases.


## Count Vectorizer

### Import the data


```python
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('movies_comments.csv')
df.head(2)

Text 	Sentiment
0 	Brokeback Mountain'is so beautiful, and so am... 	1
1 	I liked Harry Potter and I'll be sorry to see... 	1

```

### Check if the dataset is balanced

We will use **countplot** from seaborn

```python
import seaborn as sns
sns.countplot(x=df['Sentiment'])
```

![balanced_dataSet](./Images/countplot.png)

We can consider this dataset as balanced.

- Split into train and test sets.

-- First we separate features (X) and label (y).
-- Split X and y into train and test

```python
from sklearn.model_selection import train_test_split
X = df.Test
y= df.Sentiment

X_train,X_test, y_train,y_test = train_test_split(X,y, test_size =.2, random_state=234)

```

### Tokenisation - Preparation before ML processing
We want to use the algorithm **Gradient Boosting Tree** because it works very well in sentiment analysis. It fins the weights tto optimise the lost function in a classification problem. To use it we need to convert the strings in the Text column into numerical tokens -> **Bag of words**.


> fit_transform applied to a text train a model and return tokens. transform apply the tokenisation to the text based on the model we trained previously.

```python
vectorizer = CountVectorizer()
# train the vecotrizer on X_train and update X_train affecting the returned tokens = definition of fit_transform
X_train= vectorizer.fit_transform(X_train)

#update X_test with the tokens using the model trained before = definition of transform method
X_test= vectorizer.transform(X_test)

```

### ML processing

#### Train the model & Predictions

So now our data set is prepared to be used with the classifier **GradientBoostingTree**:
- create the classifier instance
- train it on X_train and y_train, remember that thot variables have been tokenised (X)
- predict on X_test

```python
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators =100 , learning_rate = 1, max_depth=1, random_state=30)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

```


#### Model evaluation

We can evaluate the model using a classification report and have an overiview using a confusion matrix
- classification_report(y_test,y_pred)
- confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

```
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred)
confusion_matrix = pd.crosstab(y_test,y_pred, rownames=['Classes Reelles'],colnames = ['Classes predites'])


              precision    recall  f1-score   support

           0       0.98      0.89      0.94       131
           1       0.91      0.99      0.95       144

    accuracy                           0.94       275
   macro avg       0.95      0.94      0.94       275
weighted avg       0.95      0.94      0.94       275

Classe prédite 	0 	1
Classe réelle 		
            0 	117 	14
            1 	2 	142

```

The model seems quite good...but we will test it on totally new datas just in case (risk of overfitting).
**Remember to tokenise the comments!!! using the method transform on the vectorizer**


```python
# Example 1
comments=['I am upset',
          'I am happy to be here, but also sad.',
          'I am grateful to be around all of you.',
          'What the hell am I doing with my life']

tokenized_comment = vectorizer.transform(comments)

clf.predict(tokenized_comment.toarray())
>>>array([1,1,1,1])

#Example 2

comments = ["Harry Potter 7 part I is an amazing movie even it is not adapted for children.", 
            "I left after 45 min because this movie was boring !", 
            "It was a good waste of time",
            "The slow pace may make the experience less engaging for some viewers.",
            "A poignant film that subtly explores the challenges of forbidden love, driven by remarkable performances."
           ]

tokenized_comments = vectorizer.transform(comments)

clf.predict(tokenized_comments.toarray())
#expected = [1, 0, 0, 0, 1]

>>>array([1,0,1,1,1])

```

Clearly it is not working so well now...

We will see how TF-IDF works.

# TF-IDF

