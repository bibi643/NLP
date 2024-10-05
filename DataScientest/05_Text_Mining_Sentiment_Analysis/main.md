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

The other method we saw in previous notebook was TF-IDF. We will proceed exactly the same way as before.
- Separate fetaures, labels
- Split in train and test sets.
- Create a vectorizer and train and update it on X_train
- transform the X_test in tokens using the model trained
- create a classifier and fit and predict
- Evaluate the classifier model

```python
X_tfidf = df.Text
y_tfidf= df.Sentiment

X_train_tfidf,X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y_tfidf, test_size = .2, random_state= 30)


vec_tfidf = TfidfVectorizer()
X_train_tfidf = vec_tfidf.fit_transform(X_train_tfidf)
X_test_tfidf = vec_tfidf.transform(X_test_tfidf)


clf_tf_idf = GradientBoostingClassifier(n_estimator = 100, learning_rate =1, max_depth=1, random_state=30)
clf_tfidf.fit(X_train_tfidf,y_train_tfidf)
y_pred_tfidf= clf_tfidf.predict(X_test_tfidf)


print(classification_report(y_test_tfidf,y_pred_tfidf)
pd.crosstab(y_test_tfidf, y_pred_tfidf, rownames= ['Real Classes'], colnames= ['Predicted Classes'])
```

Let's try it again on new random comments from ourselves. So **first we have to tokenize** the comment before feeding/doing predictions with the model.

```python

### Insérez votre code ici
comments=['I am upset',
          'I am happy to be here, but also sad.',
          'I am grateful to be around all of you.',
          'What the hell am I doing with my life']
tokenized_comment = vec_tfidf.transform(comments)

clf.predict(tokenized_comment.toarray())

>>>array([1, 0, 0, 0, 1])
```

Notice now how _good waste of time_ is negative!!!

# Stemming

For now we worked on raw text, without any pretreatment like **stemming or lemming** as we saw in another notebook.
Let's try this now!

Remember stemming consists in transforming each word to its stem.

- Create a stemmer


```python
from nltk.stem.snowball import PorterStemmer

stemmer = PorterStemmer()


def stemming(mots):
  sortie = []
  for string in mots:
    radical = stemmer.stem(string)
    if (radical not in sortie): sortie.append(radical)
  return sortie
```


- Apply the stem on the text. So we need to separate features and label again!!

```python
X_stem = df.Test
y_stem = df.Sentiment
#Pour effectuer le stemming, il faut décomposer les phrases en suite de mots
X_stem = X_stem.str.split()
for i in range (0, len(X_stem)):
    X_stem[i] = stemming(X_stem[i])
    X_stem[i] = ' '.join(X_stem[i]) #On rassemble ici notre suite de mots en phrases
```

>_Note_: ' '.join(X_stem[i]) means we join a space with X_stem[i])


```

Now we perform exactly the same as before aka vectorizer using TfidVectorizer and building a classifier GradientBoostingClassifier and evaluate it.

```python
X_train_stem,X_test_stem,y_train_stem,y_test_stem =train_test_split(X_stem,y_stem,test_size =.2,random_state=30)
vec_stem = TfidfVectorizer()
X_train_stem = vec_stem.fit_transform(X_train_stem)
X_test_stem = vec_stem.transform(X_test_stem)


clf_stem = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth =1, random_state = 0)

clf_stem.fit(X_train_stem,y_train_stem)
y_pred_stem=clf_stem.predict(X_test_stem)


# Calcul et affichage de classification_report
print(classification_report(y_test_stem, y_pred_stem) )

# Calcul et affichage de la matrice de confusion
conf_matrix_stem = pd.crosstab(y_test_stem, y_pred_stem, rownames=['Classe réelle'], colnames=['Classe prédite'])
conf_matrix_stem

>>>
              precision    recall  f1-score   support

           0       0.95      0.97      0.96       131
           1       0.97      0.96      0.97       144

    accuracy                           0.96       275
   macro avg       0.96      0.96      0.96       275
weighted avg       0.96      0.96      0.96       275

Classe prédite 	0 	1
Classe réelle 		
            0 	127 	4
            1 	6 	138

```


# Lemmatisation
We saw in another notebook that the lemmatisation was more advanced than stemmisation. Let's checkout now!


```python
### Importer le package nécessaire
from nltk.stem import WordNetLemmatizer

# Initialiser un lemmatiseur
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatisation(mots) :
    sortie = []
    for string in mots :
        radical = wordnet_lemmatizer.lemmatize(string)
        if (radical not in sortie) : sortie.append(radical)
    return sortie
```


```python
# Séparer le jeu de données en données d'entraînement et données test 
X_train_lem, X_test_lem, y_train_lem, y_test_lem = train_test_split(X_lem, y_lem, test_size=0.2, random_state = 30)

vec_lem = TfidfVectorizer()

# Mettre à jour la valeur de X_train_lem et X_test_lem
X_train_lem = vec_lem.fit_transform(X_train_lem)
X_test_lem = vec_lem.transform(X_test_lem)

# Créer un classificateur clf_lem et entraîner le modèle sur l'ensemble d'entraînement
clf_lem = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train_lem, y_train_lem)

# Calculer les prédictions 
y_pred_lem = clf_lem.predict(X_test_lem)


# Calcul et affichage de classification_report
print( classification_report(y_test_lem, y_pred_lem) )

# Calcul et affichage de la matrice de confusion
conf_matrix_lem = pd.crosstab(y_test_lem, y_pred_lem, rownames=['Classe réelle'], colnames=['Classe prédite'])
conf_matrix_lem

```

This is even better

# StopWords and Regex

To go even further we can do one last thing. Dealing with stopwords and regex to filter useless words.
```python

df['Text'].head(20)

0      Brokeback Mountain'is so beautiful, and so am...
1      I liked Harry Potter and I'll be sorry to see...
2                     The Da Vinci Code'it was AWESOME.
3                      and i love brokeback mountain...
4                        but I LOVE brokeback mountain.
5     " Brokeback Mountain " is a great short story ...
6     " Brokeback Mountain " was a beautiful movie a...
7       " Brokeback Mountain " was also very excellent.
8                " brokeback mountain "-it was awesome.
9     " Brokeback Mountain was an awesome movie. < 33..
10    " Harry Potter " series ( because of the delic...
11    " I could have discussed the abortion ban in S...
12                           " I love you Harry Potter!
13    " I loved Brokeback Mountain and I haven't see...
14    " Mission Impossible 3 " was awesome and if yo...
15    " Now some people will say to me, Joe, I liked...
16             " The Da Vinci Code " is awesome though.
17    @ shraddha: You are a Harry Potter fan too, th...
18           * brokeback mountain is an awesome movie..
19                      * dies * I love Harry Potter...

```

We can see here words like 
- connectors: and, so..
- special characters: , <>, @
- punctuation: !,?,
- and more


Because stopwords and regex are cxase sensitive we put everything in lower case.
```python

#On met le texte en minuscule
df["Text"] = df["Text"].str.lower()
df["Text"].head()

```
To apply stopwrds we need to split sentences in a list of words, using word_tokenize directly on the coprus

```python
#Importer le package pour la tokenization
from nltk.tokenize import word_tokenize

df["Text"] = df["Text"].apply(word_tokenize)
df.head()

 	                            Text 	                    Sentiment
0 	[Brokeback, Mountain'is, so, beautiful, ,, and... 	1
1 	[I, liked, Harry, Potter, and, I, 'll, be, sor... 	1
2 	[The, Da, Vinci, Code'it, was, AWESOME, .] 	        1
3 	[and, i, love, brokeback, mountain, ...] 	          1
4 	[but, I, LOVE, brokeback, mountain, .] 	            1

```

We can use now stopwords. We will use the english stopwords

```python
# Importer stopwords de la classe nltk.corpus
from nltk.corpus import stopwords

# Initialiser la variable des mots vides
stop_words = set(stopwords.words('english'))
print(stop_words)
```

We will add more stopwrods to the set using **update** and then apply the function stop_words_filtering we created in a previous notebook.

```python
new_stop_words = [",", ".", "``", "@", "*", "(", ")", "...", "!", "?", "-", "_", ">", "<", ":", "/", "=", "--", "©", "~", ";", "\\", "\\\\"]

#Ajouter à la liste des stopwords des éléments de syntaxe que nous avons repéré et qui ne servent pas à l'analyse du texte
stop_words.update(new_stop_words)

#Définir la fonction stop_words_filtering
def stop_words_filtering(mots) : 
    tokens = []
    for mot in mots:
        if mot not in stop_words:
            tokens.append(mot)
    return tokens

df["Text"] = df["Text"].apply(stop_words_filtering)
df['Text'].head()

rokeback, Mountain'is, beautiful, amazing, f...
1    [I, liked, Harry, Potter, I, 'll, sorry, see, ...
2                   [The, Da, Vinci, Code'it, AWESOME]
3                          [love, brokeback, mountain]
4                       [I, LOVE, brokeback, mountain]
Name: Text, dtype: object

```

Now we need to join everything together
```python
#Reconstitution des phrases après la tokenization
for i in range (0, len(df['Text'])):
    df.loc[i, 'Text'] = ' '.join(df.loc[i, 'Text'])

df['Text'].head()

0    Brokeback Mountain'is beautiful amazing freaki...
1            I liked Harry Potter I 'll sorry see grow
2                         The Da Vinci Code'it AWESOME
3                              love brokeback mountain
4                            I LOVE brokeback mountain
Name: Text, dtype: object

```


There are still some parasites like ..., numbers... We will delete all these using regex.



```python
#Importer le package re
import re

#Regex pour tous les '.', '..', '...', ...
for i in range (0, len(df['Text'])):
    df.loc[i, 'Text'] = re.sub(r"\.+", '', df.loc[i, 'Text'])

#Regex pour tous les chiffres et nombres
for i in range (0, len(df['Text'])):
    df.loc[i, 'Text'] = re.sub(r"[0-9]+", '', df.loc[i, 'Text'])

#Vérification que tous les éléments qu'on a voulu enlever ont bien été enlever du dataframe
filtered = [i for i in df['Text']]
print(filtered)

```

And here we go again like before, vecotrizing etc etc etc..

```python
# Séparer la variable explicative de la variable à prédire
X_vf, y_vf = df.Text, df.Sentiment

#Pour effectuer la lemmatisation on sépare les phrase pour on les rassemble
X_vf = X_vf.str.split()

for i in range (0, len(X_vf)):
    X_vf[i] = lemmatisation(X_vf[i])
    X_vf[i] = ' '.join(X_vf[i])

# Séparer le jeu de données en données d'entraînement et données test 
X_train_vf, X_test_vf, y_train_vf, y_test_vf = train_test_split(X_vf, y_vf, test_size=0.2, random_state = 30)

vec_vf = TfidfVectorizer()

# Mettre à jour la valeur de X_train_vf et X_test_vf
X_train_vf = vec_vf.fit_transform(X_train_vf)
X_test_vf = vec_vf.transform(X_test_vf)



# Créer un classificateur clf_vf et entraîner le modèle sur l'ensemble d'entraînement
clf_vf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train_vf, y_train_vf)

# Calculer les prédictions 
y_pred_vf = clf_vf.predict(X_test_vf)


# Calcul et affichage de classification_report
print(classification_report(y_test_vf, y_pred_vf) )

# Calcul et affichage de la matrice de confusion
conf_matrix_vf = pd.crosstab(y_test_vf, y_pred_vf, rownames=['Classe réelle'], colnames=['Classe prédite'])
conf_matrix_vf
```



# Conclusion

Nous avons pu mettre en application dans un problème de Machine Learning les différents aspects abordés dans le module 131 Text Mining, à savoir la prédiction de sentiment avec le modèle de Gradient Boosting à l'aide de :

- l'algorithme Bag of Words avec le CountVectorizer
- l'algorithme Bag of Words avec le TF-IDF
- le regroupement de famille de mots selon leur racine avec la racinisation, ou stemming
- le regroupement de famille de mots selon leur lemme avec la lemmatisation
- la suppression des stopwords
- la suppression des regexs

    Les modèles basés sur l'algorithme Bag of Words sont aujourd'hui les plus fondamentaux dans le Traitement Automatique du Langage Naturel (ou Natural Language Processing en anglais). Pour aller plus loin sur les possibilités d'analyse et de traitement de texte, une autre méthode d'apprentissage, le **Word Embedding, est également exploitée pour le Traitement Automatique du Langage Naturel et est capable de capturer le contexte, la similarité sémantique et syntaxique (genre, synonymes, ...) d'un texte**. On peut ainsi imaginer des pistes d'améliorations supplémentaires pour nos prédictions sur l'analyse de sentiments.







