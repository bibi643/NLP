# Word_Cloud

# Data Preparation

We will use the techniques we saw previously to create a word cloud. We need to calculate the frequency of the word of a corpus. We will use **pandas**, **NLTK**, **wordlcoud**.
The data set here are comments from movie reviewers.

```python
import pandas as pd
df = pd.read_csv('movies_comments.csv')
df.head()
df.shape
>>>
                                                Text  Sentiment
0   Brokeback Mountain'is so beautiful, and so am...          1
1   I liked Harry Potter and I'll be sorry to see...          1
2                  The Da Vinci Code'it was AWESOME.          1
3                   and i love brokeback mountain...          1
4                     but I LOVE brokeback mountain.          1
Taille du dataset : (1371, 2)


```

We see the df has 2 columns:
- text
- Sentiment


Let's compile/put all together the comments.
```python
text = ""
for comment in df.Text : 
    text += comment

# Importer stopwords de la classe nltk.corpus
from nltk.corpus import stopwords

# Initialiser la variable des mots vides
stop_words = set(stopwords.words('english'))
print(stop_words)

```


# Wordcloud Library

The library wordCloud implements an algortihm  to display a wordcloud of a text.
Steps:
- Tokenise the text passed in parameter.
- filter the stopwords
- calculate the frequency of the words
- Visual rep of keywords, with highest frequency with a cloud shape

The method WordCloud has multiple parameters:
- background_color
- max_words
- stopwords is a string to specify the words to filter from the corpus.
