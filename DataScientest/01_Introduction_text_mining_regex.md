# Introduction to text mining - Regex (Regular Expressions)

## Introduction: Library re

Like in other ML fields, datas have to be preprocessed to be understood by the algorithm later on. In text field, the goal is to **identify regular expressions**. A regex is a string of character following a **pattern**. In python we use the library **re**.

The simplest regex is the string itself, for example if we are looking for all strings "Bonjour" in a text, the associated regex would be "Bonjour".


For optimisations purposes, a regex has to be **compiled** first, and then executed by an engine and detected in a textfile finally.
- The function **compile** compiles the regex. It takes in argument the regex and return an object type **pattern**
_Note_: To specify we are using a raw string we can place a r (like raw string, equivalent of f strgin format I guess) before the regex.
```python
import re
r = re.compile("a") # will search for a in the text.
r = re.compile(r"a")
```

- the functions **findall**, **finditer, search, match** to detect the regex.

findall(), is **applied to the compiled regex** and takes in argument **the text where we are searching the regex**. It returns a list of the words corresponding to the regex.

**Example**: Find nous in a specific sentence

```python

r = re.compile (r"nous") # build the regex and compile it.
txt = 'A la maison, nous avons une télévision. Nous sommes heureux.'
r.findall(txt) # look for the regex in a text

>>> ['nous']
```

**Note**:
1- re library is case sensitive and here it detected only nous and not Nous.
2- We are limited if we have to type every single possibility of regex.

So we will introduce tools to simplify those tasks such as:
- string classes
- quantifier
- metacharacters
- group of capture

Those tools allow to write more compact regex and find more diverse informations.
