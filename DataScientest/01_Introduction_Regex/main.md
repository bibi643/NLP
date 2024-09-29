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



## String character classes

**String classes are used to represent a group of characters in a concise way**. 

A regex  defining a string class starts and finish by **[]**.

It can be defined in an:
- **extensive** way: we list all the characters to find in a text between [] like [aZ6] to dectect a, Z, or 6.
- **intensive** way: if we want to retrieve a structured sequence, we specify the first and last character with - in the middle. [a-d] to find a,b,c,d.
- **negative** way: to **exclude** characters we place the character **^** before the chracters. [^aZ]] to find everthing except a or Z.
- **union** of classes: **concatenating** character classes like [a-zA-Z] to detect all small and capital letters.

_Example_: 
- Let's define a regex to detect all capital letters.
- Find all in text

```python
r =re.compile(r"[A-Z]")
              
r.findall(txt)
>>>['A','N']
```



![Alt text](./Images/01_character_classes.png)

Below a cheat sheet of the regex we can build using **character classes**.


![CheatSheet](./Images/01_cheatSheet.png)



## Quantifiers

**Quantifiers** are symbols to specify a certain number of repetitions of characters., group or classes of characters to retrieve in a string of chracters.


_Examples_:
- * to detect if an expression is present 0,1 or multiple times. -> cha*t to detect cht, chat, chaat, chaaaat... (equivalent to {0,})
- + to detect if an expression is present 1 or more times -> cha+t to detect chat, chaat, chaaaat... (equivalent to {1,})
- ? to detect if an epxression is present 0 or 1 -> cha?t to detect cht, chat.(equivalent to {0,1})
- {n} to detect an exact number of repetition. -> cha{2}t to detect chaat. 
- {n,} to detect at least n repetition. -> cha{2,}t to detect chaat, chaaat, chaaaat...
- {n,m} to detect n times but no more than m times. -> cha{1,3}t to detect chat, chaat, chaaat


All those quantifiers can be applied on class characters like ch[a-z]?t to detect cht, chat, chbt, chct...chzt

_Example_

- Compile a regex to detect numbers in txt.

```python
txt = "J'habite au 137 bd Auguste Blanqui, au 12ème étage"

### Insérez votre code
r = re.compile(r"[0-9]+")
r.findall(txt)

>>>['137','12']
```
As I did in my first attempt, using * instead + will also returns the spaces.

- Compile a regex to find 3 digits numbers

```python
r = re.compile("[0-9]{3,}")
r.findall(txt)
>>>['137']

```

Here a list of shortcuts


![ShortCuts](./Images/01_shortcuts_regex.png)

