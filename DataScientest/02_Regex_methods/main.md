# Introduction to text mining,common mehtods.


re has 4 major methods to search regex in a text file. Those methods are **findall**, **finditer**, **match**, **search**.

## findall() & finditer()

findall() and finditer() find all string characters which follow the schema describe by the regex. 
- findall() returns a list.
- finditer() returns an iterable.

The iterator allows to go through a list of values in a loop but it will load the element in memory only when necessary.
Both methods are quite similar.  The objects in the iterator are type **match**, so to be displayed we need to use the method **group**.



```python
txt = "findall et finditer"
reg = re.compile("([a-zA-Zàéè\']+)")

# Utilisation de findall
mots = reg.findall(txt)
# Tous les mots sont stockés dans la mémoire
print('Pour la méthode findall(), on a :', mots)

print('Les mots trouvé sont donc :')
for mot in mots:
    print(mot)


# Utilisation de finditer
matches = reg.finditer(txt)
# Les mots sont chargés dans la mémoire sur demande
print('\nPour la méthode finditer(), on a :',matches)

print('Les mots trouvé sont donc :')
for match in matches:
    # match est un objet de type Match
    print(match.group(1))
```


## The method match()

The method match() only returns the first occurence of a string of character described by a regex. findall () returns all occurences. Moreover the regex has to be at the begining of the text or it will be returned None.

Match is also interesting when working with group capture, ie (pattern_to_indentify).


```python
r = re.compile('(a(b)c)d')

#Utilisation de group
#.group(0) renvoie la chaîne entière détectée par l'expression
print('group(0) renvoie', g.group(0))

# .group(1) renvoie le premier groupe capturé
print('group(1) renvoie', g.group(1))

# .group(2) renvoie le deuxième groupe capturé
print('group(2) renvoie', g.group(2))


#Utilisation de groups
# .groups() renvoie tous les groupes capturés
print('groups() renvoie', g.groups())


>>>group(0) renvoie abcd
group(1) renvoie abc
group(2) renvoie b
groups() renvoie ('abc', 'b')
```


## search()




## split()

The method split can cut a text on a string character describe by a regex.

Let's separate a phrase by the words, aka the separator will be anything that is not a word -> [^a-zA-Z0-9_]= \W

```python
#Compilateur
r = re.compile(r"\W+")

#Exemple
txt = "L'exemple... parfait pour comprendre"

#Split
print(r.split(txt))

>>> ['L', 'exemple', 'parfait', 'pour', 'comprendre']

```

```python

#Compilateur

r = re.compile(r"\W")

​

#Exemple

txt = "L'exemple... parfait pour comprendre"

​

#Split

print(r.split(txt))

>>>['L', 'exemple', '', '', '', 'parfait', 'pour', 'comprendre']


```

What is the difference between \W and \W+... well the + means at least one character, so it filters the spaces.

To keep apostrophes...

```python
#compilateur
r = re.compile(r"[^A-Za-z0-9_']+")

#split
print(r.split(txt))

>>>["L'exemple", 'parfait', 'pour', 'comprendre']
```
If parenthesis are used for group capture, the values of the separators are also displayed.

```python
r = re.compile(r"(\W+)")

print(r.split(txt))

​
>>>
['L', "'", 'exemple', '... ', 'parfait', ' ', 'pour', ' ', 'comprendre']
```

## The method sub()

This method serrve to substitute a string of character by another one.
- create a regex to detect the term wwe want to change
- we use sub(new_term, text_to_modify)

```python
r= re.compile(r"super!")
txt = "c'est super cool comme superstition")
print (r.sub('cool', txt)

>>>#compilateur

r = re.compile(r"super")

​

txt = "c'est super cool comme superstition"

​

print(r.sub('cool', txt))

​c'est cool cool comme **cool**stition
```

We need to use the tag \b...\b to specify we want only this pattern to subsitute and not this pattern inside another one

```python
r = re.compile(r"\bsuper\b")
print(r.sub('cool', txt))

>>>r = re.compile(r"\bsuper\b")

print(r.sub('cool', txt))

​

c'est cool cool comme superstition






## Verbose

re.verbose est un outil intéressant qui peut vous aider à commenter les patterns à l'aide de #, dans l'optique d'être plus compréhensible et lisible. Par exemple, pour identifier les liens html, on passe de re.compile(r"https?://[A-Za-z0-9\.\-/]+"), qui est illisible, à

 re.compile(r"""

   https?                       #identifie http ou https
   ://                          #commun à tous les liens
   [A-Za-z0-9\.\-/]+            #suite du lien 

""", re.VERBOSE)



```python
txt = 'g.petoit93@gmail.com oliver.small459@orange.fr \n m.lameinère@yahoo.fr'
r = re.compile(r"""[A-Za-z0-9\.-éà]+  #n'importe quelle suite de caractères au moins 1 fois
                    @                 #le caractère @
                    [a-zA-Z\.-]+      #la suite de caractères après le @
                    """, re.VERBOSE)

email = r.findall(txt)
print(email)

```


