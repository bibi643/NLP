# Community Detection
Ce notebook se focalise sur **l'analyse des réseaux sociaux**. 

L'objectif principal est d'être capable de résoudre des problèmes de **détection de communautés au sein de graphes.**
Par définition, un réseau social désigne un ensemble d'individus (au sens large : famille, collègues, organisations, ...) qui sont liés d'une certaine manière et qui interagissent. Le développement, et donc l'analyse, des réseaux sociaux ont connu une très forte accélération ces dernières années.


# Détection des communautés¶

Une communauté est formée par un ensemble d'individus qui interagissent plus souvent entre eux qu'avec les autres. Il s'agit donc de groupes d'individus qui ont formés des liens plus forts ou qui ont des affinités communes.

La détection de communautés a pour rôle de mettre en évidence ces groupes, formés implicitement et ne résultant pas d'un choix explicite. Elle permet ainsi d'identifier des profils types, d'effectuer des actions ciblées, de mieux ajuster les recommandations et d'identifier les acteurs centraux.

Par exemple, dans une entreprise les employés forment des communautés explicites (les services), mais certains vont collaborer plus souvent avec d'autres et former des communautés implicites. La détection de communautés permet de les identifier.

# Représentation d'un réseau social

Un réseau social peut être représenté par un graphe 𝐺(𝑉,𝐸) où 𝑉 représente l'ensemble des sommets (vertices en anglais) et 𝐸 l'ensemble des arêtes (edges en anglais). On rappelle qu'un graphe peut être représenté à l'aide de sa matrice adjacence (𝐴𝑖𝑗).

Au sens du graphe, une communauté est constitué par un ensemble de nœuds qui sont fortement liés entre eux, et faiblement liés avec les nœuds situés en dehors de la communauté.

**Détecter les communautés dans un réseau est un problème qui se ramène souvent à un problème de clustering**. Les communautés peuvent être des groupes disjoints ou des groupes avec chevauchement. Nous traiterons principalement des cas de figure où un sommet ne peut appartenir qu'à une communauté, pour des raisons de simplicité, notamment sur les points théoriques.


**image1**

Exemple d'application : le réseau de Karaté de Zachary

Le réseau de club de Karaté de Zachary est composé de deux communautés disjointes. En effet il décrit les relations d'amitiés entre 34 membres d'un club de Karaté observées lors de la gestion d'un conflit durable entre l'entraîneur "Mr. Hi" et l'administrateur du club "John A". Ce réseau est composé de 78 arêtes liant les membres qui ont interagi dans un autre contexte que les entrainements du club.

La moitié des membres ont formé un nouveau club autour de M. Hi et les membres de l'autre partie ont trouvé un nouvel instructeur ou abandonné le karaté. En se basant sur les données collectées, Zachary a correctement assigné tous les membres du club sauf un aux groupes qu'ils ont rejoints après le conflit.

(a) Exécuter la cellule de code ci-dessous pour créer et afficher le réseau social du club de Karaté

```python
    # Importer les packages nécessaires

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
%matplotlib inline

# Fixer la graine, afin d'avoir des valeurs reproductibles
np.random.seed(1)

# Générer le réseau de karaté de Zachary
z = nx.karate_club_graph()

# Listes des noeuds de chaque communautés
mr_hi_node = [0]
mr_hi_group = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21]
john_a_node = [33]
john_a_group = [9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# Positionner les noeuds en utilisant l'algorithme Fruchterman-Reingold
# Il permet un positionnement idéal des points sur un graphique pour une visualisation claire.
pos = nx.fruchterman_reingold_layout(z)

# Afficher les deux communautés du réseau de Karaté
plt.figure(figsize = (20, 10))
nx.draw_networkx_nodes(z, pos, node_size = 400, nodelist = mr_hi_group, node_color = "#272839")
nx.draw_networkx_nodes(z, pos, node_size = 800, nodelist = mr_hi_node, node_color = "#272839")
nx.draw_networkx_nodes(z, pos, node_size = 800, nodelist = john_a_node, node_color = "#75DFC1")
nx.draw_networkx_nodes(z, pos, node_size = 400, nodelist = john_a_group, node_color = "#75DFC1")
nx.draw_networkx_edges(z, pos, alpha = 0.3, edge_color = "#48dbc8")
nx.draw_networkx_labels(z, pos, {0:"Hi", 33:"John"}, font_color = "white")
plt.show()
```
**image2**



Le réseau de Karaté de Zachary est stocké dans la variable z.

(b) Calculer le nombre d'arêtes et de sommets du réseau z.

(c) Vérifier que le réseau de Karaté est connexe.

(d) Affecter respectivement aux variables A, diameter et density la matrice d'adjacence, le diamètre et la densité du réseau z.

Le diamètre d'un graphe est la plus grande distance parmi les plus courtes distances entre deux noeuds. Cette valeur donne une idée de l'étendue d'un graphe, et de la manière dont les noeud sont inter-connectés. 



```python
# Nombre de sommets et d'arêtes
print("Nombre d'arêtes \t:", z.size())
print("Nombre de sommets \t:", z.order())

# Vérifier que le réseau est connexe
is_connected = nx.is_connected(z)
print("Le réseau de Karaté est", (1 - is_connected) * "non", 'connexe')

A = nx.adjacency_matrix(z)
diameter = nx.diameter(z)
density = nx.density(z)

>>>
Nombre d'arêtes 	: 78
Nombre de sommets 	: 34
Le réseau de Karaté est  connexe

```


# Approche Agglomérative

L'approche agglomérative consiste à définir une mesure de similarité entre les sommets à partir de la matrice d'adjacence et à ensuite utiliser la Classification Ascendante Hiérarchique (CAH). L'algorithme commence par isoler tous les points dans une communauté, puis à grouper les communautés les plus similaires ensemble.

Il existe de nombreuses métriques de similarité comme celle de Jaccard ou cosinus. Dans cet exercice, nous utiliserons le plus court chemin entre les noeuds. Il est calculé par le biais de la fonction all_pairs_shortest_path_length de NetworkX qui renvoie un dictionnaire contenant la longueur du plus court chemin entre deux sommets.

(e) Dans un dictionnaire path_length, stocker les plus courts chemins entre tous les noeuds du graphe
(f) Faire une double boucle pour créer une matrice de distance distances à partir de ce dictionnaire.

 

Cette matrice devra donc être une matrice carrée dont la dimension sera égale au nombre de noeuds dans le graphe.

```python
# Calculer la longueur de plus court chemin entre tous les sommets du graphe z
path_length = dict(nx.all_pairs_shortest_path_length(z))

# Définir la matrice distances
n = len( z.nodes() )
distances = np.zeros((n, n))
for u in path_length.keys() :
    for v in path_length[u].keys() :
        distances[u][v] = path_length[u][v]
```


Une fois la matrice de similarité générée, nous appliquons la classification ascendante hiérarchique. Elle va grouper les individus les plus semblables entre eux. Pour cela on utilise la fonction linkage du sous-package scipy.cluster.hierarchy.

(g) Importer la fonction linkage du sous package scipy.cluster.hierarchy
(h) Dans une variable hclust stocker le résultat de la CAH appliquée sur la matrice des distances avec la méthode de Ward. La méthode de Ward permet de grouper les clusters.

```python
from scipy.cluster.hierarchy import linkage

hclust = linkage(distances, method = 'ward')
```
Nous allons maintenant utiliser une technique de visualisation pour les clustering hiérarchiques : le dendrogramme. Cette figure montre les différents groupes obtenus à chaque étape et les liaisons entre eux. La première ligne représente les données, les nœuds représentent les regroupements auxquels les données appartiennent. La hauteur de chaque nœud est proportionnelle à la valeur de la dissemblance intergroupe entre ses deux noeuds fils.

(i) Importer la fonction dendrogram du sous package scipy.cluster.hierarchy.
(j) Utiliser la fonction dendrogram appliquée à hclust.

 

Le paramètre entiercolor_threshold permet de définir la manière dont sera colorée le dendrogramme, mettant en évidence les clusters identifiés
```python

from scipy.cluster.hierarchy import dendrogram


# Initialisation de la figrue
plt.figure(figsize=(20, 10))
# Affichage du dendrogramme
plt.title("Dendrogramme CAH")
dendrogram(hclust, color_threshold = 17)
plt.show()
```
**image3**

Approche Agglomérative basée sur la maximisation de la modularité

La force d'une communauté est évaluée par l'écart entre le nombre de connexions observées et le nombre obtenu si les connexions avaient été distribuées aléatoirement entre les nœuds. Mathématiquement, cette force est :

    𝑄(𝐶)=∑𝑖,𝑗∈𝐶𝐴𝑖𝑗−𝑑𝑖𝑑𝑗2𝑚

- 𝑚 est le nombre de connections observées
- 𝐶 est la communauté étudiée
- 𝑑𝑖, 𝑑𝑗 sont les degrés de centralité des sommets 𝑖 et 𝑗

La modularité d'un réseau après partitionnement en K communautés mesure la qualité d'un partitionnement. Mathématiquement, cette modularité est :

12𝑚∑𝑘=1𝐾∑𝑖,𝑗∈𝐶𝑘(𝐴𝑖,𝑗−𝑑𝑖𝑑𝑗2𝑚)

La modularité prend des valeurs entre −1 et 1. On peut considérer qu'un graphe a une structure de communautés significative quand un partitionnement obtient un score de modularité supérieur à 0,3.

**La méthode de Louvain consiste à maximiser la fonction de modularité pour partitionner un graphe**, en s'assurant que le nombre et le poids des liens est plus important à l'intérieur des partitions qu'entre les partitions. 
Au lancement de l'algorithme:
- tous les sommets appartiennent à une partition différente.
- Ils sont regroupés itérativement dans des partitions de modularité optimale.
- Arrivé à une première situation d'optimum, les sommets partitionnés ensemble sont groupés et traités ensuite comme un seul sommet.
- On reprend alors à la première étape, et ce jusqu'à ce qu’il n'y ait plus aucun gain de modularité possible.

**La fonction best_partition de la classe community.community_louvain permet de trouver la meilleure partition en utilisant la méthode de Louvain.**

(k) Importer la fonction best_partition de la classe community.community_louvain.

(l) Stocker dans une variable best_part le partitionnement retourné la méthode de Louvain.

```python
# Importer la classe community_louvain
from community.community_louvain import best_partition

# Calculer et afficher la meilleure répartition des groupes
best_part = best_partition(z)
print('La meilleure répartition est:\n', best_part)

>>>La meilleure répartition est:
 {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0, 8: 2, 9: 0, 10: 1, 11: 0, 12: 0, 13: 0, 14: 2, 15: 2, 16: 1, 17: 0, 18: 2, 19: 0, 20: 2, 21: 0, 22: 2, 23: 3, 24: 3, 25: 3, 26: 2, 27: 3, 28: 3, 29: 2, 30: 2, 31: 3, 32: 2, 33: 2}

```
Nous allons maintenant afficher le partitionnement effectué sur le graphe du groupe de karaté.

L'objet retourné par la fonction best_partition contient les indices des clusters assignés à chaque point. On les récupère avec la méthode values.

(m) Créer une liste node_cluster contenant le cluster auquel appartiennent les noeuds du graphe.

(n) Afficher le graphique en utilisant la disposition de Fruchterman Raynold.

(o) Afficher les noeuds en les colorant selon leur cluster.

(p) Afficher les arêtes et les étiquettes.

```python
# Afficher les différents groupes de la meilleure répartition du réseau z

plt.figure(figsize=(10,10))
plt.axis('off')
pos = nx.fruchterman_reingold_layout(z)
node_cluster = list(best_part.values())
nx.draw_networkx_nodes(z, pos, node_color = node_cluster)
nx.draw_networkx_edges(z, pos, alpha = 0.5)
nx.draw_networkx_labels(z, pos, {0:"Hi", 33:"John"}, font_color = "white")
plt.show()

```
**image4**


# Approche divisive, algorithme de Girvan-Newman

Dans cette partie, nous présentons une autre approche de détection de communautés au sein d'un graphe. Plutôt que de grouper les sommets, nous allons voir comment il est possible d'identifier les arêtes qui font le lien entre des communautés quasi-disjointes.

L'importance d'une connexion entre deux sommets peut être matérialisée par le edge betweenness. Le edge betweenness indique la fréquence à laquelle la connexion est empruntée quand on considère le plus court chemin entre chaque paire de nœuds. Plus la valeur est élevée, plus la connexion est importante : elle établit un pont entre des groupes de sommets.

Le principe de l'algorithme de Girvan-Newman est de couper les ponts entre les communautés afin de les isoler. Pour cela, l'algorithme fonctionne de la façon suivante :

Tant qu'il y a des arêtes :

- Evaluer l'edge betweeneness de toutes les arêtes
- Supprimer l'arête avec la plus grande edge betweeneness

La méthode girvan_newman de la classe networkx.algorithms.community, appliquée à un graphe , renvoie un itérateur contenant les partitions successives créées.

(q) Importez la fonction girvan_newman du sous package networkx.algorithms.community.

(r) Stockez dans la variable communities_generator le résultat de l'algorithme de Girvan-Newman sur le graphe du club de karaté.

```python
from networkx.algorithms.community import girvan_newman
communities_generator = nx.algorithms.community.girvan_newman(z)
```

On obtient alors un itérateur, que l'on peut utiliser de la manière suivante :

# Accéder à la première répartition des sommets du graphe G 
top_level_communities = next(communities_generator)
# Accéder à la répartition suivante 
next_level_communities = next(communities_generator)

A chaque fois que l'on appelle la fonction next on obtient un nouveau partitionnment, sous la forme d'un tuple d'ensembles.

(s) Dans une variable community affectez le premier partitionnement créé par l'algorithme de Girvan-Newman et l'afficher.

```python
community = next(communities_generator)
community

>>>({0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21},
 {2, 8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33})
```


# Évaluation de l'approche divisive

Afin d'évaluer la qualité d'une partition, la fonction performance de la classe networkx.algorithms.community.quality compare la somme des arêtes intercommunautaires absentes et des arêtes intracommunautaires avec le nombre d'arêtes dans un graphe complet ayant le même nombre de sommets. Cette fonction prend en paramètres un graphe de type NetworkX Graph et une répartition des sommets de ce graphe représentée sous forme d'une séquence de sommets. Elle renvoie un score entre 0 et 1 qui mesure la qualité d'un partitionnement.

(t) Importez la fonction performance du sous package networkx.algorithms.community.quality et calculez la performance du partitionnement community.
```python
from networkx.algorithms.community.quality import performance

performance(z,community)
>>>0.61
```

Nous allons maintenant déterminer quelle répartition donne la meilleure performance.

(u) Recalculer le générateur communities_generator grâce à la fonction girvan_newman.

(v) Bouclez sur les éléments du générateur pour déterminer lequel fournit la meilleure performance. Faites en sorte de conserver une liste des scores dans la variable scores et la meilleure répartition dans la variable best_partition.

(w) Tracez la courbe de la performance en fonction du nombre de tours de boucle et affichez la meilleure répartition.

 On fait ici le choix de recalculer le générateur des communautés pour ne pas avoir de soucis en cas d'exécutions multiples des cellules précédentes.

  ```python
# Calculer un itérateur 
communities_generator = nx.algorithms.community.girvan_newman(z)

#On initialise les variables nécessaires à la boucle
best_score = 0
scores = []

# Mesure de la performance de chaque partitionnement
for partition in communities_generator :
    cur_score = performance(z, partition)
    
    if(cur_score >= best_score) :
        best_partition = partition
        best_score = cur_score
        
    scores.append(cur_score)

# Tracer les performances de chaque partitionnement
plt.plot(range(len(scores)), scores);
print(best_partition)

>>>({0, 1, 3, 7, 13}, {2, 28}, {10, 4}, {16, 5, 6}, {32, 33, 8, 23, 29, 30}, {9}, {11}, {12}, {14}, {15}, {17}, {18}, {19}, {20}, {21}, {22}, {24, 25, 31}, {26}, {27})

```
**image5**

# Conclusion

Le score obtenu par l'approche divisive est très haut puisqu'on obtient plus de 0.9. En revanche, c'est au prix d'un détection de communauté très moyenne puisque 13 points sont isolés dans cette répartition.

De façon générale, quelque soit l'approche (agglomérative ou divisive) et quelque soit l'algorithme, les métriques de score sont des indicateurs, mais la détection de communautés doit également reposer sur les connaissances du DataScientist.


```
