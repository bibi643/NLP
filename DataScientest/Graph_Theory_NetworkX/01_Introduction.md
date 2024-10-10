# Introduction to Graphs

Graphs theory is between mathematics and CS. This field study relationships and structures usings networks of nodes interconnected.

We will see here how to:
- understand vocabulary and structures of graphs.
- Handle the libray NetworkX, to create, visualise and analyse graphs.
- Discover the main algorithm of graphs theory.
- Detect communities using the algo PageRank.

# What is a graph?

A graph is a structure made of nodes and tensors/spines. Nodes represents entities and tensors the relationship between those nodes. Graphs can represent social noetwork or transport networks.

The theory of graphs is useful to solve problems such as:
- what is the shortest path between 2 specific nodes? 
- Is a graph connected? aka is it possible to go from any  node to another random node?

Some real case use are:
- establish the best route for a commercial worker?
- How to place infrastructure in an optimal way?
- How to attribute class rooms to learners without overlaying?


## Types of graphs

- Non-Oriented graphs: Tensors are not oriented. If there is a tensor between A and B, it means A is connected to B AND B is connected to A.
- Oriented graphs: Tensors are oriented. This means A is connected to B but B is not connected to A. Sometimes tensors can have weights representing the force of the relation, distance and cost...

![Type of Graphs](Oriented_NonOriented_graphs.jpeg)


From a mathematical pov, we can define a graph G as:
- A certain number of nodes (S).
- A certain number of couples of Nodes called Tensors (A).


![Graphs](graphe_oriente_02.jpg)

- S = {1,2,3,4,5,6,7}
- A = {(1,3),((2,1),(2,4),(2,3),(3,2),(3,3),(3,4),(3,5),(6,4),(6,6),(7,6)}.

(6,6), (3,3) are called loops.



# First step with NetworkX

NetworkX can 
- create
- delete
- modify a graph in term of node or Tensor.
- Has the main algorithm for the graphs theory


```python
import networkx as ns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('edgelist.csv', index_col = 0)
df.head()

 	source 	target
0 	0 	    1
1 	0 	    2
2 	0 	    3
3 	0 	    4
4 	0 	    5
  ```

This df is a list of tensors/aretes. Each line is a tensor connecting the node source to the node target. We can convert this df into a graph.

We will explain later all this code.

```python
G = nx.from_pandas_edgelist(df) # création du graphe à partir de df
nx.draw_networkx(G,font_color="white") # affichage du graphe

plt.show()
```
![Graphs](graph_NetwrokX.png)


We can see
- 12 nodes (0-11).
- From the node 0 there are 11 tensors.
- From the node 1, there 11 tensors too, but one is coming from the node 0, so we don't have to count it twice. We have then, 11+10+...+1 in this graph -> 66 tensors.


# Creating a Graph

```python
G1 = nx.Graph() # Non oriented Graph
G2 = nx.DiGraph() # Directed Graph

```

# Add a node to a Graph.

We just created a graph. So we need fill it now.
A node can be any hashable object. It can be a string, a numerical value, an image, a function, another graph...

We can use :
- **G.add_node(n,attr)**:to **add** an unique node n to the Graph G. It has an optional argument _attr_ representing a collection of attributes in a key/value format. Usually it is used, color, label, style, but we can use custom attr.
```python
G.add_node(1, name='Daniel', role= 'Support')
```
- **G.add_nodes_from(nodes,attr)**: to add several nodes simultaneously to the graph G. The argument _attr_ is applied to all nodes.
  ```python
  G.add_nodes_from([2,3],role='mentor')
  ```

- **nx.path_graph(n)**: to generate a graph with a shape of path composed of n nodes. Nodes are numbered sequentially from 0 to n-1 and are connected by the tensors. If we want to retrieve the nodes of this path we use as before the method **add_nodes__from()**. If tensors have to be kept, the method **nx.compose(G,path)** combines the path with the graph G fusionning the nodes and the tensors.

'''python 
chemin = nx.path_graph(10) # Create a path of 10 nodes
G = nx.compose(G, chemin) # fuse chemin and G
'''

_Note_: It is possible to retrieve the list of neighbour nodes of a specific nodes n, using **list(G.adj[n])**

_Example_: Add 12 nodes to the graph

```python
# Utilisation de path_graph pour générer un graphe en forme de chemin et l'ajouter à G1
H = nx.path_graph(3)
G1 = nx.compose(G1, H) # combine les 2 graphes pour conserver le chemin H

# Utilisation de add_node pour ajouter un nœud unique avec des attributs
G1.add_node(4)
G1.add_node(5)
G1.add_node(6)
G1.add_node(7, weight="2")
G1.add_node(print)
G1.add_node("Daniel")

# Utilisation de add_nodes_from pour ajouter plusieurs nœuds avec un attribut commun
G1.add_nodes_from([10, 11, 12], color='blue')

# Visualisation du graphe
nx.draw(G1, with_labels=True, font_color='black', node_size=500, node_color='#75DFC1') 
plt.show()

```

## Add Tensors/spines to the graphs.

We need now to connect the nodes of the graph. In NetworkX a tensor or spine is define by a **pair of nodes**, and as nodes, tensors can have attributes.
- The method G.add_edge(u,v,attr): add a unique tensor between the nodes u, v to the graph G. The optional argument att defines the attributes for the tensor as key/value.

```python
G.add_edge(1,2, weight =4) # tensor linking the nodes 1,2, with a weight of 4.
```

- The method G.add_edges_from(edge_list,attr): add several tensors simultaneously to the graph G  from a list of tensors.

```python
G.add_edges_from([(2,3),(3,4)]) # tensors connecting the node 2 to 3 and 3 to 4.
```

_Note_: It is not necessary to create nodes before connecting it. NetworkX will handle it automatically.

```python
edge_list = [(2, 3), (3, 4), (8, 7), (6, "Daniel"),(print,5), (8, 9), (9, 10), (10, 11), (11, 12), (10, 12), (12, 0)]
G1.add_edges_from(edge_list)

nx.draw(G1, with_labels=True, font_color='black', node_size=500, node_color='#75DFC1') 
plt.show()
```


## Delete a node

- The method G.remove_node(n) and G.remove_nodes_from(node_list) to delete a nodes and a list of nodes in the graph G.

```python
G.remove_node(1)
G.remove_nodes_from([2,3]) # delete the nodes 2 and 3.
```

- The methods G.remove_edge(u,v) and G.remove_edges_from(edge_list) are used to delete a tensor and a list of tensor of the graph G.

```python
G.remove_edge(1,2)
G_remove_edges_from([(2,3),(3,4)]) # delete the tensors (2,3) and (3,4)
```

- The method G.clear() delete all the nodes and tensors of a graph.


_Note_: When we delete a node, we also delete all the tensors from it.





## Accessors and characteristic of a graph

We have created complete grpahs, but sometimes we need to access information of the graph. NetworkX has a list of functions to consult properties of a graph aka accessors. For example, G.nodes or G.edges give us access directly to the nodes and edges and to their attributes.

```python
list(G1.nodes)
G1.nodes[1] # show all the attributes of the node 1
G1.nodes(data=True) # show the attributes of all nodes.
```

We can have access to the structure of the graph using 

- G.number_of_nodes()
- G.number_of_edges()


```python
list(G1.nodes)
list(G1.edges)


G1.nodes(data=True)
G1.edges(data=True)

>>> EdgeDataView([(0, 1, {}), (0, 12, {}), (1, 2, {}), (2, 3, {}), (4, 3, {}), (7, 8, {}), (10, 9, {}), (10, 11, {}), (10, 12, {}), (11, 12, {}), (8, 9, {})])
```

We can see here the list of edges with their attributes {}.

_Example 2_:

```python

# Récupère la liste des noeuds, attributs et arêtes
print("Les noeuds de G1 sont : ", list(G1.nodes(data=True)))
print("Les arêtes de G1 sont : ",list(G1.edges))

>>>
Les noeuds de G1 sont :  [(0, {}), (1, {}), (2, {}), (4, {}), (5, {}), (6, {}), (7, {'weight': '2'}), (10, {'color': 'blue'}), (11, {'color': 'blue'}), (12, {'color': 'blue'}), (3, {}), (8, {}), (9, {})]
Les arêtes de G1 sont :  [(0, 1), (0, 12), (1, 2), (2, 3), (4, 3), (7, 8), (10, 9), (10, 11), (10, 12), (11, 12), (8, 9)]

```
We can see here some attributes.


## Degrees of a node

Degree of a node is the number of times this nodes is touched by edges. In other words, this means the number of neighboors direclty connected to the node. If a node is connected to 5 others, thie degree is 5.

To obtaint the degree of a specific node s in a graph G we use the method **degree** as **G.degree[s]**

In a non oriented graph, the degree of a node is simply the number of edges touching it, whereas in an oriented grpah, we distinguish between in_degree(edges going **to** this node) and out_degree(edges from this node) -> deg(u)= deg(u)+ + deg(u)-

```python
G.in_degree[s]
G.out_degree[s]
G1.degree[12] # degree of the node 12. Recall G1 is non oriented Graph

>>>3

```

## Visualisation des graphs

The visualisation is super important because we can understand quickly the strucutre and relationship inside of the graph. 
**nx.draw_networkx()** to represent nodes and edges.

We have plenty of options to play with:
- pos: layout function to determine distances and positions of the nodes following a specific shape like
-- nx.circular_layout: nodes are placed in a circle shape
-- nx.spring_layout: default
-- nx.shell_layout: nodes a placed as a shell shape.
-- nx.random_layout; nodes are randomly placed
-- nx.planar_layout: nodes are placed without any intersections

- node.size, node.color: control the size and the color of the nodes
- edge_color: Define the color of the edges.
- with_labels: if True, nodes are labeled with their labels.
- font_size, font_color: custom the size and the color of the label of the nodes.


```python
  G = nx.path_graph(5)
pos= nx.spring_layout(G, k = 0.5)
nx.draw_networkx(G, pos, with_labels =True, node_color = 'skyblue')

  ```


```python
# Liste des layouts à utiliser
layouts = {
    "Circular Layout": nx.circular_layout,
    "Spring Layout": nx.spring_layout,
    "Shell Layout": nx.shell_layout,
    "Random Layout": nx.random_layout,
    "Planar Layout": nx.planar_layout
}

# Paramètres d'affichage du graphe G
options = {
    'node_color': 'skyblue',
    'edge_color': 'gray',
    'node_size': 500,
    'width': 2,
    'with_labels': True,
    'font_weight': 'bold'
}

# Affichage des graphes avec différents layouts
plt.figure(figsize=(15, 10))

for i, (name, layout) in enumerate(layouts.items(), start=1):
    plt.subplot(2, 3, i)
    nx.draw(G1, pos=layout(G1), **options)
    plt.title(name)

plt.tight_layout()
plt.show()
```


![Layout Graphs](layout_graphs.png)

