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

![Type of Graphs](Oriented_NonOriented_graphs.png)
