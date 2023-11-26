# Graph Partitioning Optimization
In the realm of modern computational sciences and engineering, the complexity of problem-solving often exceeds the capacities of traditional algorithms. Graph partitioning, a fundamental challenge in various domains, lies at the heart of numerous optimization problems.

This project delves into the intricate world of graph partitioning optimization, leveraging the power of multiobjective optimization through NSGA-II (Non-dominated Sorting Genetic Algorithm II). NSGA-II stands as a beacon in the pursuit of balancing conflicting objectives, offering a versatile and robust approach to exploring the trade-offs within complex problem landscapes.

The algorithm provided in this repository, will accept data in 2D plane $(x,y)$ and generate a connected graph from the data where the weight of the sides is the Euclidean distance between two points. So the algorithm is not limited to Graph as the input. 

## Optimized Objective Function
Define a connected graph $G = (V, E)$ where, 
* $V = {v_1, v_2, v_3, ..., v_n}$
* $E = {e_{ij} i,j \in {1,2,3, ..., n}, i\neq j }$
  
There are 3 objective functions that are gonna be optimized on this Graph Partitioning Problem,
### 1. Minimizing the Edge Cut in the Graph
$$\text{min} f_1 = \Sigma_{e_{ij} \in E} \ w_{ij}\cdot X_{ij}$$
where $w_{ij}$ is the weight of edge $e_{ij}$ and $X_{ij}$ define if $v_i$ and $v_j$ is in the same partition. With $Z = {z_1, z_2, z_3, ..., z_p}$ defined the partitions in the graph, then

$$
X_{ij} = \begin{cases}
              1 \ \ \text{$v_i \in Z_x, v_j \in Z_y; Z_x, Z_y \in Z; x\neq y$} \\
              0 \ \ \text{otherwise}
          \end{cases} 
$$

### 2. Minimizing the Difference of Each Point in Partition
$$\text{min} f_2 = \Sigma_{p=1}^{Z-1} \Sigma^{Z}_{q=p+1} |n'_p - n'_q|$$
where $n'_p$ and $n'_q$ are the number of vertex in partition $Z_p$ and $Z_q$. This objective function makes sure that all of the partitions have a balanced number of nodes inside of the partition.

### 3. Minimizing the Spread of Each Partition
$$\text{min} f_3 = \Sigma_{p=1}^Z \delta_p$$
where $\delta_p = \text{max}(x_p^U - x_p^L, y_p^U - y_p^L)$, with (x^U_p, y^U_p) is the upper bound of point in partition $p$ and (x^L_p, y^L_p) is the lower bound of point in partition $p$. This objective function makes sure that each partition is compact in size.

## Interactive Method 
The implementation of the interactive optimization method in this repository is gonna be using the NAUTILUS Navigator developed by DESDEO which is accessible here https://desdeo-mcdm.readthedocs.io/en/latest/notebooks/nautilus_navigator.html 

## How to Operate
1. Run to install all needed dependencies
```
pip install requirements.txt
```
2. Open `Graph Partitioning.ipnyb`
3. Change the input as needed and then run each of the cells
4. Enjoy the output!

For more information about the functions used, you can access the `optimization.py` file and for all the objects used, you can access the `graph.py` file.

If you have any more information about the code you can reach out to me via my LinkedIn here https://www.linkedin.com/in/felix-fern/

## Reference Used
[1] Datta, Dilip & Figueira, José & Fonseca, Carlos & Pereira, Fernando. (2008). Graph partitioning through a multi-objective evolutionary algorithm: A preliminary study. GECCO'08: Proceedings of the 10th Annual Conference on Genetic and Evolutionary Computation 2008. 625-632. 10.1145/1389095.1389222. 


