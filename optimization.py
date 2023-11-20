import networkx as nx
import numpy as np
from graph import Point, Graph
from copy import deepcopy


def generate_rand_points(graph, n, interval):
    temp = np.random.rand(2, n) * interval[0] + interval[1]
    for i in range(n):
        graph.add_point(Point(i, temp[0, i], temp[1, i], -1))
    return graph


def crossover(a, b, n_partition, iter=0, debug=False):
    a_target = np.random.randint(1, n_partition + 1)
    b_target = np.random.randint(1, n_partition + 1)

    if (debug):
        print("Crossover Target")
        print(f"Population A : Partition {a_target}")
        print(f"Population B : Partition {b_target}")

    temp_a_cross = {}
    temp_b_cross = {}

    population_a_temp = deepcopy(a)
    population_b_temp = deepcopy(b)

    population_a_res = deepcopy(a)
    population_b_res = deepcopy(b)

    for i in population_a_temp.get_partition_points(a_target):
        partition = population_b_res.get_points().get(i).get('partition')
        val = temp_a_cross.get(partition, 0)
        temp_a_cross[partition] = val + 1

    for i in population_b_res.get_partition_points(max(temp_a_cross, key=temp_a_cross.get)):
        population_b_res.update_partition(i, 0)

    for i in population_a_temp.get_partition_points(a_target):
        population_b_res.update_partition(
            i, max(temp_a_cross, key=temp_a_cross.get))

    for i in population_b_temp.get_partition_points(b_target):
        partition = population_a_res.get_points().get(i).get('partition')
        val = temp_b_cross.get(partition, 0)
        temp_b_cross[partition] = val + 1

    for i in population_a_res.get_partition_points(max(temp_b_cross, key=temp_b_cross.get)):
        population_a_res.update_partition(i, 0)

    for i in population_b_temp.get_partition_points(b_target):
        population_a_res.update_partition(
            i, max(temp_a_cross, key=temp_a_cross.get))

    G = nx.Graph(np.array(population_a_res.weight))

    a_check = False
    b_check = False

    while len(population_a_res.get_partition_points(0)) > 0 or len(population_b_res.get_partition_points(0)) > 0:
        population_a_excess = population_a_res.get_partition_points(0)
        population_b_excess = population_b_res.get_partition_points(0)

        count_null_population_a = 0
        for i in population_a_excess:
            partition_id, min_partition = -1, -1
            for j in [n for n in G.neighbors(i)]:
                partition = population_a_res.get_points().get(j).get('partition')
                if (min_partition == -1 and (partition != 0 and partition != max(temp_a_cross, key=temp_a_cross.get))):
                    partition_id, min_partition = partition, len(
                        population_a_res.get_partition_points(partition))
                elif (min_partition > len(population_a_res.get_partition_points(partition)) and (partition != 0 and partition != max(temp_a_cross, key=temp_a_cross.get))):
                    partition_id, min_partition = partition, len(
                        population_a_res.get_partition_points(partition))
            if (a_check):
                population_a_res.update_partition(
                    i, max(temp_a_cross, key=temp_a_cross.get))
            elif (min_partition != -1):
                population_a_res.update_partition(i, partition_id)
            elif (min_partition == -1):
                count_null_population_a += 1

        if (count_null_population_a == len(population_a_excess)):
            a_check = True

        count_null_population_b = 0
        for i in population_b_excess:
            partition_id, min_partition = -1, -1
            for j in [n for n in G.neighbors(i)]:
                partition = population_b_res.get_points().get(j).get('partition')
                if (min_partition == -1 and (partition != 0 and partition != max(temp_b_cross, key=temp_b_cross.get))):
                    partition_id, min_partition = partition, len(
                        population_b_res.get_partition_points(partition))
                elif (min_partition > len(population_b_res.get_partition_points(partition)) and (partition != 0 and partition != max(temp_b_cross, key=temp_b_cross.get))):
                    partition_id, min_partition = partition, len(
                        population_b_res.get_partition_points(partition))
            if (b_check):
                population_b_res.update_partition(
                    i, max(temp_b_cross, key=temp_b_cross.get))
            elif (min_partition != -1):
                population_b_res.update_partition(i, partition_id)
            elif (min_partition == -1):
                count_null_population_b += 1

        if (count_null_population_b == len(population_b_excess)):
            b_check = True

    if (iter >= n_partition ** 3):
        if (debug):
            print("All possibilities tried, but failed")
        return a, b

    if (len(population_a_res.get_unique_partition()) == n_partition and len(population_b_res.get_unique_partition()) == n_partition):
        if (population_a_res.is_all_partition_connected() and population_b_res.is_all_partition_connected()):
            if (debug):
                print("Success")
            return population_a_res, population_b_res
        else:
            if (debug):
                print("Failed, retrying with other partition\n")
            return crossover(a, b, n_partition, iter + 1)
    else:
        if (debug):
            print("Failed, retrying with other partition\n")
        return crossover(a, b, n_partition, iter + 1)


def dominates(solution1, solution2):
    dominates_obj1 = np.all(solution1[1:] <= solution2[1:])
    better_in_at_least_one = np.any(solution1[1:] < solution2[1:])
    return dominates_obj1 and better_in_at_least_one


def non_dominated_sort(population):
    fronts = []
    num_solutions = population.shape[0]
    domination_count = np.zeros(num_solutions, dtype=int)
    dominated_solutions = {i: [] for i in range(num_solutions)}

    for i, solution in enumerate(population):
        for j, other_solution in enumerate(population[i + 1:], start=i + 1):
            if dominates(solution, other_solution):
                domination_count[j] += 1
                dominated_solutions[i].append(j)
            elif dominates(other_solution, solution):
                domination_count[i] += 1
                dominated_solutions[j].append(i)

    front = []
    for i, count in enumerate(domination_count):
        if count == 0:
            front.append(i)

    fronts.append(front)
    current_rank = 1
    while len(fronts[-1]) > 0:
        next_front = []
        for i in fronts[-1]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_rank += 1
        fronts.append(next_front)

    return fronts


def crowding_distance(front, objectives):
    num_objectives = objectives.shape[1]
    num_solutions = len(front)
    distances = np.zeros(num_solutions)

    for obj_index in range(num_objectives):
        sorted_front = sorted(front, key=lambda x: objectives[x][obj_index])
        distances[0] = distances[-1] = np.inf

        if num_solutions > 2:
            min_obj_val = objectives[sorted_front[0]][obj_index]
            max_obj_val = objectives[sorted_front[-1]][obj_index]

            if max_obj_val == min_obj_val:
                continue

            for i in range(1, num_solutions - 1):
                distances[i] += (objectives[sorted_front[i+1]][obj_index] -
                                 objectives[sorted_front[i - 1]][obj_index]) / (max_obj_val - min_obj_val)

    return distances


def random_connected_partitions(graph, adjacency_matrix, num_partitions):
    G = nx.Graph(np.array(adjacency_matrix))
    G_full = nx.Graph(np.array(adjacency_matrix))

    def bfs_limit(graph, source, min_size):
        visited = set()
        queue = [source]

        while queue and len(visited) < min_size:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(neighbor for neighbor in graph.neighbors(
                    node) if neighbor not in visited)

        return visited

    nodes = list(G.nodes())

    partition_size = len(nodes) // num_partitions // 2
    partitions = []
    current_degree = 1

    for i in range(num_partitions):
        degree = {}
        for j in G.nodes():
            temp = degree.get(G.degree(j), np.array([]))
            temp = np.append(temp, int(j))
            np.random.shuffle(temp)
            degree[G.degree(j)] = temp

        while True:
            if (len(degree.get(current_degree, [])) == 0):
                current_degree += 1
            else:
                temp = degree.get(current_degree)
                np.random.shuffle(temp)

                start_node = degree.get(current_degree)[0]
                break

        partition = list(bfs_limit(G, start_node, partition_size))
        G.remove_nodes_from(partition)
        partitions.append(partition)

    while (len(G.nodes())):
        updated_node = []
        for i in G.nodes():
            index, min_partition = -1, -1
            for j, val in enumerate([n for n in G_full.neighbors(i)]):
                for k, val_p in enumerate(partitions):
                    if (val in val_p and min_partition == -1):
                        index = k
                        min_partition = len(val_p)
                    elif (val in val_p and len(val_p) < min_partition):
                        index = k
                        min_partition = len(val_p)
            if (min_partition != -1):
                updated_node.append(i)
                partitions[index].append(i)
        G.remove_nodes_from(updated_node)

    partition_vector = np.zeros(len(nodes), dtype=int)
    for i, partition in enumerate(partitions):
        partition_vector[[int(j) for j in partition]] = i + 1

    for i, val in enumerate(partition_vector):
        graph.update_partition(i, val)
