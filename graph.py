import numpy as np
import networkx as nx


class Point:
    def __init__(self, id, x, y, partition):
        self.id = id
        self.partition = partition
        self.x = x
        self.y = y

    def get_pos(self):
        return self.x, self.y


class Graph:
    def __init__(self, points={}, weight=np.array([[]])):
        self.points = points
        self.weight = weight
        self.all_weight = weight

    def get_points(self):
        return self.points

    def get_partition_points(self, partition):
        temp = {}
        for i in self.points:
            if self.points[i].get('partition') == partition:
                temp[i] = self.points[i]

        return temp

    def get_partition(self):
        partition_result = np.array([])
        for i in self.points:
            partition_result = np.append(
                partition_result, self.points[i].get('partition'))

        return partition_result

    def get_unique_partition(self):
        return list(dict.fromkeys(self.get_partition()))

    def get_points_coords(self):
        points_x = np.array([])
        points_y = np.array([])

        for i in self.points:
            points_x = np.append(points_x, self.points[i].get('x'))
            points_y = np.append(points_y, self.points[i].get('y'))

        return points_x, points_y

    def get_partition_points_coords(self, partition):
        points_x = np.array([])
        points_y = np.array([])

        for i in self.get_partition_points(partition):
            points_x = np.append(points_x, self.points[i].get('x'))
            points_y = np.append(points_y, self.points[i].get('y'))

        return points_x, points_y

    def get_weight(self):
        return self.weight

    def get_spread(self):
        points_x, points_y = self.get_points_coords()
        return max(max(points_x) - min(points_y), max(points_y) - min(points_y))

    def get_partition_spread(self, partition):
        points_x, points_y = self.get_partition_points_coords(partition)
        if (len(points_x) == 0):
            return 0
        return max(max(points_x) - min(points_y), max(points_y) - min(points_y))

    def get_edge_cut(self):
        edge_cut = {}
        size = len(self.points)
        for i in range(size):
            for j in range(i, size):
                if (self.weight[j, i] != 0 and self.points[i].get('partition') != self.points[j].get('partition')):
                    edge_cut[(j, i)] = self.weight[j, i]
        return edge_cut

    def get_partition_size(self, partition):
        points_x, _ = self.get_partition_points_coords(partition)
        return len(points_x)

    def is_connected(self):
        if (self.weight.size == 0):
            return False

        num_vertices = len(self.points)

        def dfs(vertex, visited):
            visited[vertex] = True
            for neighbor in range(num_vertices):
                if self.weight[vertex, neighbor] != 0 and not visited[neighbor]:
                    dfs(neighbor, visited)

        visited = [False] * num_vertices

        dfs(0, visited)

        return all(visited)

    def is_partition_connected(self, partition):
        if (self.weight.size == 0):
            return False

        num_vertices = len(self.get_partition_points(partition))

        def dfs(vertex, visited, idx=-1):
            visited[idx] = True
            for i, neighbor in enumerate(self.get_partition_points(partition)):
                if self.weight[vertex, neighbor] != 0 and not visited[i]:
                    dfs(neighbor, visited, i)

        visited = [False] * num_vertices
        dfs([v for v in self.get_partition_points(partition)][0], visited, 0)

        return all(visited)

    def is_all_partition_connected(self):
        connected = [True] * len(self.get_unique_partition())
        for i in self.get_unique_partition():
            connected[int(i) - 1] = self.is_partition_connected(i)

        return all(connected)

    def calculate_weight(self, min_s, max_s, debug=False):
        iteration = 0
        points_x, points_y = self.get_points_coords()
        size = len(points_x)
        temp = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                temp[i, j] = np.sqrt(
                    np.abs(points_x[i] - points_x[j]) ** 2 + np.abs(points_y[i] - points_y[j]) ** 2)

        self.all_weight = np.copy(temp)
        self.weight = np.zeros([size, size])
        while (not self.is_connected()):
            iteration += 1
            if (min_s != 0 and max_s != 0):
                new_weight = np.zeros([size, size])

                for j in range(size):
                    n_edge_rand = np.random.randint(min_s, max_s)
                    idx = np.argpartition(self.all_weight[j, :], n_edge_rand)
                    for k in idx[1:n_edge_rand + 1]:
                        new_weight[j, k] = self.all_weight[j, k]
                    new_weight[:, j] = new_weight[j, :]

            self.weight = np.copy(new_weight)

            if (iteration % 100 == 0):
                if (min_s < max_s - 1):
                    min_s += 1
                else:
                    max_s += 1
        if (debug):
            print('Iterations: ', iteration,
                  'Min Side:', min_s, 'Max Side:', max_s)
        return self.weight

    def update_partition(self, id, partition):
        self.points[id]['partition'] = partition

    def add_point(self, point):
        self.points[point.id] = {'x': point.x,
                                 'y': point.y, 'partition': point.partition}

    def remove_point(self, id):
        try:
            self.points.pop(id)
        except:
            print('point not in partition')

    def mutate_graph(self, debug=False):
        G = nx.Graph(self.get_weight())
        while True:
            random = np.random.randint(0, len(self.get_weight()))
            neighbor = [n for n in G.neighbors(random)]
            random_neighbor = np.random.randint(0, len(neighbor))

            if (self.points[neighbor[random_neighbor]].get("partition") != self.points[random].get("partition")):
                prev_partition = self.points[random].get("partition")
                self.update_partition(random, self.points[(
                    neighbor[random_neighbor])].get("partition"))
                if (len(self.get_partition_points(prev_partition)) > 0):
                    if (self.is_partition_connected(prev_partition)):
                        new_partition = self.points[(
                            neighbor[random_neighbor])].get("partition")
                        if (debug):
                            print(
                                f"Mutate Nodes {random} from {prev_partition} to {new_partition}")
                        self.update_partition(random, new_partition)
                        break
                else:
                    self.update_partition(random, prev_partition)
