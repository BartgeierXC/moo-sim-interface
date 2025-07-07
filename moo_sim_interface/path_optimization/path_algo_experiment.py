import numpy as np
from pymoo.vendor.hv import HyperVolume

from path_optimizer import adjust_data, load_data

ref_point = None
HV = None


class Node:
    def __init__(self, parents, pos):
        self.pos = pos
        self.parents = parents
        self.hv = np.prod(ref_point - pos)
        self.hv_rel_parents = []
        self.best_parent_paths = None
        self.max_distance_start = None

    def __str__(self):
        return f"Node: {self.pos}, parents: {str([p.pos for p in self.parents]):<80}, hv_rel_parents: {str(self.hv_rel_parents):<30}, max_distance_start: {self.max_distance_start}, hv: {self.hv}"

    def get_max_distance(self):
        if self.max_distance_start is None:
            return self.calc_max_distance_start()
        else:
            return self.max_distance_start

    def calc_max_distance_start(self):
        # this works recursively, start at the target (last node) and ask every parent node for its max distance to start
        if len(self.parents) > 0:
            self.max_distance_start = max([p.get_max_distance() for p in self.parents]) + 1
        else:
            self.max_distance_start = 0
        return self.max_distance_start

    def init_best_paths(self):
        if self.max_distance_start == 0:
            self.best_parent_paths = {}
        else:
            self.best_parent_paths = {}
            for n in range(self.max_distance_start, 0, -1):
                # the best paths don't change, can be stored in a dict with n as key
                best_parent_paths = [(node.get_best_path([], n - 1), rel_hv) for node, rel_hv in
                                     zip(self.parents, self.hv_rel_parents) if node.get_max_distance() >= n - 1]

                best_path_ret, best_rel_hv = max(best_parent_paths, key=lambda x: x[0][1])
                self.best_parent_paths[n] = (best_path_ret[0], best_path_ret[1] + best_rel_hv)

    def get_best_path(self, paths, n):
        paths = paths.copy()
        paths.append(self)
        if self.best_parent_paths is None:
            self.init_best_paths()
        if n == 0:
            if len(self.parents) == 0:
                return paths, self.hv
            else:
                return None, 0
        best_parent_path = self.best_parent_paths[n]
        return [self] + best_parent_path[0], best_parent_path[1]

    # Old version without memoization, kept for reference:

    # def get_best_path(self, paths, n):
    #     """Old version without memoization"""
    #     paths = paths.copy()
    #     paths.append(self)
    #     if n == 0:
    #         if len(self.parents) == 0:
    #             return paths, self.hv
    #         else:
    #             return None, 0
    #     best_parent_paths = [(node.get_best_path(paths, n - 1), rel_hv) for node, rel_hv in
    #                          zip(self.parents, self.hv_rel_parents) if node.get_max_distance() >= n - 1]
    #
    #     best_path_ret, best_rel_hv = max(best_parent_paths, key=lambda x: x[0][1])
    #     return best_path_ret[0], best_path_ret[1] + best_rel_hv


if __name__ == "__main__":
    # this is a simple example to test the algorithm, you can comment it out when using real data

    # simple example:
    # data_points = np.array([[1, 5], [2, 3], [4, 4], [5, 2], [6, 3], [7, 1]])
    # print(f'Shape of example data: {data_points.shape}')

    # vertex_matrix = np.array([[0, 1, 1, 1, 1, 1],
    #                           [0, 0, 0, 1, 0, 1],
    #                           [0, 0, 0, 1, 1, 1],
    #                           [0, 0, 0, 0, 0, 1],
    #                           [0, 0, 0, 0, 0, 1],
    #                           [0, 0, 0, 0, 0, 0],
    #                           ])
    # print(f'Shape of example vertex matrix: {vertex_matrix.shape}')

    # ref_point = np.array([10, 10])
    # HV = HyperVolume(ref_point)

    # load data from file and create the data_points and vertex_matrix:

    raw_data, d, t, _ = load_data('graz_plus_ma')
    ref_point = [1e8, 1]

    # raw_data, d, t, _ = load_data('graz_oc_oa')
    # ref_point = [0.5, 1]

    HV = HyperVolume(ref_point)

    data, adj_start_idx, adj_target_idx = adjust_data(raw_data, 1965, 4537, raw_data.columns[:d])
    # data, adj_start_idx, adj_target_idx = adjust_data(raw_data, 8, 784, raw_data.columns[:d])
    print(f'New start index: {adj_start_idx}, new target index: {adj_target_idx}')
    data_points = data.iloc[:, -t:].to_numpy()
    print(f'Shape of data points: {data_points.shape}')

    # Erstelle die vertex_matrix mit Vektoroperationen
    conditions = np.all(data.iloc[:, :d].to_numpy()[:, None, :] <= data.iloc[:, :d].to_numpy()[None, :, :], axis=2)
    np.fill_diagonal(conditions, 0)  # Setze die Diagonale auf 0, da ein Punkt sich selbst nicht erreichen kann
    vertex_matrix = conditions.astype(int)
    print(f'Shape of vertex matrix: {vertex_matrix.shape}')

    edges = []

    # init all the edges:
    for i, row in enumerate(vertex_matrix.T):
        edges.append(Node(row, data_points[i]))

    print(f'Number of edges: {len(edges)}')

    # init all the parents:
    for i, node in enumerate(edges):
        node.parents = [edges[j] for j, val in enumerate(node.parents) if val == 1]

    # init the rel hv improvement to parents:
    for i, node in enumerate(edges):
        node.hv_rel_parents = [HV.compute(np.array([node.pos, parent.pos])) - parent.hv for parent in node.parents]

    # find the start node, it is the index of the vertex matrix where the sum of the column is 0
    zero_columns = np.where(np.sum(vertex_matrix, axis=0) == 0)
    print(f'Zero columns: {zero_columns}')
    start_node_index = zero_columns[0][0]

    # print a selection of the vertex_matrix of size 7x7 around the start node
    print(vertex_matrix[start_node_index-3:start_node_index+4, start_node_index-3:start_node_index+4])

    # find the target node, it is the index of the vertex matrix where the sum of the column is the length of the data - 1
    target_node_index = np.where(np.sum(vertex_matrix, axis=0) == len(vertex_matrix) - 1)[0][0]

    # init the max distance to start
    # this works recursively, start at the target (last node) and ask every parent node for its max distance to start
    longest_path_len = edges[target_node_index].get_max_distance()
    print(f'First node: {edges[start_node_index]}')
    print(f'Last node index: {target_node_index}, parents: {len(edges[target_node_index].parents)}')
    print(f'Max distance: {longest_path_len}')

    # walk through the edges recursively and find the longest path, start with a capacity of n and ask every parent for its best path with n - 1
    for t_length in range(1, longest_path_len + 1):
        best_path, best_hv = edges[target_node_index].get_best_path([], t_length)
        with np.printoptions(precision=2, suppress=True):
            sequence = list(reversed([path.pos.tolist() for path in best_path]))
            print(f"Best path length {t_length}: {[(round(seq[0], 2), round(seq[1], 2)) for seq in sequence]}, hv: {best_hv}")
