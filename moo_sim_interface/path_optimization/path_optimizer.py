import igraph as ig
import numpy as np
import pandas as pd
from pymoo.vendor.hv import HyperVolume

from path_visualization import create_3d_plot, plot_pareto_optimal_paths, plot_directed_graph, construct_path_in_design_space
from path_utils import calculate_pareto_points

REF_POINT: list = []  # use a global reference point for hypervolume calculation
HYPERVOLUME: int = 0  # store the hypervolume of the Pareto front as a global variable


def calculate_shortest_sequence_constraints(data, start, end, design_names, target_names):
    """
    Calculate the shortest path in a directed graph with sequence constraints.
    :param data: a pandas DataFrame containing the design space and target space data
    :param start: the index of the start point in the DataFrame
    :param end: the index of the end point in the DataFrame
    :param design_names: a list of column names representing the design space
    :param target_names: a list of column names representing the target space
    :return: a list of indices representing the shortest path from start to end
    """
    raw_data_df = data
    all_points = raw_data_df[target_names].to_numpy()

    pareto_points, non_pareto_points = calculate_pareto_points(all_points)

    # Determine which points are Pareto optimal
    is_pareto_optimal = np.isin(all_points, pareto_points).all(axis=1)

    optimal_data_df = raw_data_df[is_pareto_optimal]

    print(f'Selected start point: {optimal_data_df.iloc[start]}')
    print(f'Selected end point: {optimal_data_df.iloc[end]}')

    # Create an empty graph with nodes only
    num_pareto_nodes = len(pareto_points)
    complete_graph = ig.Graph(directed=True)
    complete_graph.add_vertices(num_pareto_nodes)

    # Add edges with weights (Euclidean distance or infinity)
    for i in range(num_pareto_nodes):
        for j in range(num_pareto_nodes):
            if i != j:
                point1 = optimal_data_df.iloc[i][design_names].to_numpy()
                point2 = optimal_data_df.iloc[j][design_names].to_numpy()
                if np.any(point1 > point2):  # apply the "only increments" constraint
                    continue
                else:
                    distance = np.linalg.norm(point1 - point2)
                    complete_graph.add_edge(i, j, weight=distance)

    # Find the shortest path
    shortest_path = complete_graph.get_shortest_paths(start, to=end, weights='weight', output="vpath")[0]
    if not shortest_path:
        print('Could not find a path between the selected nodes. Check start and end nodes.')
    return shortest_path


def calculate_hypervolume(data, target_names):
    """
    Calculate the hypervolume of the Pareto front from the target space of the data.
    :param data: input DataFrame containing the design and target space data
    :param target_names: list of column names representing the target space
    :return:
    """
    target_space = data[target_names].to_numpy()

    pareto_points, _ = calculate_pareto_points(target_space)

    # max hypervolume of pareto points using pymoo:
    hv = HyperVolume(REF_POINT)
    global HYPERVOLUME
    HYPERVOLUME = hv.compute(pareto_points)

    print(f'Hypervolume of Pareto front: {HYPERVOLUME:<14.4f} reference point: {REF_POINT}')


def calculate_all_paths_under_constraint(data, start, end, data_limit, target_length, plot_graph, design_names,
                                         target_names):
    """
    Calculate all paths from start to end in the target space, considering the constraint "only increments" in the design space.
    :param data: the DataFrame containing the design space and target space data
    :param start: the index of the start point in the DataFrame
    :param end: the index of the end point in the DataFrame
    :param data_limit: can be used to limit the number of data points to consider, if None, all data points are used
    :param target_length: the target length of the paths, if 0, this looks at the shortest paths only
    :param plot_graph: create a plot of the resulting graph, only useful for small datasets (< 1000 points)
    :param design_names: a list of column names representing the design space
    :param target_names: a list of column names representing the target space
    :return: a list of all paths with their hypervolume difference and length difference
    """
    raw_data_df = data.iloc[:data_limit] if data_limit else data

    # print the full dataframe:
    # with pd.option_context('display.max_rows', sys.maxsize, 'display.max_columns', sys.maxsize):
    #     print(raw_data_df)

    design_space = raw_data_df[design_names].to_numpy()

    target_space = raw_data_df[target_names].to_numpy()

    pareto_points, non_pareto_points = calculate_pareto_points(target_space)

    # create the graph based on all points
    directed_graph = ig.Graph(directed=True)
    directed_graph.add_vertices(len(target_space))

    # Prepare edges and weights
    edges = []
    weights = []
    for i in range(len(target_space)):
        for j in range(len(target_space)):
            if i != j:
                point1 = design_space[i]
                point2 = design_space[j]
                if np.any(point1 > point2):
                    continue
                else:
                    distance = np.linalg.norm(point1 - point2)
                    edges.append((i, j))
                    weights.append(distance)

    # Add edges and weights to the graph
    directed_graph.add_edges(edges)
    directed_graph.es['weight'] = weights
    print(f'Number of edges: {len(directed_graph.es)}')

    # overwrite 0 weights
    directed_graph.es.select(weight=0)["weight"] = 1e-15

    # find the shortest path between the start and end nodes
    print(f'Start: {design_space[start]}')
    print(f'End: {design_space[end]}')

    # Find all paths from start to end
    # s = set(directed_graph.subcomponent(v=start, mode="out"))
    # t = set(directed_graph.subcomponent(v=end, mode="in"))
    # solution_set = s.intersection(t)
    # print(f'Number of paths from start to end: {len(solution_set)}')
    # print(f'Paths from start to end consist of: {solution_set}')

    all_paths = paths_from_to(directed_graph, start, end)

    if plot_graph:
        plot_directed_graph(directed_graph, [all_paths[0]])

    hv = HyperVolume(REF_POINT)
    # Evaluate each path
    path_evaluations = []
    for path in all_paths:
        path_points = target_space[path]
        path_length = len(path) - 1
        path_hv = hv.compute(path_points)
        hv_diff = HYPERVOLUME - path_hv
        length_diff = abs(path_length - target_length)
        path_evaluations.append((path, hv_diff, length_diff))

    return path_evaluations


def adj_list_find_paths(a, n, m, path=[]):
    """Find paths from node index n to m using adjacency list a."""
    path = path + [n]

    if n == m:
        return [path]
    # if len(path) > 14:
    #     return []
    paths = []

    for child in a[n]:
        if child not in path:
            child_paths = adj_list_find_paths(a, child, m, path)
            for child_path in child_paths:
                paths.append(child_path)
    # else:
    #     return []
    return paths


def paths_from_to(graph, source, dest):
    """Find all paths in graph from vertex source to vertex dest."""
    a = graph.get_adjlist()
    n = source
    m = dest
    return adj_list_find_paths(a, n, m)


def adjust_data(data, start, end, design_names):
    """
    Pre-filter the data according to the constraint "only increments" in the design space.
    :param data: a pandas DataFrame containing the design space and target space data
    :param start: the index of the start point in the DataFrame
    :param end: the index of the end point in the DataFrame
    :param design_names: a list of column names representing the design space
    :return: a filtered DataFrame with only the data points that are between the start and end points in the design space,
                and the new indices of the start and end points in the filtered DataFrame.
    """
    design_space = data[design_names].to_numpy()
    start_point = design_space[start]
    end_point = design_space[end]

    # Filter data where each point in the design space is greater than or equal to the start point
    # and less than or equal to the end point in all dimensions
    mask = np.all(design_space >= start_point, axis=1) & np.all(design_space <= end_point, axis=1)
    filtered_data = data[mask]
    filtered_data.reset_index(drop=True, inplace=True)
    # also return the new index of the start and end points
    new_start_index = \
        filtered_data[(filtered_data[design_names] == start_point).all(axis=1)].index[
            0]
    new_end_index = \
        filtered_data[(filtered_data[design_names] == end_point).all(axis=1)].index[0]
    print(f'Returned {len(filtered_data)} data points between the old start and end points: {start} and {end}')

    return filtered_data, new_start_index, new_end_index


def filter_paths(paths):
    """
    After collecting all paths "brute-force" from start to end, filter the paths by their length and delta-hypervolume.
    :param paths: list of tuples, where each tuple contains a path (list of indices) and its delta-hypervolume
    :return: the best paths for each length, sorted by their length.
    """
    paths_by_length = {}
    for path in paths:
        path_length = len(path[0])
        if path_length not in paths_by_length:
            paths_by_length[path_length] = []
        paths_by_length[path_length].append(path)

    # Sort the paths within each length group by delta-hypervolume
    best_paths = []
    for length in paths_by_length:
        sorted_paths = sorted(paths_by_length[length], key=lambda x: x[1])
        best_paths.extend(sorted_paths[:1])
    best_paths = sorted(best_paths, key=lambda x: len(x[0]))  # sort the best paths for each length by their length
    return best_paths


def experiment_wrapper(choice, start_node, end_node, target_length, data_limit=None,
                       plot_hyperspace=False,
                       plot_graph=False, print_all_paths=False, plot_pareto_paths=False, filter_best_paths=False,
                       print_best_paths=False, extra_plot_items_thresh=20, re_transform_into_design_space=False,
                       increments_visualization=False):
    """
    Run the path optimization experiment with the given parameters.
    :param choice: load the data set by this choice, e.g. 'graz_oc_oa', 'axel_capex_co2', 'graz_plus_ma'
    :param start_node: the index of the start point in the DataFrame
    :param end_node: the index of the end point in the DataFrame
    :param target_length: the target length of the paths, if 0, this optimizes for shortest path
    :param data_limit: can be used to limit the number of data points to consider, if None, all data points are used
    :param plot_hyperspace: plot the hyperspace, even if start and end points are not in the data
    :param plot_graph: plot the directed graph of the paths, only useful for small datasets (< 1000 points)
    :param print_all_paths: print the details of all paths found to console
    :param plot_pareto_paths: plot the optimal paths in the target space and in the metrics space
    :param filter_best_paths: consider only the best paths for each length, for plotting and printing
    :param print_best_paths: print the best paths to console, sorted by their delta-hypervolume
    :param extra_plot_items_thresh: if the number of items in the plot exceeds this threshold, do not plot them
    :param re_transform_into_design_space: if True, re-transform the paths into the design space for visualization
    :param increments_visualization: if True, visualize the increments in the design space as polar plots
    """
    raw_data, d_design, d_target, d_units = load_data(choice)

    design_names = raw_data.columns[:d_design]
    target_names = raw_data.columns[-d_target:]

    # are_all_simple_paths_monotonic(raw_data, d_design, d_target)

    calculate_hypervolume(raw_data, target_names)

    if plot_hyperspace:  # plot the hyperspace, even if start and end points are not in the data
        create_3d_plot(raw_data, start_node, end_node, d_design, d_target, d_units)

    # shortest_path = calculate_shortest_sequence_constraints(raw_data, start_node, end_node, design_names)
    # print(f'Shortest path from node {start_node} to node {end_node}: {shortest_path}')

    # stop here, if the start and end nodes are not in the data
    if start_node >= len(raw_data) or end_node >= len(raw_data):
        raise ValueError(f'Select proper start and end nodes for {len(raw_data)} data points')
    data, new_start, new_end = adjust_data(raw_data, start_node, end_node, design_names)

    moo_paths = calculate_all_paths_under_constraint(data, new_start, new_end, data_limit, target_length, plot_graph,
                                                     design_names,
                                                     target_names)
    print(f'MOO Paths from new node {new_start} to node {new_end}: {len(moo_paths)}')

    if print_all_paths:
        for i in range(0, len(moo_paths), 10):
            print(f'{"  ".join([str(x[0]) for x in moo_paths[i:i + 10]])}')

    # validate paths:
    # for path in moo_paths:
    #     nodes = path[0]
    #     for i in range(len(nodes) - 1):
    #         p1 = data.iloc[nodes[i]][['Wind_P_el_n', 'pV_P_inst', 'storage_E_max']].to_numpy()
    #         p2 = data.iloc[nodes[i + 1]][['Wind_P_el_n', 'pV_P_inst', 'storage_E_max']].to_numpy()
    #         if np.any(p1 > p2):
    #             print(f'Path {nodes} is invalid')
    #             break

    if filter_best_paths:
        moo_paths = filter_paths(moo_paths)

    if print_best_paths:
        for idx, path in enumerate(moo_paths):
            print(f'Path: {idx}, Delta-HV: {path[1]}, Path: {path[0]}')

    if plot_pareto_paths:
        plot_pareto_optimal_paths(data, moo_paths, extra_plot_items_thresh, design_names, target_names,
                                  hv=f'{HYPERVOLUME:.6f}', ref_point=str(REF_POINT), target_len=str(target_length))

    if re_transform_into_design_space:
        construct_path_in_design_space(data, moo_paths, design_names, d_units, increments_visualization)


def load_data(choice):
    """
    Load the data set based on the choice parameter.
    :param choice: which data set to load, e.g. 'graz_oc_oa', 'axel_capex_co2', 'graz_plus_ma'
    :return: the raw data as a pandas DataFrame, the number of design space dimensions,
    the number of target space dimensions, and the units of the design space data.
    """
    global REF_POINT
    if choice == 'axel_capex_co2':
        raw_data = pd.read_csv('resources/Optimierung_V3_1_3_CO2_Capex.csv', sep='\t')
        leistung_ausbau_p_inst_pv = 10000000  # [Wp]
        carport = 500000  # [Wp]
        pt_h_p_inst_max = 1000000  # [Wp]
        PtH_pinst_min = 5000  # [Wp]
        HeatStorage_Cap_max = 1000000 * 1000 * 4819  # [J/K]
        HeatStorage_Cap_min = 1000 * 1000 * 4819  # [J/K]
        mod_battery_system_Cap_max = 983600  # [Wh]
        mod_battery_system_Cap_min = 16400  # [Wh]

        raw_data.rename(
            {'pVPlantFactorExtetion.P_inst': 'pVPlant_P_inst', 'powerToHeat.max_power': 'powerToHeat_P_inst',
             'heatStorage.C': 'heatStorage_C', 'mod_battery_system.StorageCapacity': 'battery_C'},
            axis=1, inplace=True)
        raw_data['pVPlant_P_inst'] = raw_data['pVPlant_P_inst'] * leistung_ausbau_p_inst_pv + carport
        raw_data['powerToHeat_P_inst'] = raw_data['powerToHeat_P_inst'] * pt_h_p_inst_max + PtH_pinst_min
        raw_data['heatStorage_C'] = raw_data['heatStorage_C'] * HeatStorage_Cap_max + HeatStorage_Cap_min
        raw_data['battery_C'] = raw_data['battery_C'] * mod_battery_system_Cap_max + mod_battery_system_Cap_min
        d_units = ['W', 'W', 'J', 'W']
        REF_POINT = [2.5e7, 1.13e8]
        return raw_data, 4, 2, d_units
    elif choice == 'graz_oc_oa':
        raw_data = pd.read_csv('resources/2024-01-24_20-35-27_results.csv', sep=';')
        raw_data['Wind_P_el_n'] = raw_data['Wind_P_el_n'].apply(
            lambda x: 1 if x == 5.7e6 else 2 if x == 1.14e7 else 3 if x == 1.71e7 else 0)
        # change order of columns:
        raw_data = raw_data[['pV_P_inst', 'storage_E_max', 'Wind_P_el_n', 'OC', 'OA']]
        raw_data[['OC', 'OA']] = raw_data[['OC', 'OA']].apply(lambda x: 1 - x)
        raw_data.rename({'OC': '1-OC', 'OA': '1-OA'}, axis=1, inplace=True)
        d_units = ['W', 'J', 'D']
        REF_POINT = [0.5, 1]
        return raw_data, 3, 2, d_units
    elif choice == 'graz_plus_ma':
        file = 'resources/2025-04-10_10-52-35_results.csv'
        raw_data = pd.read_csv(file, sep=';')
        d_units = ['W', 'W', 'J', 'J', 'W']
        REF_POINT = [1e8, 1]
        return raw_data, 5, 2, d_units
    else:
        raise ValueError('Invalid choice')


if __name__ == '__main__':
    # Entry point for the path optimization examples.
    # Choose one of the available data sets and set the parameters for the experiment.

    experiment1 = {'choice': 'graz_oc_oa',  # 19708 paths
                   'start_node': 8,
                   'end_node': 713,
                   'target_length': 0,
                   'data_limit': None,
                   'plot_hyperspace': True,
                   'plot_graph': True,
                   'print_all_paths': False,
                   'plot_pareto_paths': True,
                   'filter_best_paths': False,
                   'print_best_paths': False,
                   'extra_plot_items_thresh': 40,
                   're_transform_into_design_space': False,
                   'increments_visualization': False}

    experiment2 = {'choice': 'graz_oc_oa',  # 639804 paths
                   'start_node': 8,
                   'end_node': 847,
                   'target_length': 0,
                   'data_limit': None,
                   'plot_hyperspace': True,
                   'plot_graph': False,
                   'print_all_paths': False,
                   'plot_pareto_paths': True,
                   'filter_best_paths': True,
                   'print_best_paths': True,
                   'extra_plot_items_thresh': 40,
                   're_transform_into_design_space': True,
                   'increments_visualization': False}

    experiment3 = {'choice': 'axel_capex_co2',  # 3 paths
                   'start_node': 25,
                   'end_node': 18,
                   'target_length': 0,
                   'data_limit': None,
                   'plot_hyperspace': True,
                   'plot_graph': False,
                   'print_all_paths': True,
                   'plot_pareto_paths': True,
                   'filter_best_paths': True,
                   'print_best_paths': True,
                   'extra_plot_items_thresh': 40,
                   're_transform_into_design_space': True,
                   'increments_visualization': False}

    experiment4 = {'choice': 'graz_plus_ma',  # 138415 paths
                   'start_node': 1965,
                   'end_node': 4537,
                   # 'start_node': 7,  # values to get the most possible paths, to test the faster algorithm
                   # 'end_node': 4737,
                   'target_length': 0,
                   'data_limit': None,
                   'plot_hyperspace': True,
                   'plot_graph': False,
                   'print_all_paths': False,
                   'plot_pareto_paths': True,
                   'filter_best_paths': True,
                   'print_best_paths': False,
                   'extra_plot_items_thresh': 40,
                   're_transform_into_design_space': True,
                   'increments_visualization': True}

    experiment_wrapper(**experiment4)

    # additional data exploration and visualization methods, to help with the understanding of the data:
    # poi_selector_plot(load_data(experiment4.get('choice')))
    # poi_selector_show_increments(load_data(experiment2.get('choice')), start_idx=8, calc_unions=True)
