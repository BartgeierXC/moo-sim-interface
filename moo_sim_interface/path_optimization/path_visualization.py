import math
from itertools import cycle

import igraph as ig
import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from path_utils import calculate_pareto_points

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


def create_3d_plot(data, start, end, d_design, d_target, d_units):
    raw_data_df = data

    target_names = raw_data_df.columns[-d_target:]
    all_points = raw_data_df[target_names].to_numpy()
    pareto_points, non_pareto_points = calculate_pareto_points(all_points)

    # Create the figure and subplots
    fig = plt.figure(figsize=(16, 9), dpi=256)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # Determine which points are Pareto optimal
    is_pareto_optimal = np.isin(all_points, pareto_points).all(axis=1)

    # First plot (3D)
    if 2 < d_design < 6:
        scatter_plot_design_names = raw_data_df.columns[:3]
        scatter_plot_color_names = raw_data_df.columns[d_design - 1]
    else:
        raise ValueError("The design space must have 3, 4 or 5 dimensions")

    sc = ax1.scatter(*[raw_data_df[name] for name in scatter_plot_design_names],
                     s=2, c=raw_data_df[scatter_plot_color_names], cmap='viridis')
    ax1.scatter(*[raw_data_df[name][is_pareto_optimal] for name in scatter_plot_design_names],
                c=raw_data_df[scatter_plot_color_names][is_pareto_optimal],
                cmap='viridis', marker='^')
    ax1.set_xlabel(scatter_plot_design_names[0])
    ax1.set_ylabel(scatter_plot_design_names[1])
    ax1.set_zlabel(scatter_plot_design_names[2])
    ax1.set_title('Plot 1: ' + ' vs. '.join(scatter_plot_design_names))
    cbar = fig.colorbar(sc, ax=ax1, label=None if d_design == 3 else scatter_plot_color_names)
    if d_units[d_design - 1] == 'D':
        # get the discrete values from the dataframe:
        discrete_tick_values = sorted(raw_data_df[scatter_plot_color_names].unique())
        ax1.set_zticks(discrete_tick_values)
        cbar.set_ticks(discrete_tick_values)

    # Second plot
    pareto_colors = raw_data_df[scatter_plot_color_names][is_pareto_optimal]
    non_pareto_colors = raw_data_df[scatter_plot_color_names][~is_pareto_optimal]
    ax2.scatter(pareto_points[:, 0], pareto_points[:, 1], c=pareto_colors, cmap='viridis', marker='^',
                label='Pareto points')
    ax2.scatter(non_pareto_points[:, 0], non_pareto_points[:, 1], s=3, c=non_pareto_colors, cmap='viridis',
                label='Non-Pareto points')
    ax2.set_xlabel(target_names[0])
    ax2.set_ylabel(target_names[1])
    ax2.set_title('Plot 2: Pareto and Non-Pareto Points')
    ax2.legend()

    # add the start and end points as little red hollow circles
    if start < len(all_points):
        ax2.scatter(all_points[start][0], all_points[start][1], c='red', marker='o', edgecolors='green', s=32,
                    alpha=0.4)
    if end < len(all_points):
        ax2.scatter(all_points[end][0], all_points[end][1], c='red', marker='o', edgecolors='red', s=32, alpha=0.3)

    # Add a title to the whole figure
    fig.suptitle(r'Design Space $D$ vs Target Space $T$', fontsize=16)
    # Show the plots
    plt.tight_layout()
    plt.show()


def plot_pareto_optimal_paths(data, full_data, optimal_paths, extra_plot_items_thresh, design_names, target_names, **kwargs):
    filtered_data_df = data

    filtered_points = filtered_data_df[target_names].to_numpy()
    all_points = full_data[target_names].to_numpy()

    pareto_points, non_pareto_points = calculate_pareto_points(all_points)

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=512)

    # Plot the Target space and overlay the paths
    ax1.scatter(pareto_points[:, 0], pareto_points[:, 1], c='blue', marker='^', label='Pareto points')
    if len(filtered_points) > 0:
        ax1.scatter(filtered_points[:, 0], filtered_points[:, 1], s=5, c='orange', label='Nodes of the graph')

    cmap = matplotlib.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, len(optimal_paths)))
    linecycler = cycle(["-", "--", "-.", ":"])

    if len(optimal_paths) < extra_plot_items_thresh:
        # Overlay the paths
        for idx, (path, hv_diff, length_diff) in enumerate(optimal_paths):
            path_points = all_points[path]
            ax1.plot(path_points[:, 0], path_points[:, 1], label=f'Path {idx}', color=colors[idx],
                     linestyle=next(linecycler), lw=1.1)

    ax1.set_xlabel(target_names[0])
    ax1.set_ylabel(target_names[1])
    ax1.set_title('Plot 3: Target Space with Paths')
    ax1.legend(loc=1, prop={'size': 8})

    # Plot the delta-hypervolume vs delta-pathlength
    hv_diffs = [hv_diff for _, hv_diff, _ in optimal_paths]
    length_diffs = [length_diff for _, _, length_diff in optimal_paths]
    ax2.scatter(hv_diffs, length_diffs, s=25, c=colors)

    if len(optimal_paths) < extra_plot_items_thresh:
        for idx, (hv_diff, length_diff) in enumerate(zip(hv_diffs, length_diffs)):
            ax2.text(hv_diff, length_diff, str(idx), fontsize=8, ha='right' if idx % 2 == 0 else 'left',
                     va='bottom' if idx % 3 == 0 else 'top')

    ax2.set_xlabel('Delta-Hypervolume')
    ax2.set_ylabel('Delta-Pathlength')
    ax2.set_title('Plot 4: Delta-Hypervolume vs Delta-Pathlength')

    handles, labels = ax2.get_legend_handles_labels()

    if len(optimal_paths) < extra_plot_items_thresh:
        # Create custom legend entries for the text items, "Hack" to add text to the legend
        for idx in range(len(optimal_paths)):
            handles.append(
                plt.Line2D([0], [0], marker='o', color='w', label=f'Path {idx:>2}, dHV: {hv_diffs[idx]:>10.6f}',
                           markerfacecolor=colors[idx], markersize=10))
    title = '\n'.join(['Hypervolume=' + kwargs.get('hv'), 'Reference point: ' + kwargs.get('ref_point'),
                       'Target length=' + kwargs.get('target_len')])
    ax2.legend(title=title, handles=handles, loc='best', prop={'size': 8})

    # Add a title to the whole figure
    fig.suptitle('Pareto Optimal Paths', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_directed_graph(graph, results=None):
    graph.es['width'] = 0.5
    if results is not None:
        for path in results:
            for i in range(len(path) - 1):
                edge_id = graph.get_eid(path[i], path[i + 1])
                graph.es[edge_id]['width'] = 2.5

    fig, ax = plt.subplots(figsize=(16, 9), dpi=128)
    ig.plot(
        graph,
        target=ax,
        # layout='circle',
        vertex_color='steelblue',
        vertex_label=range(graph.vcount()),
        edge_width=graph.es['width'],
        # edge_label=directed_graph.es["weight"],
        edge_color='#666',
        edge_align_label=True,
        edge_background='white'
    )
    plt.show()


def construct_path_in_design_space(data, optimal_paths, design_names, d_units, increments_visualization):
    # the initial system state
    initial_state_index = optimal_paths[0][0][0]
    target_state_index = optimal_paths[0][0][-1]

    initial_state = data.iloc[initial_state_index][design_names].to_numpy()
    target_state = data.iloc[target_state_index][design_names].to_numpy()

    print(f'Design Space:     {"".join([f"{name:<20}" for name in design_names])}')

    state = ' ,    '.join([f'{LETTERS[i]}: {initial_state[i] / 1000:>8.0f}kWp' if d_units[i] == 'W' else
                           f'{LETTERS[i]}: {initial_state[i] / 3.6e6:>8.0f}kWh' if d_units[i] == 'J' else
                           f'{LETTERS[i]}: {initial_state[i]:>8.0f}#' for i in range(len(design_names))])
    print("Initial State:    " + state)

    target = ' ,    '.join([f'{LETTERS[i]}: {target_state[i] / 1000:>8.0f}kWp' if d_units[i] == 'W' else
                            f'{LETTERS[i]}: {target_state[i] / 3.6e6:>8.0f}kWh' if d_units[i] == 'J' else
                            f'{LETTERS[i]}: {target_state[i]:>8.0f}#' for i in range(len(design_names))])
    print("Target State:     " + target)

    print("\nPaths and increments:")

    for idx, (path, _, _) in enumerate(optimal_paths):
        print(f"\nPath {idx}: {path}")
        for i in range(len(path) - 1):
            current_index = path[i]
            next_index = path[i + 1]
            current_state = data.iloc[current_index][design_names].to_numpy()
            next_state = data.iloc[next_index][design_names].to_numpy()
            increment = next_state - current_state

            increments = ' ,  '.join([f'{LETTERS[i]}: {increment[i] / 1000:>8.0f}' if d_units[i] == 'W' else
                                      f'{LETTERS[i]}: {increment[i] / 3.6e6:>8.0f}' if d_units[i] == 'J' else
                                      f'{LETTERS[i]}: {increment[i]:>8.0f}' for i in range(len(design_names))])
            print(f"Step {i + 1:>2}: From {current_index:>2} to {next_index:>2} -> Increment:  " + increments)

    if increments_visualization:
        increments_polar_plots(data, optimal_paths, design_names, target_state)


def increments_polar_plots(data, optimal_paths, design_names, target_state):
    # Create a subplot figure
    fig = make_subplots(rows=int(math.ceil(len(optimal_paths) / 2)), cols=2,
                        subplot_titles=[f'Path {idx}' for idx in range(len(optimal_paths))],
                        specs=[[{'type': 'polar'}, {'type': 'polar'}]] * int(math.ceil(len(optimal_paths) / 2)),
                        horizontal_spacing=0.18,
                        vertical_spacing=0.08)

    for idx, (path, _, _) in enumerate(optimal_paths):
        labels = ['Start'] * len(design_names)
        for i in range(1, len(path)):
            labels.extend([f'Step {i}'] * len(design_names))

        types = list(design_names) * len(path)

        progress = [0] * len(design_names)
        for i in range(1, len(path)):
            next_index = path[i]
            next_state = data.iloc[next_index][design_names].to_numpy()
            progress.extend((next_state / target_state).tolist())

        df = pd.DataFrame(data={'labels': labels, 'types': types, 'progress': progress})

        polar_fig = px.line_polar(df, r="progress", theta="labels", color="types",
                                  color_discrete_sequence=px.colors.sequential.Electric,
                                  direction='clockwise',
                                  template="presentation")

        for trace in polar_fig.data:
            trace.showlegend = idx == 0
            fig.add_trace(trace, row=idx // 2 + 1, col=idx % 2 + 1)

    fig.update_annotations(yshift=15)
    fig.update_layout(legend=dict(xanchor="right", x=1.5, y=0.75))
    fig.update_layout(height=400 * (math.ceil(len(optimal_paths) / 2)),
                      width=1000,
                      title_text="Increments Polar Plots",
                      title_x=0.5, )
    # fig.write_image('increments_polar_plots.png')
    fig.show(renderer="browser")


def poi_selector_plot(args, *, color_sequence=None, index=None, start_index=None):
    data, d, t, _ = args

    target_names = data.columns[-t:]
    plot_data = data
    if index is not None:
        # plot only the data, which is not selected by the index:
        plot_data = data.drop(index)

    hover_dict = {name: ':.0f' for name in plot_data.columns[:d]}
    hover_dict.update({name: False for name in target_names})
    hover_dict.update({'index': plot_data.index})
    # Create the plotly scatter plot
    fig = px.scatter(
        plot_data,
        x=target_names[0],
        y=target_names[1],
        hover_data=hover_dict,
        title='POI Selector Plot'
    )

    fig.update_layout(clickmode='event+select')
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell",
        )
    )
    if index is not None:
        # Highlight the selected points
        data['color'] = None
        data.loc[index, 'color'] = color_sequence if color_sequence is not None else 'red'
        scatter_trace = px.scatter(data.iloc[index], x=target_names[0], y=target_names[1], hover_data={'index': index},
                                   color=color_sequence, color_discrete_map='identity').update_traces(
            marker=dict(size=6)).data
        for scat in scatter_trace:
            scat.showlegend = False
            fig.add_trace(scat)

        if start_index is not None:
            # Highlight the start point
            fig.add_trace(px.scatter(data.iloc[[start_index]], x=target_names[0], y=target_names[1],
                                     hover_data={'index': [start_index]}, color_discrete_sequence=['blue'])
                          .update_traces(marker=dict(size=8, symbol='circle', line=dict(color='green', width=2)))
                          .data[0])
    # Show the plot
    fig.show()


def search_for_start_candidates(args, *, obj_2_threshold=None, print_candidates_thresh=5000):
    data, d, t, _ = args
    design_names = data.columns[:d]
    design_data = data[design_names]

    design_array = design_data.to_numpy()
    # take the points where obj_2 is larger then 25:
    start_selection = design_data[data['Residual_demand'] > obj_2_threshold] if obj_2_threshold is not None else design_data
    start_selection = start_selection.to_numpy()
    # iterate over these points and count, how many points are larger in every dimension:
    for idx, row in enumerate(start_selection):
        mask = (design_array > row).all(axis=1)
        num_larger_points = np.sum(mask)
        if num_larger_points > print_candidates_thresh:
            print(f'Point {idx} has {num_larger_points} larger points')
            print(f'Index in full data: {design_data.index[np.all(design_data == row, axis=1)]}')


def poi_selector_show_increments(args, *, start_idx, calc_unions=False):
    data, d, t, _ = args
    design_names = data.columns[:d]
    design_data = data[design_names]

    start_point = design_data.iloc[start_idx]
    # select all rows, where every entry is larger than the corresponding entry in the start point
    larger_points = design_data[(design_data > start_point).all(axis=1)]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(larger_points.index)

    colors = None

    # calculate for which target point, there are the most paths available
    if calc_unions:
        design_space_size_dict = {}
        for point in larger_points.index:
            # get the design space for this point
            design_space = data[design_names].to_numpy()
            start = design_space[start_idx]
            end = design_space[point]
            mask = np.all(design_space >= start, axis=1) & np.all(design_space <= end, axis=1)
            filtered_data = data[mask]
            design_space_size_dict[point] = len(filtered_data)

        max_key = max(design_space_size_dict, key=design_space_size_dict.get)
        min_value = min(design_space_size_dict.values())
        max_value = max(design_space_size_dict.values())
        print(f'The largest possible path space is between {start_idx} and {max_key} with {max_value} points')

        # create a color sequence, starting with blue and becoming more red for the top 10% of the values:
        norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
        colormap = matplotlib.cm.get_cmap('cool')
        colors = [matplotlib.colors.to_hex(colormap(norm(value))) for value in design_space_size_dict.values()]

    poi_selector_plot(args, color_sequence=colors, index=larger_points.index, start_index=start_idx)
