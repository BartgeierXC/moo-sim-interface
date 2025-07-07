import numpy as np


def are_all_simple_paths_monotonic(data, d_design, d_target):
    # iterate over every datapoint, check for every other datapoint, if it is increasing in the design space, is there at least one increase in the target space:
    print(f'Checking {len(data)} datapoints for monotonicity')

    data_matrix = data.to_numpy()

    for i, row in enumerate(data_matrix):
        for j in range(i + 1, len(data_matrix)):
            # check if the design space is increasing
            if np.all(data_matrix[i, :d_design] < data_matrix[j, :d_design]):
                if np.all(data_matrix[i, -d_target:] > data_matrix[j, -d_target:]):
                    print(f"Monotonicity violated between {i} and {j}")
                    print(f'Row {i}: {row[i, :d_design]} -> {data_matrix[j, :d_design]}')
                    print(f'Row {i}: {row[-d_target:]} -> {data_matrix[j, -d_target:]}')
