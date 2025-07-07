import numpy as np


def calculate_pareto_points(y):
    pareto_points = []
    non_pareto_points = []
    for i, point in enumerate(y):
        is_pareto = True
        for j, other in enumerate(y):
            if i == j:
                continue
            if np.all(point >= other) and np.any(point > other):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(point)
        else:
            non_pareto_points.append(point)
    return np.array(pareto_points), np.array(non_pareto_points)
