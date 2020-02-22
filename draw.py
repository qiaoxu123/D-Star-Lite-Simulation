from display_lite_full import display_lite_full
import numpy as np

def draw():
    M = open('map.txt', 'r')
    trajectory = open('path.txt', 'r')
    file = open('updated_points.txt', 'r')
    g = textscan(file, '%s', 'delimiter', '\n')
    file.close()
    M_change = np.zeros(np.size(trajectory, 1), 1)
    for i in range(np.size(trajectory, 1)):
        M_change[i, 1] = str2num(g{1,1}{i, 1})
    x_init = 2
    y_init = 2
    x_goal = 10
    y_goal = 10
    range = 3
    display_lite_full(M, M_change, x_init, y_init, x_goal, y_goal, trajectory, range)
    M.close()
    trajectory.close()