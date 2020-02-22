import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import rectangle


def display_lite_full(M, M_change, x_init, y_init, x_goal, y_goal, trajectory, range):
    plt.scatter(10 * (x_init - 0.5), 10 * (y_init - 0.5), 500, 'ko', 'linewidth', 2)
    plt.scatter(10 * (x_goal - 0.5), 10 * (y_goal - 0.5), 500, 'kh', 'linewidth', 2)
    plt.plot()
    plt.axis([-10 10 * np.size(M, 1) + 10 - 10 10 * np.size(M, 2) + 10])
    # set(gca, 'xtick', [])
    # set(gca, 'xticklabel', [])
    # set(gca, 'ytick', [])
    # set(gca, 'yticklabel', [])
    # title('sensed\_map', 'Fontsize', 20)
    # set(gcf, 'Position', [0, 0, 1000, 1000])

    for i in range(np.size(M, 1)):
        for j in range(np.size(M, 2)):
            if M(i, j) == 1:
                #rectangle('Position', [10 * (i - 1), 10 * (j - 1), 10, 10], 'FaceColor', 'b')

    # draw sensed_map with updated points
    for i in range(np.size(trajectory, 1)):
        arr = M_change{i, 1}
        if np.size(arr, 1) > 0:
            for j in range(1 : 2 : np.size(arr, 2) - 1):
                rectangle('Position', [10 * arr(j), 10 * arr(j + 1), 10, 10], 'FaceColor', 'b')

        if i > 1:
            x = [10 * (trajectory(i - 1, 1) - 0.5), 10 * (trajectory(i, 1) - 0.5)]
            y = [10 * (trajectory(i - 1, 2) - 0.5), 10 * (trajectory(i, 2) - 0.5)]
            plt.plot(x, y, 'r', 'linewidth', 3);