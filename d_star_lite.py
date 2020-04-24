import numpy as np
import heapq

from numpy.random import random_integers as rnd, randint
import matplotlib.pyplot as plt


class Element:
    def __init__(self, key, value1, value2):
        self.key = key
        self.value1 = value1
        self.value2 = value2

    def __eq__(self, other):
        return np.sum(np.abs(self.key - other.key)) == 0

    def __ne__(self, other):
        return self.key != other.key

    def __lt__(self, other):
        return (self.value1, self.value2) < (other.value1, other.value2)

    def __le__(self, other):
        return (self.value1, self.value2) <= (other.value1, other.value2)

    def __gt__(self, other):
        return (self.value1, self.value2) > (other.value1, other.value2)

    def __ge__(self, other):
        return (self.value1, self.value2) >= (other.value1, other.value2)


class DStarLitePlanning:
    def __init__(self, r_map, sx, sy, gx, gy):
        self.start = np.array([sx, sy])
        self.goal = np.array([gx, gy])
        self.k_m = 0
        self.rhs = np.ones((len(r_map), len(r_map[0]))) * np.inf
        self.g = self.rhs.copy()
        self.global_map = r_map
        self.sensed_map = np.zeros((len(r_map), len(r_map[0])))
        self.rhs[self.goal[0], self.goal[1]] = 0
        self.queue = []
        node = Element(self.goal, *self.CalculateKey(self.goal))
        heapq.heappush(self.queue, node)

    def CalculateKey(self, node):
        key = [0, 0]
        key[0] = min(self.g[node[0], node[1]], self.rhs[node[0], node[1]]) + self.h_estimate(self.start,
                                                                                             node) + self.k_m
        key[1] = min(self.g[node[0], node[1]], self.rhs[node[0], node[1]])
        return key

    def UpdateVertex(self, u):
        if np.sum(np.abs(u - self.goal)) != 0:
            s_list = self.succ(u)
            min_s = np.inf
            for s in s_list:
                if self.cost(u, s) + self.g[s[0], s[1]] < min_s:
                    min_s = self.cost(u, s) + self.g[s[0], s[1]]
            self.rhs[u[0], u[1]] = min_s
        if Element(u, 0, 0) in self.queue:
            self.queue.remove(Element(u, 0, 0))
            heapq.heapify(self.queue)
        if self.g[u[0], u[1]] != self.rhs[u[0], u[1]]:
            heapq.heappush(self.queue, Element(u, *self.CalculateKey(u)))

    def ComputeShortestPath(self):
        while len(self.queue) > 0 and \
                heapq.nsmallest(1, self.queue)[0] < Element(self.start, *self.CalculateKey(self.start)) or \
                self.rhs[self.start[0], self.start[1]] != self.g[self.start[0], self.start[1]]:
            u = heapq.heappop(self.queue).key
            if self.g[u[0], u[1]] > self.rhs[u[0], u[1]]:
                self.g[u[0], u[1]] = self.rhs[u[0], u[1]]
                s_list = self.succ(u)
                for s in s_list:
                    self.UpdateVertex(s)
            else:
                self.g[u[0], u[1]] = np.inf
                s_list = self.succ(u)
                s_list.append(u)
                for s in s_list:
                    self.UpdateVertex(s)

    # fetch successors and predessors
    def succ(self, u):
        s_list = [np.array([u[0] - 1, u[1] - 1]), np.array([u[0] - 1, u[1]]), np.array([u[0] - 1, u[1] + 1]),
                  np.array([u[0], u[1] - 1]), np.array([u[0], u[1] + 1]), np.array([u[0] + 1, u[1] - 1]),
                  np.array([u[0] + 1, u[1]]), np.array([u[0] + 1, u[1] + 1])]
        row = len(self.global_map)
        col = len(self.global_map[0])
        real_list = []
        for s in s_list:
            if 0 <= s[0] < row and 0 <= s[1] < col:
                real_list.append(s)
        return real_list

    # heuristic estimation
    def h_estimate(self, s1, s2):
        return np.linalg.norm(s1 - s2)

    # calculate cost between nodes
    def cost(self, u1, u2):
        if self.sensed_map[u1[0], u1[1]] == np.inf or self.sensed_map[u2[0], u2[1]] == np.inf:
            return np.inf
        else:
            return self.h_estimate(u1, u2)

    def sense(self, range_s):
        real_list = []
        row = len(self.global_map)
        col = len(self.global_map[0])
        for i in range(-range_s, range_s + 1):
            for j in range(-range_s, range_s + 1):
                if 0 <= self.start[0] + i < row and 0 <= self.start[1] + j < col:
                    if not (i == 0 and j == 0):
                        real_list.append(np.array([self.start[0] + i, self.start[1] + j]))
        return real_list


def Main(global_map, gx, gy, sx, sy):
    node = DStarLitePlanning(global_map, sx, sy, gx, gy)
    last = node.start
    last = ScanAndUpdate(node, last)
    node.ComputeShortestPath()
    while np.sum(np.abs(node.start - node.goal)) != 0:
        s_list = node.succ(node.start)
        min_s = np.inf
        for s in s_list:
            plt.plot(s[0],s[1], 'xy')
            if node.cost(node.start, s) + node.g[s[0], s[1]] < min_s:
                min_s = node.cost(node.start, s) + node.g[s[0], s[1]]
                temp = s
        node.start = temp.copy()
        print(node.start[0], node.start[1])
        plt.plot(node.start[0], node.start[1], '.b')
        last = ScanAndUpdate(node, last)
        plt.pause(0.1)

def ScanAndUpdate(node, last):
    s_list = node.sense(3)
    flag = True
    for s in s_list:
        if node.sensed_map[s[0], s[1]] != node.global_map[s[0], s[1]]:
            flag = False
            print('See a wall!')
            break
    if flag == False:
        node.k_m += node.h_estimate(last, node.start)
        last = node.start.copy()
        for s in s_list:
            if node.sensed_map[s[0], s[1]] != node.global_map[s[0], s[1]]:
                plt.plot(s[0],s[1], 'xr')
                node.sensed_map[s[0], s[1]] = node.global_map[s[0], s[1]]
                node.UpdateVertex(s)
        for i in range(len(node.queue)):
            u = heapq.heappop(node.queue).key
            temp = Element(u, *node.CalculateKey(u))
            heapq.heappush(node.queue, temp)
        heapq.heapify(node.queue)
        node.ComputeShortestPath()
    return last


# randomly generate connected maze
def maze(width, height, complexity=.06, density=.01):
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * (shape[0] // 2 * shape[1] // 2))
    # Build actual maze
    z = np.zeros(shape, dtype=float)
    # Fill borders
    z[0, :] = z[-1, :] = 1
    z[:, 0] = z[:, -1] = 1
    # Make isles
    for i in range(density):
        x, y = randint(0, shape[1] // 2) * 2, randint(0, shape[0] // 2) * 2
        z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:           neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:           neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[randint(0, len(neighbours) - 1)]
                if z[y_, x_] == 0:
                    z[y_, x_] = 1
                    z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    return z


if __name__ == "__main__":
    # set start and goal point
    sx = 1
    sy = 1
    gx = 49
    gy = 49
    grid_size = 2.0

    # set obstable positions
    ox, oy = [], []
    # global_map = maze(width=50, height=50)
    # global_map[global_map == 1] = np.inf
    # np.savetxt("global_map.txt", global_map)
    global_map = np.loadtxt('map/global_map.txt')

    for i in range(1, len(global_map)):
        for j in range(1, len(global_map[i])):
            if global_map[i][j] == np.inf:
                ox.append(i)
                oy.append(j)
    plt.grid(True)
    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")

    Main(global_map, gx, gy, sx, sy)

    # plt.plot(rx, ry, "-r")
    plt.show()
