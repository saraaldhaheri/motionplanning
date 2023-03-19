
# import math
# import heapq
# import matplotlib.pyplot as plt

# # define the obstacles
# obstacles = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2), (9, 5, 2), (8, 10, 1)]

# # define the start and goal positions
# start = (0, 0)
# goal = (6, 10)

# # define the radius of the robot
# robot_radius = 0.8

# # define a function to check if a point is within an obstacle
# def within_obstacle(point):
#     for obs in obstacles:
#         distance = math.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
#         if distance <= (robot_radius + obs[2]):
#             return True
#     return False

# # define a function to get the neighbors of a point
# def get_neighbors(point):
#     neighbors = []
#     for x in range(point[0]-1, point[0]+2):
#         for y in range(point[1]-1, point[1]+2):
#             if x == point[0] and y == point[1]:
#                 continue
#             neighbor = (x, y)
#             if not within_obstacle(neighbor):
#                 neighbors.append(neighbor)
#     return neighbors

# # define a function to calculate the distance between two points
# def distance(point1, point2):
#     return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# # define the A* search algorithm
# def a_star(start, goal):
#     frontier = [(0, start)]
#     came_from = {start: None}
#     cost_so_far = {start: 0}
#     while frontier:
#         current = heapq.heappop(frontier)[1]
#         if current == goal:
#             break
#         for next in get_neighbors(current):
#             new_cost = cost_so_far[current] + distance(current, next)
#             if next not in cost_so_far or new_cost < cost_so_far[next]:
#                 cost_so_far[next] = new_cost
#                 priority = new_cost + distance(goal, next)
#                 heapq.heappush(frontier, (priority, next))
#                 came_from[next] = current
    
#     # create a list of the path from start to goal
#     path = [goal]
#     current = goal
#     while current != start:
#         current = came_from[current]
#         path.append(current)
#     path.reverse()
#     return path

# # run the A* algorithm
# path = a_star(start, goal)

# # plot the environment and the path
# fig, ax = plt.subplots()
# ax.set_xlim([-2, 15])
# ax.set_ylim([-2, 15])

# # plot the obstacles
# for obs in obstacles:
#     circle = plt.Circle((obs[0], obs[1]), obs[2], color='blue', fill=False)
#     ax.add_artist(circle)

# # plot the start and goal positions
# ax.scatter(start[0], start[1], color='green', marker='x', s=100)
# ax.scatter(goal[0], goal[1], color='red', marker='x', s=100)

# # plot the path
# x = [point[0] for point in path]
# y = [point[1] for point in path]
# ax.plot(x, y, color='blue')
# plt.axis("equal")
# plt.grid(True)
# plt.pause(0.01)
# plt.show()

import math
import heapq
import matplotlib.pyplot as plt
import time
from memory_profiler import profile

# define the obstacles
obstacles = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2), (9, 5, 2), (8, 10, 1)]

# define the start and goal positions
start = (0, 0)
goal = (6, 10)

# define the radius of the robot
robot_radius = 0.8

# define a function to check if a point is within an obstacle
def within_obstacle(point):
    for obs in obstacles:
        distance = math.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
        if distance <= (robot_radius + obs[2]):
            return True
    return False

# define a function to get the neighbors of a point
def get_neighbors(point):
    neighbors = []
    for x in range(point[0]-1, point[0]+2):
        for y in range(point[1]-1, point[1]+2):
            if x == point[0] and y == point[1]:
                continue
            neighbor = (x, y)
            if not within_obstacle(neighbor):
                neighbors.append(neighbor)
    return neighbors

# define a function to calculate the distance between two points
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

@profile(precision=4)
# define the A* search algorithm
def a_star(start, goal):
    tic = time.perf_counter()
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        current = heapq.heappop(frontier)[1]
        if current == goal:
            break
        for next in get_neighbors(current):
            new_cost = cost_so_far[current] + distance(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + distance(goal, next)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current
    
    # create a list of the path from start to goal
    path = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()

    toc = time.perf_counter()
    print(f"Motion planning completed in {toc - tic:0.4f} seconds")
    return path

# run the A* algorithm and animate the path
path = a_star(start, goal)

fig, ax = plt.subplots()
ax.set_xlim([-2, 15])
ax.set_ylim([-2, 15])

# plot the obstacles
for obs in obstacles:
    circle = plt.Circle((obs[0], obs[1]), obs[2], color='blue', fill=False)
    ax.add_artist(circle)

# plot the start and goal positions
ax.scatter(start[0], start[1], color='green', marker='x', s=100)
ax.scatter(goal[0], goal[1], color='red', marker='x', s=100)

# animate the path
for i in range(len(path)-1):
    x = [path[i][0], path[i+1][0]]
    y = [path[i][1], path[i+1][1]]
    ax.plot(x, y, color='blue')

ax.scatter(start[0], start[1], color='green', marker='x', s=100)
ax.scatter(goal[0], goal[1], color='red', marker='x', s=100)

x, y = [], []
for point in path:
    x.append(point[0])
    y.append(point[1])
    ax.plot(x, y, color='blue')
    plt.pause(0.01)
    plt.axis("equal")
    plt.grid(True)
    plt.show()