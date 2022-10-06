from queue import PriorityQueue
from copy import deepcopy


def dijkstra(graph, start_vertex):
    path_info = {v: {'distance': float('-inf'), 'path': []} for v in range(graph.v)}
    # D = {v: float('inf') for v in range(graph.v)}
    # D[start_vertex] = 1.0
    path_info[start_vertex]['distance'] = 1.0
    path_info[start_vertex]['path'].append(start_vertex)

    pq = PriorityQueue()
    pq.put((-1.0, start_vertex))

    while not pq.empty():
        (_, current_vertex) = pq.get()
        graph.visited.append(current_vertex)

        for neighbor in range(graph.v):
            if graph.edges[current_vertex][neighbor] != -1:
                distance = graph.edges[current_vertex][neighbor]
                if neighbor not in graph.visited:
                    # old_cost = D[neighbor]
                    old_cost = path_info[neighbor]['distance']
                    # new_cost = D[current_vertex] * distance
                    new_cost = path_info[current_vertex]['distance'] * distance
                    if new_cost > old_cost:

                        pq.put((-new_cost, neighbor))
                        # D[neighbor] = new_cost
                        path_info[neighbor]['distance'] = new_cost
                        path_info[neighbor]['path'] = deepcopy(path_info[current_vertex]['path'])
                        path_info[neighbor]['path'].append(neighbor)

    return path_info
