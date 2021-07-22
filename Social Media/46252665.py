import math
import multiprocessing
import operator
from collections import deque

import networkx as nx


def get_graph():
    graph = nx.Graph()
    for line in open('data.txt').readlines():
        node = [i for i in map(int, line.split())]
        for edge in node[1:]:
            graph.add_edge(node[0], edge)
    return graph


def print_output(centrality, algorithm):
    return print(f'\n{algorithm}: '
                 f'{[i[0] for i in sorted(centrality.items(), key=operator.itemgetter(1), reverse=True)[:10]]}')


def pagerank_centrality(graph):
    alpha = 0.85
    iteration = 100
    epsilon = 3.0e-07

    rank = {}
    for n in [i for i in graph.nodes]:
        rank[n] = 0

    for _ in range(iteration):
        prev_rank = rank.copy()
        for node in range(len(graph)):
            _total = 0
            for n in graph[node]:
                _total += rank[n] / len([edge for edge in graph.neighbors(n)])
            rank[node] = (1 - alpha) / len(graph) + (alpha * _total)
        if sum([abs(prev_rank[_node] - rank[_node]) for _node in rank]) / len(graph) < epsilon:
            return print_output(rank, 'PageRank Centrality')


def brandes_betweenness_centrality(graph):
    nodes = [i for i in graph.nodes]
    adjacent = sorted([i for i in graph.adjacency()], key=lambda tup: tup[0])
    edges = {}
    for adj in adjacent:
        for i in adjacent[adj[0]]:
            if isinstance(i, dict):
                edges[adj[0]] = [*i]
    centrality = dict((v, 0) for v in nodes)
    for s in nodes:
        s_stack = []
        predecessors = dict((w, []) for w in nodes)
        shortest_path = dict((t, 0) for t in nodes)
        distance = dict((t, math.inf) for t in nodes)
        shortest_path[s], distance[s] = 1, 0
        q_queue = deque([])
        q_queue.append(s)
        while q_queue:
            v = q_queue.popleft()
            s_stack.append(v)
            for w in edges[v]:
                if distance[w] == math.inf:
                    distance[w] = distance[v] + 1
                    q_queue.append(w)
                if distance[w] == distance[v] + 1:
                    shortest_path[w] = shortest_path[w] + shortest_path[v]
                    predecessors[w].append(v)
        dependency = dict((v, 0) for v in nodes)
        while s_stack:
            w = s_stack.pop()
            for v in predecessors[w]:
                dependency[v] = dependency[v] + (shortest_path[v] / shortest_path[w]) * (1 + dependency[w])
            if w != s:
                centrality[w] = centrality[w] + dependency[w]

    return print_output(centrality, 'Betweenness Centrality')


if __name__ == "__main__":
    undirected_graph = get_graph()

    # brandes_betweenness_centrality(undirected_graph)
    # pagerank_centrality(undirected_graph)

    multiprocessing.Process(target=brandes_betweenness_centrality, args=(undirected_graph,)).start()
    multiprocessing.Process(target=pagerank_centrality, args=(undirected_graph,)).start()
