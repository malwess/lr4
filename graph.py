'''class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def bfs(root):
        queue = deque([root])
        while queue:
            node = queue.popleft()
            print(node.value)
            for child in node.children:
                queue.append(child)

    bfs(root)

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, value):
        if value not in self.nodes:
            self.nodes[value] = []

    def add_edge(self, from_node, to_node):
        if from_node in self.nodes and to_node in self.nodes:
            self.nodes[from_node].append(to_node)

    graph = Graph()
    for node in ["A", "B", "C", "D"]:
        graph.add_node(node)


root = Node("A")
node_b = Node("B")
node_c = Node("C")
node_d = Node("D")
node_e = Node("E")
node_f = Node("F")

root.add_child(node_b)
root.add_child(node_c)
node_b.add_child(node_d)
node_b.add_child(node_e)
node_c.add_child(node_f)
graph.add_edge("A", "B")
graph.add_edge("A", "C")
graph.add_edge("B", "D")
graph.add_edge("C", "D")'''

"""def bfs(graph, start):
    visited = [start]
    queue = [start]
    paths = {start: [start]}
    #visited.append(start)

    while len(queue) > 0:
        first = queue[0]
        queue = queue[1:]
        print(f'Посещена вершина: {first}')
        #print(first, end = ' ')
        for neighbor in graph[first]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
                paths[neighbor] = paths[first] + [neighbor]
    print('все пути от начальной вершины')
    for v in paths:
        print(f'{start} => {v}: {paths[v]}')
    return paths

graph = {
    1: [2,3],
    2: [1,4,5],
    3: [1,6],
    4: [2],
    5: [2,6],
    6: [3,5],
}

bfs(graph,1"""






