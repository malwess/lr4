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


class Graph:
    #Граф, реализованный через список смежности

    def __init__(self, directed=False):

        self.graph = {}  # Словарь: вершина -> список соседних вершин
        self.directed = directed

    def add_vertex(self, vertex):
        #Добавить вершину в граф
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, vertex1, vertex2):
        #Добавить ребро между vertex1 и vertex2
        # Добавляем вершины, если их еще нет
        if vertex1 not in self.graph:
            self.add_vertex(vertex1)
        if vertex2 not in self.graph:
            self.add_vertex(vertex2)

        # Добавляем ребро vertex1 -> vertex2
        if vertex2 not in self.graph[vertex1]:
            self.graph[vertex1].append(vertex2)

        # Для неориентированного графа добавляем обратное ребро
        if not self.directed and vertex1 not in self.graph[vertex2]:
            self.graph[vertex2].append(vertex1)

    def remove_edge(self, vertex1, vertex2):
        #Удалить ребро между vertex1 и vertex2
        if vertex1 in self.graph and vertex2 in self.graph[vertex1]:
            self.graph[vertex1].remove(vertex2)

        if not self.directed and vertex2 in self.graph and vertex1 in self.graph[vertex2]:
            self.graph[vertex2].remove(vertex1)

    def remove_vertex(self, vertex):
        #Удалить вершину и все связанные с ней ребра
        if vertex in self.graph:
            # Удаляем все ребра, ведущие к этой вершине
            for v in self.graph:
                if vertex in self.graph[v]:
                    self.graph[v].remove(vertex)

            # Удаляем саму вершину
            del self.graph[vertex]

    def get_vertices(self):
        #Получить список всех вершин
        return list(self.graph.keys())

    def get_neighbors(self, vertex):
        if vertex in self.graph:
            return self.graph[vertex].copy()  # Возвращаем копию, чтобы не изменять оригинал
        return []

    def has_vertex(self, vertex):
        #Проверить, существует ли вершина в графе
        return vertex in self.graph

    def has_edge(self, vertex1, vertex2):
        #Проверить, существует ли ребро между vertex1 и vertex2
        if vertex1 in self.graph and vertex2 in self.graph:
            return vertex2 in self.graph[vertex1]
        return False

    def degree(self, vertex):
        #Степень вершины (количество исходящих ребер)
        if vertex in self.graph:
            return len(self.graph[vertex])
        return 0





    def bfs(self, start_vertex):
        #Обход графа в ширину

        if start_vertex not in self.graph:
            return []

        visited = []
        queue = []

        # Начинаем с начальной вершины
        visited.append(start_vertex)
        queue.append(start_vertex)

        while queue:
            current = queue.pop(0)  # Берем первую вершину из очереди

            # Добавляем всех непосещенных соседей
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.append(neighbor)
                    queue.append(neighbor)

        return visited

    def bfs_with_paths(self, start_vertex):

        #BFS с сохранением пути от начальной вершины до каждой

        if start_vertex not in self.graph:
            return {}

        visited = {start_vertex: [start_vertex]}
        queue = [start_vertex]

        while queue:
            current = queue.pop(0)
            current_path = visited[current]

            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = current_path + [neighbor]
                    queue.append(neighbor)

        return visited





    def dfs(self, start_vertex):

        #Обход графа в глубину

        if start_vertex not in self.graph:
            return []

        visited = []
        stack = []

        # Добавляем начальную вершину в стек
        stack.append(start_vertex)

        while stack:
            current = stack.pop()  # Берем последнюю вершину из стека

            if current not in visited:
                visited.append(current)

                # Добавляем соседей в стек
                # Чтобы сохранить порядок обхода как в рекурсивной версии,
                # добавляем соседей в обратном порядке
                neighbors = self.graph.get(current, [])
                for neighbor in reversed(neighbors):
                    if neighbor not in visited:
                        stack.append(neighbor)

        return visited

    def dfs_recursive(self, start_vertex, visited=None):

        #Рекурсивный обход графа в глубину
        #Возвращает список вершин в порядке их посещения

        if start_vertex not in self.graph:
            return []

        if visited is None:
            visited = []

        if start_vertex not in visited:
            visited.append(start_vertex)

            for neighbor in self.graph.get(start_vertex, []):
                if neighbor not in visited:
                    self.dfs_recursive(neighbor, visited)

        return visited

    def find_path(self, start_vertex, end_vertex):

        if start_vertex not in self.graph or end_vertex not in self.graph:
            return []

        # Используем BFS для поиска пути
        visited = {start_vertex: None}  # Словарь: вершина -> предыдущая вершина
        queue = [start_vertex]

        while queue:
            current = queue.pop(0)

            # Если достигли конечной вершины, восстанавливаем путь
            if current == end_vertex:
                path = []
                while current is not None:
                    path.append(current)
                    current = visited[current]
                return path[::-1]  # Разворачиваем путь

            # Добавляем соседей
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)

        return []  # Путь не найден

    def connected_components(self):
        visited_global = set()
        components = []

        for vertex in self.graph:
            if vertex not in visited_global:
                # Находим все вершины, достижимые из текущей
                component = []
                stack = [vertex]
                visited_local = set()

                while stack:
                    current = stack.pop()
                    if current not in visited_local:
                        visited_local.add(current)
                        visited_global.add(current)
                        component.append(current)

                        for neighbor in self.graph.get(current, []):
                            if neighbor not in visited_local:
                                stack.append(neighbor)

                components.append(component)

        return components

    def is_connected(self):
        if not self.graph:
            return True  # Пустой граф считается связным

        vertices = list(self.graph.keys())
        # Проверяем, достижимы ли все вершины из первой
        reachable = self.bfs(vertices[0])
        return len(reachable) == len(vertices)

    def __str__(self):
        result = []
        for vertex in sorted(self.graph.keys()):
            neighbors = sorted(self.graph[vertex])
            result.append(f"{vertex}: {neighbors}")
        return "\n".join(result)

    def __repr__(self):
        return f"Graph(directed={self.directed}, vertices={len(self.graph)})"


# Пример использования и тестирования
def main():
    print("=== Пример 1: Неориентированный граф ===")
    g1 = Graph(directed=False)

    # Добавляем вершины и ребра
    edges = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'D'),
        ('B', 'E'),
        ('C', 'F'),
        ('E', 'F'),
        ('F', 'G')
    ]

    for v1, v2 in edges:
        g1.add_edge(v1, v2)

    print("Граф G1:")
    print(g1)
    print()

    print("Все вершины:", g1.get_vertices())
    print("Соседи вершины 'B':", g1.get_neighbors('B'))
    print("Существует ли ребро A-B?", g1.has_edge('A', 'B'))
    print("Степень вершины 'B':", g1.degree('B'))
    print()

    print("Обход в ширину (BFS) из 'A':", g1.bfs('A'))
    print("Обход в глубину (DFS итеративный) из 'A':", g1.dfs('A'))
    print("Обход в глубину (DFS рекурсивный) из 'A':", g1.dfs_recursive('A'))
    print()

    print("BFS с путями из 'A':")
    paths = g1.bfs_with_paths('A')
    for vertex, path in sorted(paths.items()):
        print(f"  Путь к {vertex}: {path}")
    print()

    print("Поиск пути от 'A' до 'G':", g1.find_path('A', 'G'))
    print("Поиск пути от 'D' до 'G':", g1.find_path('D', 'G'))
    print()

    print("Связные компоненты:", g1.connected_components())
    print("Граф связный?", g1.is_connected())
    print()

    print("=== Пример 2: Ориентированный граф ===")
    g2 = Graph(directed=True)

    g2.add_edge('A', 'B')
    g2.add_edge('A', 'C')
    g2.add_edge('B', 'D')
    g2.add_edge('C', 'D')
    g2.add_edge('D', 'E')
    g2.add_edge('E', 'F')

    print("Граф G2:")
    print(g2)
    print()

    print("Обход в ширину (BFS) из 'A':", g2.bfs('A'))
    print("Обход в глубину (DFS) из 'A':", g2.dfs('A'))
    print()

    print("=== Пример 3: Граф с несколькими компонентами связности ===")
    g3 = Graph(directed=False)

    # Первая компонента связности
    g3.add_edge('A', 'B')
    g3.add_edge('B', 'C')
    g3.add_edge('C', 'A')

    # Вторая компонента связности
    g3.add_edge('D', 'E')

    # Третья компонента (изолированная вершина)
    g3.add_vertex('F')

    print("Граф G3:")
    print(g3)
    print()

    print("Связные компоненты:")
    for i, component in enumerate(g3.connected_components(), 1):
        print(f"  Компонента {i}: {component}")

    print("Граф связный?", g3.is_connected())
    print()

    print("=== Пример 4: Удаление вершин и ребер ===")
    g4 = Graph(directed=False)
    g4.add_edge('A', 'B')
    g4.add_edge('A', 'C')
    g4.add_edge('B', 'C')
    g4.add_edge('C', 'D')

    print("Исходный граф G4:")
    print(g4)
    print()

    print("Удаляем ребро A-B:")
    g4.remove_edge('A', 'B')
    print(g4)
    print()

    print("Удаляем вершину C:")
    g4.remove_vertex('C')
    print(g4)


if __name__ == "__main__":
    main()




