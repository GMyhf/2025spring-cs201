class Vertex:
    def __init__(self, key):
        self.key = key
        self.neighbors = {}

    def get_neighbor(self, other):
        return self.neighbors.get(other, None)

    def set_neighbor(self, other, weight=0):
        self.neighbors[other] = weight

    def __repr__(self):  # 为开发者提供调试信息
        return f"Vertex({self.key})"

    def __str__(self):  # 面向用户的输出
        return (
                str(self.key)
                + " connected to: "
                + str([x.key for x in self.neighbors])
        )

    def get_neighbors(self):
        return self.neighbors.keys()

    def get_key(self):
        return self.key


class Graph:
    def __init__(self):
        self.vertices = {}

    def set_vertex(self, key):
        self.vertices[key] = Vertex(key)

    def get_vertex(self, key):
        return self.vertices.get(key, None)

    def __contains__(self, key):
        return key in self.vertices

    def add_edge(self, from_vert, to_vert, weight=0):
        if from_vert not in self.vertices:
            self.set_vertex(from_vert)
        if to_vert not in self.vertices:
            self.set_vertex(to_vert)
        self.vertices[from_vert].set_neighbor(self.vertices[to_vert], weight)

    def get_vertices(self):
        return self.vertices.keys()

    def __iter__(self):
        return iter(self.vertices.values())


def constructLaplacianMatrix(n, edges):
    graph = Graph()
    for i in range(n):  # 添加顶点
        graph.set_vertex(i)

    for edge in edges:  # 添加边
        a, b = edge
        graph.add_edge(a, b)
        graph.add_edge(b, a)

    laplacianMatrix = []  # 构建拉普拉斯矩阵
    for vertex in graph:
        row = [0] * n
        row[vertex.get_key()] = len(vertex.get_neighbors())
        for neighbor in vertex.get_neighbors():
            row[neighbor.get_key()] = -1
        laplacianMatrix.append(row)

    return laplacianMatrix


n, m = map(int, input().split())  # 解析输入
edges = []
for i in range(m):
    a, b = map(int, input().split())
    edges.append((a, b))

laplacianMatrix = constructLaplacianMatrix(n, edges)  # 构建拉普拉斯矩阵

for row in laplacianMatrix:  # 输出结果
    print(' '.join(map(str, row)))