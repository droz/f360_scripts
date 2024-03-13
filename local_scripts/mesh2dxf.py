""" Open a mesh exported by ExportMeshSketches.py, analyse
    it and create a DXF file for all the panels """


import json
from skspatial.objects import Plane, Vector, Point, Line
import ezdxf

class Vertex:
    def __init__(self, xyz : list[float]):
        # The point coordinates
        self.point = Point(xyz)
        # The neighboor vertices
        self.edges = []

    def __str__(self):
        return 'Vertex : (%f, %f, %f), %d edges' % (self.point[0], self.point[1], self.point[2], len(self.edges))


def maybeAddVertexToGraph(graph : list[Vertex], vertex : Vertex, tolerance : float = 0.001):
    """ Find a vertex in the graph by proximity. If it is not found, add it.
    This is a linear search, so it is not efficient.
    Args:
        graph: The graph to search, as a list of Vertex
        vertex: The new vertex, as a Vertex
    Returns:
        The graph vertex if found, or the new vertex if not found.
    """
    for graph_vertex in graph:
        if Vector(graph_vertex.point - vertex.point).norm() < tolerance:
            return graph_vertex
    graph.append(vertex)
    return vertex

def buildMeshGraph(mesh_data):
    """ Build a graph of the mesh """
    graph = []
    for edge in mesh_data:
        start = Vertex(edge['start'])
        end = Vertex(edge['end'])

        # Try to find the start and end vertices in the graph
        start = maybeAddVertexToGraph(graph, start)
        end = maybeAddVertexToGraph(graph, end)

        # Add the edge to the graph
        start.edges.append(end)
        end.edges.append(start)

    num_edges = sum([len(vertex.edges) for vertex in graph]) / 2
    print('Graph has %d vertices and %d edges' % (len(graph), num_edges))

    return graph



with open('/tmp/mesh_export.json') as file:
    json_data = json.load(file)

meshes = json_data['meshes']

for mesh in meshes:
    name = mesh['name']
    mesh_data = mesh['edges']
    print(name)
    print("-" * len(name))
    # Build a graph of this mesh
    graph = buildMeshGraph(mesh_data)
    print()