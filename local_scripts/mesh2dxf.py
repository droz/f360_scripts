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

def polygonsAreEqual(poly1 : list[Vertex], poly2 : list[Vertex]):
    """ Check if two polygons are the same, regardless of the starting point or the orientation.
    Args:
        poly1: The first polygon, as a list of Vertex
        poly2: The second polygon, as a list of Vertex
    Returns:
        True if the polygons are the same, False otherwise
    """
    if len(poly1) != len(poly2):
        return False
    try:
        idx = poly2.index(poly1[0])
    except ValueError:
        return False
    poly2_rot1 = poly2[idx:] + poly2[:idx]
    poly2_rot2 = poly2[idx::-1] + poly2[:idx:-1]
    return poly1 == poly2_rot1 or poly1 == poly2_rot2

def polygonIsInPolygons(polygon : list[Vertex], polygons : list[list[Vertex]]):
    """ Check if a polygon is in a list of polygons.
    Args:
        polygon: The polygon to check, as a list of Vertex
        polygons: The list of polygons
    Returns:
        True if the polygon is in the list, False otherwise
    """
    return any([polygonsAreEqual(polygon, poly) for poly in polygons])

def findPolygons(graph : list[Vertex], max_edges : int):
    """ Find the polygons in the graph
    
    Args:
        graph: The graph to search, as a list of Vertex
        max_edges: The maximum number of edges the polygons should have
    Returns:
        A list of polygons, each polygon being a list of Vertices
    """
    # We are going to find the polygon by walking the graph and finding cycles
    def explorePath(path, polygons):
        # If the last vertex is the same vertex as the first one, we have found a cycle
        if len(path) > 2 and path[-1] is path[0]:
            polygons.append(path[:-1])
            return
        # If the path is maximum length, we are done
        if len(path) > max_edges:
            return
        # Otherwise we explore the neighboors
        for edge in path[-1].edges:
            # If this is the previous vertex, skip it
            if len(path) > 1 and edge == path[-2]:
                continue
            # Otherwise, explore this edge
            new_path = path + [edge]
            explorePath(new_path, polygons)            
    polygons = []
    for vertex in graph:
        explorePath([vertex], polygons)

    # We found many duplicates (the same polygon in reverse order,
    # or the same polygon with a different starting point).
    # Remove them here.
    polygons_no_dupes = []
    for polygon1 in polygons:
        if polygonIsInPolygons(polygon1, polygons_no_dupes):
            continue
        polygons_no_dupes.append(polygon1)

    return polygons_no_dupes

def findRedundantQuads(quads, triangles):
    """ Some polygons are actually made of two triangles.
        This removes them from the list.
    Args:
        quads: The list of quads
        triangles: The list of triangles
    Returns:
        The list of quads, minus the ones that are actually two triangles
    """
    new_quads = []
    for quad in quads:
        # Check if the quad is actually two triangles
        triangle1 = [quad[0], quad[1], quad[3]]
        triangle2 = [quad[1], quad[2], quad[3]]
        triangle3 = [quad[0], quad[1], quad[2]]
        triangle4 = [quad[0], quad[2], quad[3]]
        if (polygonIsInPolygons(triangle1, triangles) and
            polygonIsInPolygons(triangle2, triangles)):
            continue
        if (polygonIsInPolygons(triangle3, triangles) and
            polygonIsInPolygons(triangle4, triangles)):
            continue
        new_quads.append(quad)
    return new_quads


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

    # Find the polygons in the graph
    polygons = findPolygons(graph, 4)
    triangles = [poly for poly in polygons if len(poly) == 3]
    quads = [poly for poly in polygons if len(poly) == 4]

    # Remove the quads that are actually two triangles
    quads = findRedundantQuads(quads, triangles)
    print('Found %d triangles and %d quads' % (len(triangles), len(quads)))

    print()