""" Open a mesh exported by ExportMeshSketches.py, analyse
    it and create a DXF file for all the panels """


import json
from skspatial.objects import Plane, Vector, Point, Line
import ezdxf

class Vertex:
    """ This class is used to represent a vertex in th emesh graph"""
    def __init__(self, xyz : list[float]):
        # The point coordinates
        self.point = Point(xyz)
        # The neighboor vertices
        self.edges = []

    def __str__(self):
        return 'Vertex : (%f, %f, %f), %d edges' % (self.point[0], self.point[1], self.point[2], len(self.edges))

    def __repr__(self):
        return self.__str__()

class Facet:
    """ This class is used to represent a facet in the mesh """
    def __init__(self, vertices : list[Vertex]):
        # The vertices of the facet
        self.vertices = vertices
        # The projected polygon, as a list of 2D points
        self.polygon2d = None

    def subFacet(self, indexes : list[int]):
        """ Create a sub-facet from a list of indexes.
        Args:
            indexes: The indexes of the vertices to use
        Returns:
            The sub-facet
        """
        return Facet([self.vertices[index] for index in indexes])

    def isSame(self, facet : list[Vertex]):
        """ Check if another facet is actually the same as this one,
            regardless of the starting point or the orientation.
        Args:
            facet: The facet to compare against
        Returns:
            True if the facets are the same, False otherwise
        """
        poly1 = self.vertices
        poly2 = facet.vertices
        if len(poly1) != len(poly2):
            return False
        try:
            idx = poly2.index(poly1[0])
        except ValueError:
            return False
        poly2_rot1 = poly2[idx:] + poly2[:idx]
        poly2_rot2 = poly2[idx::-1] + poly2[:idx:-1]
        return poly1 == poly2_rot1 or poly1 == poly2_rot2

    def isInList(self, facets : list['Facet']):
        """ Check if a facet is part of a list of facet, regardless of start or orientation.
        Args:
            facets: The list of facets to check against
        Returns:
            True if this facet is in the list, False otherwise
        """
        return any([self.isSame(list_facet) for list_facet in facets])

    def adjustPoint(self, index):
        """ Adjust a 2d point to make the edges lengths match between 2d and 3d.
        Args:
            index: The index of the point to adjust
        """
        index_before = index - 1
        if index_before < 0:
            index_before += len(self.vertices)
        index_after = index + 1
        if index_after >= len(self.vertices):
            index_after -= len(self.vertices)
        p2d_b = self.polygon2d[index_before]
        p2d   = self.polygon2d[index]
        p2d_a = self.polygon2d[index_after]
        p3d_b = self.vertices[index_before].point
        p3d   = self.vertices[index].point
        p3d_a = self.vertices[index_after].point
        v3_b = Vector(p3d - p3d_b)
        v3_a = Vector(p3d_a - p3d)
        l3_b = v3_b.norm()
        l3_a = v3_a.norm()
        # We iterate the correction 10 times, it should be enough to converge
        for i in range(10):
            v2_b = Vector(p2d - p2d_b)
            v2_a = Vector(p2d_a - p2d)
            l2_b = v2_b.norm()
            l2_a = v2_a.norm()
            # Nudge the point roughly in the correct direction
            p2d += v2_b * (l3_b - l2_b) / l2_b
            p2d -= v2_a * (l3_a - l2_a) / l2_a

    def project(self):
        """ Project the facet on a plane """
        if len(self.vertices) < 3:
            raise ValueError('Polygon with less than 3 points cannot be projected !!!')
        # Start by fitting a plane to the polygon and projecting the points on it
        poly3d = [vertex.point for vertex in self.vertices]
        plane = Plane.best_fit(poly3d)
        poly3d_proj = [plane.project_point(point) for point in poly3d]

        # Now we need to rotate the polygon to fit a consistent coordinate frame
        # We use the first edge as x axis.
        x = Vector(poly3d_proj[1] - poly3d_proj[0]).unit()
        z = plane.normal
        y = z.cross(x)
        self.polygon2d = []
        for point in poly3d_proj:
            v = Vector(point - plane.point)
            px = x.scalar_projection(v)
            py = y.scalar_projection(v)
            self.polygon2d.append(Point([px, py]))

        # By doing the projection, if the facet was not planar,
        #  we ended up changing the length of the edges. For a quad,
        #  we can fix that by adjusting the points.
        if len(self.vertices) == 4:
            # Find the smallest diagonal, we will keep it constant and adjust the other points
            d1 = Vector(poly3d[2] - poly3d[0]).norm()
            d2 = Vector(poly3d[3] - poly3d[1]).norm()
            if d1 > d2:
                adjust_indexes = [0, 2]
            else:
                adjust_indexes = [1, 3]
            # Now we can adjust the points
            for index in adjust_indexes:
                self.adjustPoint(index)
                
            # Adjust the points to make the edges lengths match
            for index in [0, 2]:
                index_before = index - 1
                if index_before < 0:
                    index_before += 4
                index_after = index + 1
                if index_after >= 4:
                    index_after -= 4
                p2d_b = self.polygon2d[index_before]
                p2d   = self.polygon2d[index]
                p2d_a = self.polygon2d[index_after]
                p3d_b = poly3d_proj[index_before]
                p3d   = poly3d_proj[index]
                p3d_a = poly3d_proj[index_after]
                v3_b = Vector(p3d - p3d_b)
                v3_a = Vector(p3d_a - p3d)
                v2_b = Vector(p2d - p2d_b)
                v2_a = Vector(p2d_a - p2d)
                l3_b = v3_b.norm()
                l3_a = v3_a.norm()
                l2_b = v2_b.norm()
                l2_a = v2_a.norm()
                
                p2d += v2_b * (l3_b - l2_b) / l2_b
                p2d -= v2_a * (l3_a - l2_a) / l2_a

    def __str__(self):
        return 'Facet : %d vertices' % len(self.vertices)

    def __repr__(self):
        return self.__str__()

class Mesh:
    """ This class is used to represent a mesh as a graph of vertices and a list of facets."""
    def __init__(self, name : str):
        # The name of the mesh
        self.name = name
        # The vertices of the mesh
        self.vertices = []
        # The facets of the mesh
        self.facets = []

    def maybeAddVertex(self, vertex : Vertex, tolerance : float = 0.001):
        """ Find a vertex in the graph by proximity. If it is not found, add it.
        This is a linear search, so it is not efficient.
        Args:
            graph: The graph to search, as a list of Vertex
            vertex: The new vertex, as a Vertex
        Returns:
            The graph vertex if one already matched, or the new vertex that was added otherwise.
        """
        for graph_vertex in self.vertices:
            if Vector(graph_vertex.point - vertex.point).norm() < tolerance:
                return graph_vertex
        self.vertices.append(vertex)
        return vertex

    def buildFromSegments(self, segments : list[dict]):
        """ Build a graph of the mesh from a list of segment coordinates."""
        self.vertices = []
        for edge in segments:
            start = Vertex(edge['start'])
            end = Vertex(edge['end'])

            # Try to find the start and end vertices in the graph
            start = self.maybeAddVertex(start)
            end = self.maybeAddVertex(end)

            # Add the neighboors to the vertices
            start.edges.append(end)
            end.edges.append(start)

    def explorePath(self, path : list[ Vertex ], max_depth : int):
        """ This is used to explore a path within the mesh,
            in a recursive manner.
        Args:
            path: The current path, as a list of Vertex
            max_depth: The maximum depth of the recursion    
        """
        # If the last vertex is the same vertex as the first one, we have found a facet
        if len(path) > 2 and path[-1] is path[0]:
            self.facets.append(Facet(path[:-1]))
            return
        # If the path is maximum length, we are done
        if len(path) > max_depth:
            return
        # Otherwise we explore the neighboors
        for edge in path[-1].edges:
            # If this is the previous vertex, skip it
            if len(path) > 1 and edge == path[-2]:
                continue
            # Otherwise, explore this edge
            new_path = path + [edge]
            self.explorePath(new_path, max_depth)

    def removeRedundantQuads(self):
        """ Some polygons are actually made of two triangles.
            This removes them.
        """
        triangles = [facet for facet in self.facets if len(facet.vertices) == 3]
        quads = [facet for facet in self.facets if len(facet.vertices) == 4]
        new_quads = []
        for quad in quads:
            # Check if the quad is actually two triangles
            #triangle1 = Facet([quad.vertices[0], quad.vertices[1], quad.vertices[3]])
            #triangle2 = Facet([quad.vertices[1], quad.vertices[2], quad.vertices[3]])
            #triangle3 = Facet([quad.vertices[0], quad.vertices[1], quad.vertices[2]])
            #triangle4 = Facet([quad.vertices[0], quad.vertices[2], quad.vertices[3]])
            triangle1 = quad.subFacet([0, 1, 3])
            triangle2 = quad.subFacet([1, 2, 3])
            triangle3 = quad.subFacet([0, 1, 2])
            triangle4 = quad.subFacet([0, 2, 3])
            if (triangle1.isInList(triangles) and
                triangle2.isInList(triangles)):
                continue
            if (triangle3.isInList(triangles) and
                triangle4.isInList(triangles)):
                continue
            new_quads.append(quad)
        self.facets = triangles + new_quads

    def findFacets(self, max_edges : int):
        """ Find the facets in the mesh.

        Args:
            max_edges: The maximum number of edges the polygons should have
        """
        # We are going to find the polygon by walking the graph and finding cycles
        polygons = []
        for vertex in self.vertices:
            self.explorePath([vertex], max_edges)

        # We found many duplicates (the same polygon in reverse order,
        # or the same polygon with a different starting point).
        # Remove them here.
        facets_no_dupes = []
        for facet in self.facets:
            if facet.isInList(facets_no_dupes):
                continue
            facets_no_dupes.append(facet)
        self.facets = facets_no_dupes

    def __str__(self) -> str:
        num_edges = sum([len(vertex.edges) for vertex in self.vertices]) / 2
        num_triangles = len([facet for facet in self.facets if len(facet.vertices) == 3])
        num_quads = len([facet for facet in self.facets if len(facet.vertices) == 4])
        return 'Mesh %s : %d vertices, %d edges, %d facets (%d triangles, %d quads)' % (self.name, len(self.vertices), num_edges, len(self.facets), num_triangles, num_quads)

    def __repr__(self):
        return self.__str__()

def projectPolygon(polygon):
    """ Project a polygon on a plane. It will use the most suitable plane.
    Args:
        polygon: The polygon to project, as a list of Vertex
    Returns:
        The projected polygon, as a list of new Vertices
    """
    if len(polygon) < 3:
        raise ValueError('Polygon with less than 3 points cannot be projected !!!')
    # Start by fitting a plane to the polygon and projecting the points on it
    poly3d = [vertex.point for vertex in polygon]
    plane = Plane.best_fit(poly3d)
    poly3d_proj = [plane.project_point(point) for point in poly3d]
    # Now we need to define a coordinate frame. We'll use the first edge as x axis.
    x = Vector(poly3d[1] - poly3d[0]).unit()
    z = plane.normal
    y = z.cross(x)
    poly2d = []
    for point in poly3d_proj:
            v = Vector(point - plane.point)
            px = x.scalar_projection(v)
            py = y.scalar_projection(v)
            poly2d.append(Point([px, py]))

    # Find the best fit plane
    plane = Plane.best_fit([vertex.point for vertex in polygon])
    # Project the polygon on the plane
    return [plane.project_point(vertex.point) for vertex in polygon]

with open('/tmp/mesh_export.json') as file:
    json_data = json.load(file)

meshes = json_data['meshes']

for mesh in meshes:
    name = mesh['name']
    mesh_data = mesh['edges']
    print(name)
    print("-" * len(name))

    # Build a graph of this mesh
    mesh = Mesh(name)
    mesh.buildFromSegments(mesh_data)

    # Find the facets in the mesh3
    mesh.findFacets(4)

    # Remove the redundant quads (those that are made of two existing triangles)
    mesh.removeRedundantQuads()

    print(mesh)

    # Project the polygons on a plane
    #quads = [projectPolygon(quad) for quad in quads]


    print()