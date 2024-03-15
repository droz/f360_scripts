""" Open a mesh exported by ExportMeshSketches.py, analyse
    it and create a DXF file for all the panels """


import json
from itertools import chain
from skspatial.objects import Plane, Vector, Point, Line
import ezdxf
from matplotlib import pyplot as plt
import numpy as np
import argparse
from dataclasses import dataclass

@dataclass
class Parameters:
    """ This class contains all the geometrical parameters for this script."""
    # The spacing between stitches
    stitch_spacing_m : float = 0.1
    # The distance from the ends towhere the first stitches should be
    end_stitch_distance_m : float = 0.03
    # The spacing between stitches for the panels on each side of the edge
    ab_stitch_spacing_m : float = 0.01
    # The distance from the edge of the facet to the stitch holes
    stitch_edge_pullback_m : float = 0.01
    # The diameter of the rods used for the structure
    skeleton_rod_diameter_m : float = 0.00635
    # The clearance between the chamfer at the corner of the panels and
    # the corner vertex
    corner_chamfer_clearance_m : float = 0.01

class Vertex:
    """ This class is used to represent a vertex in the mesh graph"""
    def __init__(self, xyz : list[float]):
        # The point coordinates
        self.point = Point(xyz)
        # The neighboor vertices
        self.neighboors = []

    def __str__(self):
        return 'Vertex : (%f, %f, %f), %d neighboors' % (self.point[0], self.point[1], self.point[2], len(self.neighboors))

    def __repr__(self):
        return self.__str__()

class Edge:
    """ This class is used to represent an edge in the mesh graph"""
    def __init__(self, start : Vertex, end : Vertex):
        # The start vertex
        self.start = start
        # The end vertex
        self.end = end

    def isSame(self, edge : 'Edge'):
        """ Check if another edge is actually the same as this one.
        Args:
            edge: The edge to compare against
        Returns:
            True if the edges are the same, False otherwise
        """
        return (self.start == edge.start and self.end == edge.end) or (self.start == edge.end and self.end == edge.start)

    def isInList(self, edges : list['Edge']):
        """ Check if a facet is part of a list of edges, regardless of orientation.
        Args:
            edges: The list of edges to check against
        Returns:
            True if this edge is in the list, False otherwise
        """
        return any([self.isSame(list_edge) for list_edge in edges])

    def __str__(self):
        return 'Edge : %s -> %s' % (self.start, self.end)

    def __repr__(self):
        return self.__str__()

class Facet:
    """ This class is used to represent a facet in the mesh """
    def __init__(self, vertices : list[Vertex]):
        # The vertices of the facet
        self.vertices = vertices
        # The edges of the facet, as a list of edges
        self.edges = None
        # The projected polygon, as a list of 2D points
        self.polygon2d = None
        # The origin of the 2D representation of the facet
        self.origin2d = None
        # The 2D contours that we are going to cut,
        # as a list of list of 2D points
        self.contours2d = None
        # The index of the facet
        self.index = None
        # The list of stitch locations for each edge of the facet,
        #  as a list of lists of fractional distances (0 is the start of the edge, 1 the end)
        # Each location corresponds to a pair of (stitch hole, stitch relief).
        # The location is the point in-between these two.
        self.stitches = None

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
        """ Check if a facet is part of a list of facets, regardless of start or orientation.
        Args:
            facets: The list of facets to check against
        Returns:
            True if this facet is in the list, False otherwise
        """
        return any([self.isSame(list_facet) for list_facet in facets])

    def shareVertex(self, facet : 'Facet'):
        """ Check if two facets share a vertex.
        Args:
            facet: The facet to compare against
        Returns:
            True if the facets share a vertex, False otherwise
        """
        return set(self.vertices) & set(facet.vertices)

    def shareSide(self, facet : 'Facet'):
        """ Check if two facets share a side.
        Args:
            facet: The facet to compare against
        Returns:
            share_side: True if the facets share a side, False otherwise
            same_orientation: True if the facets are oriented the same way, False otherwise
        """
        for v1l, v2l in zip(self.vertices, self.vertices[1:] + self.vertices[:1]):
            for v1, v2 in zip(facet.vertices, facet.vertices[1:] + facet.vertices[:1]):
                if (v1l == v1 and v2l == v2):
                    return True, False
                if (v1l == v2 and v2l == v1):
                    return True, True
        return False, None

    def containsEdge(self, edge : Edge):
        """ Check if an edge is part of the facet.
        Args:
            edge: The edge to compare against
        Returns:
            True if the edge is part of the facet, False otherwise
        """
        for v1, v2 in zip(self.vertices, self.vertices[1:] + self.vertices[:1]):
            if (v1 == edge.start and v2 == edge.end) or (v1 == edge.end and v2 == edge.start):
                return True
        return False, None

    def findEdge(self, vertex1 : Vertex, vertex2 : Vertex):
        """ Find an edge between two vertices.
        Args:
            vertex1: The first vertex
            vertex2: The second vertex
        Returns:
            The edge if it exists, None otherwise
            The orientation of the edge, True if it goes from vertex1 to vertex2, False otherwise
        """
        for edge in self.edges:
            if edge.start == vertex1 and edge.end == vertex2:
                return edge, True
            if edge.start == vertex2 and edge.end == vertex1:
                return edge, False
        return None

    def center(self):
        """ Return the center of the facet """
        x = self.plane.point[0]
        y = self.plane.point[1]
        z = self.plane.point[2]
        return x, y, z

    def flip(self):
        """ Flip the orientation of the facet """
        self.vertices.reverse()

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
        self.fitPlane()
        poly3d = [vertex.point for vertex in self.vertices]
        poly3d_proj = [self.plane.project_point(point) for point in poly3d]

        # Now we need to rotate the polygon to fit a consistent coordinate frame
        # We use the first edge as x axis.
        x = Vector(poly3d_proj[1] - poly3d_proj[0]).unit()
        z = self.plane.normal
        y = z.cross(x)
        self.polygon2d = []
        for point in poly3d_proj:
            v = Vector(point - self.plane.point)
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

    def fitPlane(self):
        """ Fit a plane to the facet """
        if len(self.vertices) < 3:
            raise ValueError('Facet with less than 3 points cannot be projected !!!')
        # Fit a plane to the facet
        poly3d = [vertex.point for vertex in self.vertices]
        self.plane = Plane.best_fit(poly3d)
        # Make sure the normal is pointing outwards
        local_normal_fit = []
        for v0, v1, v2 in zip(self.vertices, self.vertices[1:] + self.vertices[:1], self.vertices[2:] + self.vertices[:2]):
            d10 = Vector(v0.point - v1.point)
            d12 = Vector(v2.point - v1.point)
            normal = d12.cross(d10).unit()
            normal_fit = self.plane.normal.dot(normal)
            local_normal_fit.append(normal_fit)
        if sum(local_normal_fit) < 0:
            self.plane.normal = - self.plane.normal

    def addStitchPoints(self, params : Parameters):
        """ Add stitch points to the facet.
        Args:
            params: The parameters to use
        """
        self.stitches = []
        for v1, v2 in zip(self.vertices, self.vertices[1:] + self.vertices[:1]):
            length = Vector(v2.point - v1.point).norm()
            # If the edge is too short to even add one pair of stitches, we don't add any
            if length < 2 * params.ab_stitch_spacing_m:
                self.stitches.append([])
                continue    
            length_left = length - 2 * params.end_stitch_distance_m
            num_stitches = np.floor(length_left / params.stitch_spacing_m + 0.5)
            # If the edge is too short for two pairs of end stitches we only add one
            if num_stitches < 1:
                self.stitches.append([0.5])
                continue
            # OK, we have room for two pairs of end stitches, plus other stitches in the middle,
            # compute how many and attempt to space them regularly
            stitch_spacing_actual_m = length_left / num_stitches
            self.stitches.append(list(np.arange(params.end_stitch_distance_m, length, stitch_spacing_actual_m) / length))

    def offsetPoints2D(self, edge_index : int, along_edge_m : list[float], across_edge_m : list[float]):
        """ Offset the 2D points relative to an edge
        Args:
            edge_index: The index of the edge to offset from
            along_edge_m: The distance along the edge, in meters
            across_edge_m: The distance across the edge, in meters
        """
        next_edge_index = (edge_index + 1) % len(self.vertices)
        edge_dir = Vector(self.polygon2d[next_edge_index] - self.polygon2d[edge_index]).unit()
        cross_dir = Vector([-edge_dir[1], edge_dir[0]])
        assert len(along_edge_m) == len(across_edge_m)
        return [self.polygon2d[edge_index] + edge_dir * along_edge_m[i] + cross_dir * across_edge_m[i] for i in range(len(along_edge_m))]

    def offsetPoints3D(self, edge_index : int, along_edge_m : list[float], across_edge_m : list[float]):
        """ Offset the 3D points relative to an edge
        Args:
            edge_index: The index of the edge to offset from
            along_edge_m: The distance along the edge, in meters
            across_edge_m: The distance across the edge, in meters
        """
        next_edge_index = (edge_index + 1) % len(self.vertices)
        edge_dir = Vector(self.vertices[next_edge_index].point - self.vertices[edge_index].point).unit()
        cross_dir = self.plane.normal.cross(edge_dir)
        assert len(along_edge_m) == len(across_edge_m)
        return [self.vertices[edge_index].point + edge_dir * along_edge_m[i] + cross_dir * across_edge_m[i] for i in range(len(along_edge_m))]

    def generateContours(self):
        """ Generate 2D contours from the 2D polygon """
        num_vertices = len(self.vertices)
        # We start by figuring out how much each edge should be cut by at the
        # end to create a chamfer that guarantees the correct clearance to the corner
        cut_distances = []
        for i in range(len(self.vertices)):
            i_previous = i - 1 if i > 0 else num_vertices - 1
            i_next = i + 1 if i < num_vertices - 1 else 0
            v0 = Vector(self.polygon2d[i] - self.polygon2d[i_previous])
            v1 = Vector(self.polygon2d[i] - self.polygon2d[i_next])
            angle = np.abs(v0.angle_between(v1))
            chamfer_distance = params.corner_chamfer_clearance_m / np.cos(angle / 2)
            # The rod diameter compensation may also end up making the new corner
            # further away than the clearance, check for that here
            rod_pullback_distance = params.skeleton_rod_diameter_m / 2 / np.tan(angle / 2)
            cut_distances.append(max(chamfer_distance, rod_pullback_distance))

        panel_contour = []
        for i in range(num_vertices):
            l = Vector(self.polygon2d[(i+1) % num_vertices] - self.polygon2d[i]).norm()
            d0 = cut_distances[i]
            d1 = cut_distances[(i+1) % num_vertices]
            # Compute the two end-points
            p0, p1 = self.offsetPoints2D(i, [0 + d0, l - d1], [params.skeleton_rod_diameter_m / 2] * 2)
            # Then the stitch clearance points for the edge
            #edge = 


        #     for edge, orientation in facet.edges:
        #         edge_vec = Vector(edge.end.point - edge.start.point)
        #         stitch_dir = facet.plane.normal.cross(edge_vec.unit())
        #         if orientation:
        #             stitches = edge.stitches_a
        #             stitch_dir = facet.plane.normal.cross(edge_vec.unit())
        #         else:
        #             stitches = edge.stitches_b
        #             stitch_dir = edge_vec.unit().cross(facet.plane.normal)
        #         if stitches is not None:
        #             dots += [edge.start.point + l * edge_vec + stitch_dir * self.params.stitch_edge_pullback_m for l in stitches]
        # ax.scatter([dot[0] for dot in dots],
        #            [dot[1] for dot in dots],
        #            [dot[2] for dot in dots], c='g', marker='o', s = 5)
        #     panel_contour += [p0, p1]
        self.contours2d = [panel_contour]

    def __str__(self):
        return 'Facet : %d vertices' % len(self.vertices)

    def __repr__(self):
        return self.__str__()

class Mesh:
    """ This class is used to represent a mesh as a graph of vertices and a list of facets."""
    def __init__(self, name : str, params : Parameters):
        # The name of the mesh
        self.name = name
        # The parameters
        self.params = params
        # The vertices of the mesh
        self.vertices = []
        # The edges in the mesh
        self.edges = []
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
            start.neighboors.append(end)
            end.neighboors.append(start)
        print('Built graph with %d vertices' % len(self.vertices))

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
        for edge in path[-1].neighboors:
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
        print('Removed %d redundant quads' % (len(quads) - len(new_quads)))

    def findEdges(self):
        """ Find the edges in the mesh """
        self.edges = []
        for vertex in self.vertices:
            for neighboor in vertex.neighboors:
                edge = Edge(vertex, neighboor)
                if not edge.isInList(self.edges):
                    self.edges.append(edge)
        # Associate edges to each facets
        for facet in self.facets:
            facet.edges = []
            for edge in self.edges:
                if facet.containsEdge(edge):
                    facet.edges.append(edge)

        print('Found %d edges' % len(self.edges))

    def findFacets(self, max_edges : int):
        """ Find the facets in the mesh.

        Args:
            max_edges: The maximum number of edges the polygons should have
        """
        # We are going to find the polygon by walking the graph and finding cycles
        self.facets = []
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

        print('Found %d facets' % len(self.facets))

    def plot(self):
        """ Plot the mesh in 3D """
        fig = plt.figure()
        fig.suptitle(self.name)
        fig.set_figwidth(10)

        # 3D view
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        # Get the bounding box of the mesh
        min_x = min([vertex.point[0] for vertex in self.vertices])
        max_x = max([vertex.point[0] for vertex in self.vertices])
        min_y = min([vertex.point[1] for vertex in self.vertices])
        max_y = max([vertex.point[1] for vertex in self.vertices])
        min_z = min([vertex.point[2] for vertex in self.vertices])
        max_z = max([vertex.point[2] for vertex in self.vertices])
        max_range = max([max_x - min_x, max_y - min_y, max_z - min_z])

        # The facets
        for facet in self.facets:
            if len(facet.vertices) == 3:
                ptA = facet.vertices[0].point
                ptB = facet.vertices[1].point
                ptC = facet.vertices[2].point
                xs = np.array([[ptA[0], ptB[0]], [ptC[0], np.nan]])
                ys = np.array([[ptA[1], ptB[1]], [ptC[1], np.nan]])
                zs = np.array([[ptA[2], ptB[2]], [ptC[2], np.nan]])
            if len(facet.vertices) == 4:
                ptA = facet.vertices[0].point
                ptB = facet.vertices[1].point
                ptC = facet.vertices[3].point
                ptD = facet.vertices[2].point
                xs = np.array([[ptA[0], ptB[0]], [ptC[0], ptD[0]]])
                ys = np.array([[ptA[1], ptB[1]], [ptC[1], ptD[1]]])
                zs = np.array([[ptA[2], ptB[2]], [ptC[2], ptD[2]]])
            if len(facet.vertices) > 4:
                continue
            ax.plot_surface(xs, ys, zs, color='r', shade=False, alpha=0.4)
        # The normals
        for facet in self.facets:
            facet.fitPlane()
        ax.quiver([facet.plane.point[0] for facet in self.facets],
                  [facet.plane.point[1] for facet in self.facets],
                  [facet.plane.point[2] for facet in self.facets],
                  [facet.plane.normal[0] for facet in self.facets],
                  [facet.plane.normal[1] for facet in self.facets],
                  [facet.plane.normal[2] for facet in self.facets], length=max_range * 0.05, color='b', alpha = 0.3)
        # The facet indices if they exists
        for facet in self.facets:
            if facet.index is not None:
                ax.text(facet.plane.point[0], facet.plane.point[1], facet.plane.point[2], '%d' % facet.index, color='k', horizontalalignment='center', verticalalignment='center', weight='bold')
        # The edges
        xs = list(chain.from_iterable([edge.start.point[0], edge.end.point[0], np.nan] for edge in self.edges))
        ys = list(chain.from_iterable([edge.start.point[1], edge.end.point[1], np.nan] for edge in self.edges))
        zs = list(chain.from_iterable([edge.start.point[2], edge.end.point[2], np.nan] for edge in self.edges))
        ax.plot(xs, ys, zs, 'k')
        # The stitches
        dots = []
        for facet in self.facets:
            if not facet.stitches:
                continue
            for i in range(len(facet.vertices)):
                v1 = facet.vertices[i]
                v2 = facet.vertices[(i+1) % len(facet.vertices)]
                length = Vector(v2.point - v1.point).norm()
                stitches = facet.stitches[i]
                if not stitches:
                    continue
                dots += facet.offsetPoints3D(i,
                                             [r * length + self.params.ab_stitch_spacing_m / 2 for r in stitches],
                                             [self.params.stitch_edge_pullback_m] * len(stitches))
        ax.scatter([dot[0] for dot in dots],
                   [dot[1] for dot in dots],
                   [dot[2] for dot in dots], c='g', marker='o', s = 5)
        # The vertices
        ax.scatter([vertex.point[0] for vertex in self.vertices],
                   [vertex.point[1] for vertex in self.vertices],
                   [vertex.point[2] for vertex in self.vertices], 'o')
        ax.set_box_aspect(None, zoom=1.7)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)
        ax.axis('equal')
        ax.axis('off')
        ax.azim = 45

        # 2D view
        ax = fig.add_subplot(1, 2, 2)
        for facet in self.facets:
            # The 2D polygon if it exists
            if facet.polygon2d:
                xs = [point[0] + facet.origin2d[0] for point in facet.polygon2d]
                ys = [point[1] + facet.origin2d[1] for point in facet.polygon2d] 
                ax.plot(xs + [xs[0]], ys + [ys[0]], 'r')
            # The index if it exists
            if facet.index is not None:
                ax.text(facet.origin2d[0], facet.origin2d[1], '%d' % facet.index, color='k', horizontalalignment='center', verticalalignment='center')
            # The contours if they exist
            if facet.contours2d:
                for contour in facet.contours2d:
                    if contour:
                        xs = [point[0] + facet.origin2d[0] for point in contour]
                        ys = [point[1] + facet.origin2d[1] for point in contour]
                        ax.plot(xs + [xs[0]], ys + [ys[0]], 'b')
        ax.axis('equal')

        plt.draw()

    def orientFacets(self):
        """ Orient the facets so that the normal is consistent between faces """
        if not self.facets:
            return
        unoriented_facets = set(self.facets)
        num_connected_meshes = 0
        while unoriented_facets:
            # We take one face at random and explore all the faces connected to it
            num_connected_meshes += 1
            oriented_facets = set([unoriented_facets.pop()])
            while oriented_facets:
                facet = oriented_facets.pop()
                # Find all the neighboors and orient them the same way as the current facet
                new_oriented_facets = set()
                for neighboor in unoriented_facets:
                    share_side, same_orientation = facet.shareSide(neighboor)
                    if not share_side:
                        continue
                    if not same_orientation:
                        neighboor.flip()
                    new_oriented_facets.add(neighboor)
                    oriented_facets.add(neighboor)
                unoriented_facets = unoriented_facets - new_oriented_facets
            # Try to make the mesh as concave as possible.
            # For each facet, we count how many other mesh points are
            #  on each side of the facet. That will tell us if that face
            #  is locally more concave or convex. If there are more concave
            #  than convex facets, we flip all the facets.
            num_concave_facets = 0
            num_convex_facets = 0
            for facet in self.facets:
                facet.fitPlane()
                num_concave_vertices = 0
                num_convex_vertices = 0
                for neighboor_facet in self.facets:
                    if not facet.shareVertex(neighboor_facet):
                        continue
                    for vertex in neighboor_facet.vertices:
                        if vertex in facet.vertices:
                            continue
                        if facet.plane.distance_point_signed(vertex.point) > 0:
                            num_concave_vertices += 1
                        else:
                            num_convex_vertices += 1
                if num_concave_vertices > num_convex_vertices:
                    num_concave_facets += 1
                else:
                    num_convex_facets += 1
            if num_concave_facets > num_convex_facets:
                for facet in self.facets:
                    facet.flip()
        print("Oriented %d sub-mesh." % num_connected_meshes)

    def indexFacets(self):
        """ Assign indexes to the facets """
        # Sort facets based on x, y, and z coordinates
        self.facets.sort(key=lambda facet: facet.center())
        for index, facet in enumerate(self.facets):
            facet.index = index

    def generate2DContours(self):
        """ Create 2D contours from the 3D facets """
        for facet in self.facets:
            # Start by projecting all the facets
            facet.project()
            # Then we generate the contours
            facet.generateContours()

        # Generate offsets to layout the 2D polygons on a grid
        rows = np.ceil(np.sqrt(len(self.facets)))
        # Extract the widths and heights of the facets
        facet_widths = []
        facet_heights = []
        for facet in self.facets:
            xs = [point[0] for point in facet.polygon2d]
            ys = [point[1] for point in facet.polygon2d]
            facet_widths.append(max(xs) - min(xs))
            facet_heights.append(max(ys) - min(ys))
        max_width = max(facet_widths)
        max_height = max(facet_heights)
        row = 0
        col = 0
        for facet in self.facets:
            facet.origin2d = Point([max_width * col * 1.05, max_height * row * 1.05])
            col += 1
            if col == rows:
                col = 0
                row += 1

    def addStitchPoints(self):
        """ Add stitch points to the facets in the mesh. """
        # We go over each facet to ad stitch points
        for facet in self.facets:
            facet.addStitchPoints(self.params)

    def writeDxf(self, filename : str):
        dxf = ezdxf.new('R2010')
        msp = dxf.modelspace()
        for facet in mesh.facets:
            if not facet.polygon2d:
                continue
            msp.add_lwpolyline([(point + facet.origin2d) * 1000 for point in facet.polygon2d], close=True)
            msp.add_lwpolyline([(point * 0.5 + facet.origin2d) * 1000 for point in facet.polygon2d][::-1], close=True)
        dxf.saveas(filename)

    def __str__(self) -> str:
        num_triangles = len([facet for facet in self.facets if len(facet.vertices) == 3])
        num_quads = len([facet for facet in self.facets if len(facet.vertices) == 4])
        num_poly2d = len([facet for facet in self.facets if facet.polygon2d])
        return 'Mesh %s : %d vertices, %d edges, %d facets (%d triangles, %d quads), %d 2D polygons' % (
            self.name, len(self.vertices), len(self.edges), len(self.facets), num_triangles, num_quads, num_poly2d)

    def __repr__(self):
        return self.__str__()

parser = argparse.ArgumentParser(description='Convert a mesh to a DXF file')
parser.add_argument('json', type=str, help='The input JSON file')
parser.add_argument('-d', '--dxf', type=str, help='The output DXF file')
parser.add_argument('-p', '--plot', action='store_true', help='Plot the mesh')
parser.add_argument('-m', '--mesh_name', type=str, action='append', help='The name of the mesh/meshes to convert')
args = parser.parse_args()

# Default parameters
params = Parameters()

# Read the json file
with open(args.json) as file:
    json_data = json.load(file)
meshes = json_data['meshes']

# Process each mesh
for mesh in meshes:
    name = mesh['name']
    mesh_data = mesh['edges']

    if args.mesh_name and name not in args.mesh_name:
        continue

    print(name)
    print("-" * len(name))

    # Build a graph of this mesh
    mesh = Mesh(name, params) 
    mesh.buildFromSegments(mesh_data)

    # Find the facets in the mesh3
    mesh.findFacets(4)

    # Find the edges in the mesh
    mesh.findEdges()

    # Remove the redundant quads (those that are made of two existing triangles)
    mesh.removeRedundantQuads()

    # Orient the facets so that the normal is consistent between faces
    mesh.orientFacets()

    # Add stitch points to all the edges
    mesh.addStitchPoints()

    # Assign indexes to the facets
    mesh.indexFacets()

    # Generate all the 2D facets
    mesh.generate2DContours()

    # Export DXF if required
    if args.dxf:
        mesh.writeDxf(args.dxf)

    # Current state of the mesh
    print(mesh)

    # Plot the mesh if required
    if args.plot:
        mesh.plot()


    print()

if args.plot:
    plt.show()