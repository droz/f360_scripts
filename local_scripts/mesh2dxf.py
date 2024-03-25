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
from HersheyFonts import HersheyFonts

@dataclass
class Parameters:
    """ This class contains all the geometrical parameters for this script."""
    # The gap between the bar and the panel
    bar_panel_gap_m : float = 0.0
    # The spacing between stitches
    nominal_stitch_spacing_m : float = 0.1
    # The distance from the ends towhere the first stitches should be
    end_stitch_distance_m : float = 0.03
    # The minimum spacing allowed between two stitches from different panels
    minimum_stitch_spacing_m : float = 0.01
    # The distance from the edge of the facet to the stitch holes
    stitch_hole_pullback_m : float = 0.006
    # The width of a stitch hole
    stitch_hole_width_m : float = 0.005
    # The height of a stitch hole
    stitch_hole_height_m : float = 0.003
    # The diameter of the rods used for the structure
    skeleton_rod_diameter_m : float = 0.00635
    # The clearance between the chamfer at the corner of the panels and
    # the corner vertex
    corner_chamfer_clearance_m : float = 0.00635
    # The vertical size of the font used for the labels
    label_font_height_m : float = 0.010

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
        # The facets that are using this edge
        self.facets = []

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
        # The plane that the facet lies on
        self.plane = None
        # The points of the facet in 3D, as a list of 3D points
        # This is a convenient link to the geometry of the vertices
        self.facet3d = None
        # The geometry of the actual panel in 3D, as a list of 3D points
        # It should be inset relative to the facet
        self.panel3d = None
        # The 2D projection of the facet on the plane, as a list of 2D points
        self.facet2d = None
        # The 2D projection of the panel on the plane, as a list of 2D points
        # It should be inset relative to the facet
        self.panel2d = None
        # The origin of the 2D representation of the facet
        self.origin2d = None
        # The 2D contours that we are going to cut,
        # as a list of list of 2D points
        self.contours2d = None
        # The segments forming the text label, as a list of (start, end) 2D points
        self.text_segments = None
        # The center of the 2D bounding box
        self.center2d = None
        # The index of the facet
        self.index = None
        # The list of chamfer sizes for each corner of the facet.
        # This distance is the distance along the edge that should be
        # cut to form the chamfer.
        self.chamfers = None
        # The list of stitch hole locations for each edge of the panel,
        #  as a list distances from the start vertex of the edge
        self.stitch_holes = None
        # The list of stitch relief locations for each edge of the panel,
        #  as a list distances from the start vertex of the edge
        self.stitch_reliefs = None

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
            The index of the edge in the facet, or None if it is not part of the facet
        """
        for i, (v1, v2) in enumerate(zip(self.vertices, self.vertices[1:] + self.vertices[:1])):
            if (v1 == edge.start and v2 == edge.end) or (v1 == edge.end and v2 == edge.start):
                return i
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
        if self.plane:
            self.plane.normal = - self.plane.normal
        # Check that none of the other properties of the face have been computed.
        # They would all get invalidated by the flip.
        if self.facet2d or self.facet3d or self.panel2d or self.panel3d or self.chamfers or self.edges or self.stitch_holes:
            raise ValueError('Cannot flip a facet that has already been processed !!!')

    def adjust2DPanelPoint(self, index):
        """ Adjust a 2d panel point to make the edges lengths match between 2d and 3d.
        Args:
            index: The index of the point to adjust
        """
        index_before = index - 1
        if index_before < 0:
            index_before += len(self.vertices)
        index_after = index + 1
        if index_after >= len(self.vertices):
            index_after -= len(self.vertices)
        p2d_b = self.panel2d[index_before]
        p2d   = self.panel2d[index]
        p2d_a = self.panel2d[index_after]
        p3d_b = self.panel3d[index_before]
        p3d   = self.panel3d[index]
        p3d_a = self.panel3d[index_after]
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

    def offsetPoints2D(self, edge_index : int, coords : list[Point]):
        """ Offset the 2D points relative to an edge
        Args:
            edge_index: The index of the edge to offset from
            offset_coords: The offset coordinates (x is the distance along the edge,
                           y is the offset distance inside the facet)
        """
        next_edge_index = (edge_index + 1) % len(self.vertices)
        edge_dir = Vector(self.facet2d[next_edge_index] - self.facet2d[edge_index]).unit()
        cross_dir = Vector([-edge_dir[1], edge_dir[0]])
        return [self.facet2d[edge_index] + edge_dir * coord[0] + cross_dir * coord[1] for coord in coords]

    def panelPoints3D(self, edge_index : int, coords : list[Point]):
        """ Compute the 3D positions of points on a panel given coordinates
            that are relative to an edge
        Args:
            edge_index: The index of the edge to offset from
            coords: The coordinates relative to the edge as a 2D point
                    x is the distance along the edge,
                    y is the offset distance inside the facet
        """
        next_edge_index = (edge_index + 1) % len(self.vertices)
        edge_dir = Vector(self.panel3d[next_edge_index] - self.panel3d[edge_index]).unit()
        cross_dir = self.plane.normal.cross(edge_dir)
        return [self.panel3d[edge_index] + edge_dir * coord[0] + cross_dir * coord[1] for coord in coords]

    def stitchPoints3D(self, edge : int, params : Parameters):
        """ For a given edge index, compute the 3D positions of the stitch points
        Args:
            edge_index: The index of the edge to offset from
            params: The parameters to use
        Returns:
            a list of pairs of 3D points, each pair being the hole and loop of a stitch
        """
        # We can get the hole positions directly from the stitch coordinates
        holes = self.panelPoints3D(edge, [(d, params.stitch_hole_pullback_m) for d in self.stitch_holes[edge]])
        # For the loops, we can project the holes on the edge itself
        p0 = self.facet3d[edge]
        p1 = self.facet3d[(edge + 1) % len(self.vertices)]
        edge_dir = Vector(p1 - p0).unit()
        edge_line = Line(p0, edge_dir)
        loops = [edge_line.project_point(hole) for hole in holes]
        return list(zip(holes, loops))

    def stitchReliefs3D(self, edge : int):
        """ For a given edge index, compute the 3D positions of the stitch reliefs
        Args:
            edge_index: The index of the edge to offset from
            params: The parameters to use
        Returns:
            a list of 3D points, each being the relief of a stitch
        """
        # We can get the relief positions directly from the stitch coordinates
        if self.stitch_reliefs[edge] is None:
            return []
        return self.panelPoints3D(edge, [(d, 0) for d in self.stitch_reliefs[edge]])

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

    def generate3DPanel(self, params : Parameters):
        """ Generate the 3D panel, with the correct inset
        Args:
            params: The parameters to use
        """
        inset = params.skeleton_rod_diameter_m / 2 + params.bar_panel_gap_m
        # Go over each vertex and move it toward the inside
        self.facet3d = [vertex.point for vertex in self.vertices]
        self.panel3d = []
        self.chamfers = []
        for p0, p1, p2 in zip(self.facet3d[-1:] + self.facet3d[:-1], self.facet3d, self.facet3d[1:] + self.facet3d[:1]):
            v0 = Vector(p0 - p1)
            v1 = Vector(p2 - p1)
            # We compute how much we need to move each point toward the inside
            v_bisector = (v0.unit() + v1.unit()).unit()
            angle = np.abs(v0.angle_between(v1))
            vertex_shift = inset / np.sin(angle / 2)
            self.panel3d.append(p1 + v_bisector * vertex_shift)
            # This is a good opportunity to compute the chamfer sizes.
            chamfer_cut = params.corner_chamfer_clearance_m - vertex_shift
            chamfer = max(0, chamfer_cut / np.cos(angle / 2))
            self.chamfers.append(chamfer)

    def generate2DPanel(self):
        """ Generate the 3D panel, with the correct inset """
        if len(self.vertices) < 3:
            raise ValueError('Polygon with less than 3 points cannot be projected !!!')
        # Start by fitting a plane to the polygon and projecting the points on it
        self.fitPlane()
        facet3d_proj = [self.plane.project_point(point) for point in self.facet3d]
        panel3d_proj = [self.plane.project_point(point) for point in self.panel3d]

        # Now we need to rotate the polygon to fit a consistent coordinate frame
        # We use the first edge of the facet as x axis.
        x = Vector(facet3d_proj[1] - facet3d_proj[0]).unit()
        z = self.plane.normal
        y = z.cross(x)
        self.facet2d = []
        self.panel2d = []
        for point in facet3d_proj:
            v = Vector(point - self.plane.point)
            px = x.scalar_projection(v)
            py = y.scalar_projection(v)
            self.facet2d.append(Point([px, py]))
        for point in panel3d_proj:
            v = Vector(point - self.plane.point)
            px = x.scalar_projection(v)
            py = y.scalar_projection(v)
            self.panel2d.append(Point([px, py]))

        # By doing the projection, if the facet was not planar,
        #  we ended up changing the length of the edges. For a quad,
        #  we can fix that by adjusting the points.
        if len(self.vertices) == 4:
            # Find the smallest diagonal, we will keep it constant and adjust the other points
            d1 = Vector(self.panel3d[2] - self.panel3d[0]).norm()
            d2 = Vector(self.panel3d[3] - self.panel3d[1]).norm()
            if d1 > d2:
                adjust_indexes = [0, 2]
            else:
                adjust_indexes = [1, 3]
            # Now we can adjust the points
            for index in adjust_indexes:
                self.adjust2DPanelPoint(index)

    def addStitchPoints(self, params : Parameters):
        """ Add stitch points to the facet.
        Args:
            params: The parameters to use
        """
        self.stitch_holes = []
        for p0, p1, p2, p3, chamfer1, chamfer2 in zip(self.panel2d[-1:] + self.panel2d[:-1],
                                                      self.panel2d,
                                                      self.panel2d[1:] + self.panel2d[:1],
                                                      self.panel2d[2:] + self.panel2d[:2],
                                                      self.chamfers,
                                                      self.chamfers[1:] + self.chamfers[:1]):
            length = Vector(p2 - p1).norm()
            # We need to figure out where we are going to place the first and last stitches.
            # This will depend on the angle of the edge with the previous and next edges.
            ve0 = Vector(p0 - p1).unit()
            ve1 = Vector(p2 - p1).unit()
            ve2 = Vector(p2 - p3).unit()
            angle1 = np.abs(ve0.angle_between(ve1))
            angle2 = np.abs(ve1.angle_between(ve2))
            pullback = params.stitch_hole_pullback_m + params.stitch_hole_height_m
            d1 = pullback / np.tan(angle1 / 2) + 2 * params.stitch_hole_width_m
            d2 = pullback / np.tan(angle2 / 2) + 2 * params.stitch_hole_width_m
            d1 = max(d1, params.end_stitch_distance_m, chamfer1)
            d2 = max(d2, params.end_stitch_distance_m, chamfer2)

            # If the edge is too short to even add one pair of stitches, we don't add any
            length_left = length - d1 - d2
            if length_left <= 2 * params.minimum_stitch_spacing_m:
                self.stitch_holes.append([])
                continue
            num_stitches = int(np.floor(length_left / params.nominal_stitch_spacing_m + 0.5))
            # If the edge is too short for two pairs of end stitches we only add one
            if num_stitches < 1:
                self.stitch_holes.append([(d1 + length - d2) / 2])
                continue
            # OK, we have room for two pairs of end stitches, plus other stitches in the middle,
            # space them regularly
            self.stitch_holes.append(list(np.linspace(d1, length - d2, num_stitches + 1)))
        self.stitch_reliefs = [None] * len(self.vertices)

    def generateContours(self, params : Parameters):
        """ Generate 2D contours from the 2D polygon
        Args:
            params: The parameters to use
        """
        num_vertices = len(self.vertices)
        inset = params.skeleton_rod_diameter_m / 2 + params.bar_panel_gap_m
        pullback = params.stitch_hole_pullback_m
        shift = params.ab_stitch_spacing_m / 2
        width = params.stitch_hole_width_m
        height = params.stitch_hole_height_m
        # We start by figuring out how much each edge should be cut by at the
        # end to create a chamfer that guarantees the correct clearance to the corner
        cut_distances = []
        for p0, p1, p2 in zip(self.facet2d[-1:] + self.facet2d[:-1], self.facet2d, self.facet2d[1:] + self.facet2d[:1]):
            v0 = Vector(p1 - p0)
            v1 = Vector(p1 - p2)
            angle = np.abs(v0.angle_between(v1))
            chamfer_distance = params.corner_chamfer_clearance_m / np.cos(angle / 2)
            # The rod diameter compensation may also end up making the new corner
            # further away than the clearance, check for that here
            rod_pullback_distance = inset / np.tan(angle / 2)
            cut_distances.append(max(chamfer_distance, rod_pullback_distance))


        self.stitch_holes = [[]]*num_vertices

        panel_contour = []
        self.contours2d = []
        for i in range(num_vertices):
            length = Vector(self.facet2d[(i+1) % num_vertices] - self.facet2d[i]).norm()
            d0 = cut_distances[i]
            d1 = cut_distances[(i+1) % num_vertices]
            # Compute the two end-points
            p0, p1 = self.offsetPoints2D(i, [(0 + d0, inset), (length - d1, inset)])
            # Then the stitch clearance points for the edge
            stitches = self.stitch_holes[i]
            if not stitches:
                panel_contour += [p0, p1]
                continue
            ps = [None] * len(stitches) * 4
            ps[0::4] = self.offsetPoints2D(i, [(stitch * length - shift - width / 2, inset) for stitch in stitches])
            ps[1::4] = self.offsetPoints2D(i, [(stitch * length - shift - width / 2, inset + height) for stitch in stitches])
            ps[2::4] = self.offsetPoints2D(i, [(stitch * length - shift + width / 2, inset + height) for stitch in stitches])
            ps[3::4] = self.offsetPoints2D(i, [(stitch * length - shift + width / 2, inset) for stitch in stitches])
            panel_contour += [p0] + ps + [p1]
            # Then the holes for the stitches
            for stitch in stitches:
                hole_coords = [(stitch * length + shift - width / 2, inset + pullback),
                               (stitch * length + shift - width / 2, inset + pullback + height),
                               (stitch * length + shift + width / 2, inset + pullback + height),
                               (stitch * length + shift + width / 2, inset + pullback)]
                ph = self.offsetPoints2D(i, hole_coords)
                self.contours2d.append(ph)

        # We may have ended up with zero length segments, remove them here
        panel_contour = [panel_contour[i] for i in range(len(panel_contour)) if panel_contour[i].distance_point(panel_contour[(i + 1) % len(panel_contour)]) > 1e-6]

        self.contours2d.append(panel_contour)

    def generateLabel(self, mesh_name, font, font_size):
        """ Generate a label for the facet
        Args:
            mesh_name: The name of the mesh
            font: The font to use
            font_size: The size of the font
        """
        self.text_segments = []
        words = (mesh_name + ' ' + str(self.index)).split()
        for i, word in enumerate(words):
            segments = list(font.lines_for_text(word))
            min_x = min([min(start[0], end[0]) for start, end in segments])
            max_x = max([max(start[0], end[0]) for start, end in segments])
            text_pos = Point((-(min_x + max_x) / 2, ((len(words)) / 2 - i - 1) * 1.1)) * font_size
            for start, end in font.lines_for_text(word):
                self.text_segments.append([Point(start) * font_size + text_pos, Point(end) * font_size + text_pos])
            

    def __str__(self):
        return 'Facet : %d vertices' % len(self.vertices)

    def __repr__(self):
        return self.__str__()

class Mesh:
    """ This class is used to represent a mesh as a graph of vertices and a list of facets."""
    def __init__(self, name : str, params : Parameters = None):
        # The name of the mesh
        self.name = name
        # The parameters
        if params is None:
            self.params = Parameters()
        else:
            self.params = params
        # The vertices of the mesh
        self.vertices = []
        # The edges in the mesh
        self.edges = []
        # The facets of the mesh
        self.facets = []
        # Some error points that we want to flag on the display
        self.error_points = []

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
            facet.edges = [None] * len(facet.vertices)
            for edge in self.edges:
                index = facet.containsEdge(edge)
                if index is not None:
                    facet.edges[index] = edge
                    edge.facets.append(facet)

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
        """ Plot the mesh """
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
            if len(facet.panel3d) == 3:
                ptA = facet.panel3d[0]
                ptB = facet.panel3d[1]
                ptC = facet.panel3d[2]
                xs = np.array([[ptA[0], ptB[0]], [ptC[0], np.nan]])
                ys = np.array([[ptA[1], ptB[1]], [ptC[1], np.nan]])
                zs = np.array([[ptA[2], ptB[2]], [ptC[2], np.nan]])
            if len(facet.vertices) == 4:
                ptA = facet.panel3d[0]
                ptB = facet.panel3d[1]
                ptC = facet.panel3d[3]
                ptD = facet.panel3d[2]
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
        stitches = []
        for facet in self.facets:
            if not facet.stitch_holes:
                continue
            for edge_index in range(len(facet.vertices)):
                stitches += facet.stitchPoints3D(edge_index, self.params)
        xs = list(chain.from_iterable([stitch[0][0], stitch[1][0], np.nan] for stitch in stitches))
        ys = list(chain.from_iterable([stitch[0][1], stitch[1][1], np.nan] for stitch in stitches))
        zs = list(chain.from_iterable([stitch[0][2], stitch[1][2], np.nan] for stitch in stitches))
        ax.plot(xs, ys, zs, 'g')
        reliefs = []
        for facet in self.facets:
            if not facet.stitch_reliefs:
                continue
            for edge_index in range(len(facet.vertices)):
                reliefs += facet.stitchReliefs3D(edge_index)
        xs = [point[0] for point in reliefs]
        ys = [point[1] for point in reliefs]
        zs = [point[2] for point in reliefs]
        ax.scatter(xs, ys, zs, c='b', marker='.')
        # The error points
        xs = [point[0] for point in self.error_points]
        ys = [point[1] for point in self.error_points]
        zs = [point[2] for point in self.error_points]
        ax.scatter(xs, ys, zs, c='r', marker='X')
        # Setup the viewport
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
            # The 2D facet if it exists
            if facet.facet2d:
                xs = [point[0] + facet.origin2d[0] for point in facet.facet2d]
                ys = [point[1] + facet.origin2d[1] for point in facet.facet2d] 
                ax.plot(xs + [xs[0]], ys + [ys[0]], 'r--', alpha = 0.6, linewidth = 1)
            # The 2D panel if it exists
            if facet.facet2d:
                xs = [point[0] + facet.origin2d[0] for point in facet.panel2d]
                ys = [point[1] + facet.origin2d[1] for point in facet.panel2d] 
                ax.plot(xs + [xs[0]], ys + [ys[0]], 'g:', alpha = 0.6, linewidth = 1)
            # The label if it exists
            if facet.text_segments:
                xs = list(chain.from_iterable([segment[0][0] + facet.origin2d[0], segment[1][0] + facet.origin2d[0], np.nan] for segment in facet.text_segments))
                ys = list(chain.from_iterable([segment[0][1] + facet.origin2d[1], segment[1][1] + facet.origin2d[1], np.nan] for segment in facet.text_segments))
                ax.plot(xs, ys, 'k')
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
            if num_concave_facets < num_convex_facets:
                for facet in self.facets:
                    facet.flip()
        print("Oriented %d sub-mesh." % num_connected_meshes)

    def indexFacets(self):
        """ Assign indexes to the facets. We need an indexing scheme
        that is as stable as possible when the mesh is modified."""
        # Sort facets based on x, y, and z coordinates
        # The key is a heuristic. It is trying to deal with facets that
        # have very close center coordinates, but are not exactly the same.
        self.facets.sort(key=lambda facet: -facet.center()[1] * 1e6 + facet.center()[2] * 1e3 + facet.center()[0])
        for index, facet in enumerate(self.facets):
            facet.index = index
        print('Assigned %d indexes to the facets' % len(self.facets))

    def generatePanels(self):
        """ Compute the shape of the panels, compensated for the rod diameter """
        for facet in self.facets:
            # First the 3D inset
            facet.generate3DPanel(self.params)
            # Then the 2D projection
            facet.generate2DPanel()
        print('Generated %d panels' % len(self.facets))

    def generate2DLayout(self):
        """ Generate a 2D layout for all the panels of the mesh."""
        # Generate offsets to layout the 2D polygons on a grid
        rows = np.ceil(np.sqrt(len(self.facets)))
        # Extract the widths and heights of the facets. Also compute the center
        # of their bounding boxes
        facet_widths = []
        facet_heights = []
        for facet in self.facets:
            xs = [point[0] for point in facet.facet2d]
            ys = [point[1] for point in facet.facet2d]
            facet_widths.append(max(xs) - min(xs))
            facet_heights.append(max(ys) - min(ys))
            facet.center2d = Point(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2))
        max_width = max(facet_widths)
        max_height = max(facet_heights)
        row = 0
        col = 0
        for facet in self.facets:
            facet.origin2d = Point([max_width * (col + 0.5) * 1.05, max_height * (row + 0.5) * 1.05]) - facet.center2d
            col += 1
            if col == rows:
                col = 0
                row += 1
        return max_width * rows * 1.05, max_height * rows * 1.05

    def generate2DContours(self):
        """ Create 2D contours from the 3D facets.
        Args:
            pullback: The pullback to apply to the 2D contours
        Returns:
            width: The width of the 2D layout
            height: The height of the 2D layout
        """
        # Prepare the font
        font = HersheyFonts()
        font.load_default_font()
        font.normalize_rendering(1.0)
        for facet in self.facets:
            # Then we generate the contours
            facet.generateContours(self.params)
            # And finally the label
            facet.generateLabel(self.name, font, self.params.label_font_height_m)
        num_contours = sum([len(facet.contours2d) for facet in self.facets])
        num_segments = sum([len(contours) for facet in self.facets for contours in facet.contours2d])
        print('Generated %d 2D contours with %d segments' % (num_contours, num_segments))

    def addStitchPoints(self):
        """ Add stitch points to the facets in the mesh. """
        # We go over each facet to add stitch points
        for facet in self.facets:
            facet.addStitchPoints(self.params)
        # We now need to make sure that no pair of stitches overlap.
        num_adjustments = 0
        for facet1 in self.facets:
            for index1, edge in enumerate(facet1.edges):
                if len(edge.facets) < 2:
                    continue
                facet2 = edge.facets[1] if edge.facets[0] is facet1 else edge.facets[0]
                index2 = [index for index, edge2 in enumerate(facet2.edges) if edge2 is edge]
                if len(index2) != 1:
                    raise ValueError('Edge not found in facet')
                index2 = index2[0]
                stitches1 = facet1.stitchPoints3D(index1, self.params)
                stitches2 = facet2.stitchPoints3D(index2, self.params)
                p1 = facet1.facet3d[index1]
                p2 = facet2.facet3d[index2]
                edge_center = (p1 + p2) / 2
                for i1, (_, loop1) in enumerate(stitches1):
                    for i2, (_, loop2) in enumerate(stitches2):
                        dist = p1.distance_point(loop2) - p1.distance_point(loop1)
                        if np.abs(dist) < self.params.minimum_stitch_spacing_m - 1e-6:
                            num_adjustments += 1
                            # We are going to move the stitch that is the closest to the center of the edge
                            d1c = edge_center.distance_point(loop1)
                            d2c = edge_center.distance_point(loop2)
                            if d1c < d2c:
                                # We are moving d1
                                facet1.stitch_holes[index1][i1] += dist - np.sign(dist) * self.params.minimum_stitch_spacing_m
                            else:
                                # We are moving d2
                                facet2.stitch_holes[index2][i2] += dist - np.sign(dist) * self.params.minimum_stitch_spacing_m
                # At this point we should have no overlaps. We transfer the location of the stitches to the other facet
                stitches1 = facet1.stitchPoints3D(index1, self.params)
                panel2_p1 = facet2.panel3d[index2]
                panel2_p2 = facet2.panel3d[(index2 + 1) % len(facet2.vertices)]
                panel2_edge_dir = Vector(panel2_p2 - panel2_p1).unit()
                facet2.stitch_reliefs[index2] = [(loop1 - panel2_p1).dot(panel2_edge_dir) for _, loop1 in stitches1]

        num_holes = sum([len(holes) for facet in self.facets for holes in facet.stitch_holes if holes is not None])
        num_reliefs = sum([len(reliefs) for facet in self.facets for reliefs in facet.stitch_reliefs if reliefs is not None])
        print('Added %d stitch holes and %d stitch reliefs, Adjusted %d to avoid overlap' % (num_holes, num_reliefs, num_adjustments))

    def exportToDxf(self, dxf_doc : ezdxf.document.Drawing, dxf_offset : Point):
        """ Export the mesh to a DXF file.
        Args:
            dxf_doc: The DXF document to export to
            dxf_offset: The offset to apply to the mesh
        """
        # Go over each facet and add the 2D contours to the DXF.
        base_name = self.name.lower().replace(' ', '_')
        dxf_modelspace = dxf_doc.modelspace()
        cut_layer = base_name + '_cuts'
        mark_layer = base_name + '_marks'
        dxf_doc.layers.add(cut_layer)
        dxf_doc.layers.add(mark_layer)
        for facet in mesh.facets:
            if not facet.contours2d:
                continue
            layer_name = base_name + '_' + str(facet.index)
            origin = facet.origin2d + dxf_offset
            for contour in facet.contours2d:
                dxf_modelspace.add_lwpolyline([(point + origin) * 1000 for point in contour], close=True, dxfattribs={'layer': cut_layer})
            for segment in facet.text_segments:
                dxf_modelspace.add_line((segment[0] + origin) * 1000, (segment[1] + origin) * 1000, dxfattribs={'layer': mark_layer})

    def __str__(self) -> str:
        num_triangles = len([facet for facet in self.facets if len(facet.vertices) == 3])
        num_quads = len([facet for facet in self.facets if len(facet.vertices) == 4])
        return 'Mesh %s : %d vertices, %d edges, %d facets (%d triangles, %d quads)' % (
            self.name, len(self.vertices), len(self.edges), len(self.facets), num_triangles, num_quads)

    def __repr__(self):
        return self.__str__()

parser = argparse.ArgumentParser(description='Convert a mesh to a DXF file')
parser.add_argument('json', type=str, help='The input JSON file')
parser.add_argument('-d', '--dxf', type=str, help='The output DXF file')
parser.add_argument('-p', '--plot', action='store_true', help='Plot the mesh')
parser.add_argument('-m', '--mesh', type=str, action='append', help='The name of the mesh/meshes to convert')
parser.add_argument('--gap', type=str, action='append', help='Defines an extra gap between the bars and '
                    'the panels for a given mesh. Format is "mesh_name:gap", gap is in meters')
args = parser.parse_args()

# Read the json file
with open(args.json) as file:
    json_data = json.load(file)
meshes = json_data['meshes']

# Start a DXF file if required
if args.dxf:
    dxf_doc = ezdxf.new('R2010', setup=True)

# Process each mesh
dxf_width = 0
for mesh in meshes:
    name = mesh['name']
    mesh_data = mesh['edges']

    if args.mesh and name not in args.mesh:
        continue

    print(name)
    print("-" * len(name))

    # Create a mesh object and assign parameters
    mesh = Mesh(name) 
    for gap_str in args.gap or []:
        mesh_name, gap_str = gap_str.split(':')
        if mesh_name == mesh.name:
            mesh.params.bar_panel_gap_m = float(gap_str)
            break

    # Build a graph of this mesh
    mesh.buildFromSegments(mesh_data)

    # Find the facets in the mesh3
    mesh.findFacets(4)

    # Remove the redundant quads (those that are made of two existing triangles)
    mesh.removeRedundantQuads()

    # Orient the facets so that the normal is consistent between faces
    mesh.orientFacets()

    # Assign indexes to the facets
    mesh.indexFacets()

    # Find the edges in the mesh
    mesh.findEdges()

    # Generate the panels (with the correct inset)
    mesh.generatePanels()

    # Generate the 2D layout
    width, height = mesh.generate2DLayout()

    # Add stitch points to all the edges
    mesh.addStitchPoints()

    # Export DXF if required
    if args.dxf:
        mesh.exportToDxf(dxf_doc, Point((dxf_width, 0)))
        dxf_width += width

    # Plot the mesh if required
    if args.plot:
        mesh.plot()

    # Current state of the mesh
    print(mesh)
    print()

# Save the DXF file if required
if args.dxf:
    dxf_doc.saveas(args.dxf)

if args.plot:
    plt.show()

