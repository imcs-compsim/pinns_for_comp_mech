import gmsh
import sys
import numpy as np


class QuarterCirclewithHole(object):
    def __init__(
        self, center, inner_radius, outer_radius, mesh_size=0.15, gmsh_options=None
    ):
        self.center = center
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.mesh_size = mesh_size
        self.gmsh_options = gmsh_options

    def generateGmshModel(self, visualize_mesh=False):
        """
        Generates the quarter of a circle including a hole.

        Parameters
        ----------
        visualize_mesh : boolean
            a booelan value to show the mesh using Gmsh or not
        Returns
        -------
        gmsh_model: Object
            gmsh model
        """
        # Parameters
        xc = self.center[0]
        yc = self.center[1]
        zc = self.center[2]
        r1 = self.inner_radius
        r2 = self.outer_radius

        # Mesh size.
        lcar = self.mesh_size * r1

        # create gmsh model instance
        gmsh_model = gmsh.model
        factory = gmsh_model.occ

        # initialize gmsh
        gmsh.initialize(sys.argv)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lcar)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lcar)

        if self.gmsh_options:
            for command, value in self.gmsh_options.items():
                if type(value).__name__ == "str":
                    gmsh.option.setString(command, value)
                else:
                    gmsh.option.setNumber(command, value)

        # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1);

        gmsh_model.add("QuarterCirclewithHole")

        # Actually allows for a filled ellipse.
        # create the small disk
        s1 = factory.addDisk(xc, yc, zc, r1, r1)
        # create the large disk
        s2 = factory.addDisk(xc, yc, zc, r2, r2)
        # create the rectangle
        s3 = factory.addRectangle(xc, yc, zc, r2, r2)
        # substract the small disk from the large one
        s4, ss4 = factory.cut([(2, s2)], [(2, s1)])
        # intersect it with the rectangle
        factory.intersect(s4, [(2, s3)])

        gmsh_model.occ.synchronize()

        # generate mesh
        gmsh_model.mesh.generate(2)

        if visualize_mesh:
            if "-nopopup" not in sys.argv:
                gmsh.fltk.run()

        return gmsh_model


class CirclewithHole(object):
    def __init__(
        self, center, inner_radius, outer_radius, mesh_size=0.15, gmsh_options=None
    ):
        self.center = center
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.mesh_size = mesh_size
        self.gmsh_options = gmsh_options

    def generateGmshModel(self, visualize_mesh=False):
        """
        Generates a circle including a hole.

        Parameters
        ----------
        visualize_mesh : boolean
            a booelan value to show the mesh using Gmsh or not
        Returns
        -------
        gmsh_model: Object
            gmsh model
        """

        # Parameters
        xc = self.center[0]
        yc = self.center[1]
        zc = self.center[2]
        r1 = self.inner_radius
        r2 = self.outer_radius

        # Mesh size.
        lcar = self.mesh_size * r1

        # create gmsh model instance
        gmsh_model = gmsh.model
        factory = gmsh_model.occ

        # initialize gmsh
        gmsh.initialize(sys.argv)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lcar)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lcar)

        if self.gmsh_options:
            for command, value in self.gmsh_options.items():
                if type(value).__name__ == "str":
                    gmsh.option.setString(command, value)
                else:
                    gmsh.option.setNumber(command, value)

        # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1);

        gmsh_model.add("CirclewithHole")

        # Actually allows for a filled ellipse.
        # create the small disk
        s1 = factory.addDisk(xc, yc, zc, r1, r1)
        # create the large disk
        s2 = factory.addDisk(xc, yc, zc, r2, r2)
        # substract the small disk from the large one
        factory.cut([(2, s2)], [(2, s1)])
        # intersect it with the rectangle

        gmsh_model.occ.synchronize()

        # generate mesh
        gmsh_model.mesh.generate(2)

        if visualize_mesh:
            if "-nopopup" not in sys.argv:
                gmsh.fltk.run()

        return gmsh_model


class Block_2D(object):
    def __init__(
        self, coord_left_corner, coord_right_corner, mesh_size=0.15, gmsh_options=None
    ):
        self.coord_left_corner = coord_left_corner
        self.coord_right_corner = coord_right_corner
        self.mesh_size = mesh_size
        self.gmsh_options = gmsh_options

    def generateGmshModel(self, visualize_mesh=False):
        """
        Generates a circle including a hole.

        Parameters
        ----------
        visualize_mesh : boolean
            a booelan value to show the mesh using Gmsh or not
        Returns
        -------
        gmsh_model: Object
            gmsh model
        """

        # Parameters
        x0 = self.coord_left_corner[0]
        y0 = self.coord_left_corner[1]
        x1 = self.coord_right_corner[0]
        y1 = self.coord_right_corner[1]
        assert x1 > x0
        assert y1 > y0
        l = x1 - x0
        h = y1 - y0
        # Mesh size.
        lcar = self.mesh_size * min(h, l)

        # create gmsh model instance
        gmsh_model = gmsh.model
        factory = gmsh_model.occ

        # initialize gmsh
        gmsh.initialize(sys.argv)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lcar)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lcar)

        if self.gmsh_options:
            for command, value in self.gmsh_options.items():
                if type(value).__name__ == "str":
                    gmsh.option.setString(command, value)
                else:
                    gmsh.option.setNumber(command, value)

        # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1);

        gmsh_model.add("Rectangle")

        factory.addRectangle(x0, y0, 0, l, h)

        gmsh_model.occ.synchronize()

        # generate mesh
        gmsh_model.mesh.generate(2)

        if visualize_mesh:
            if "-nopopup" not in sys.argv:
                gmsh.fltk.run()

        return gmsh_model


class Block_3D(object):
    def __init__(
        self, coord_left_corner, coord_right_corner, mesh_size=0.15, gmsh_options=None
    ):
        self.coord_left_corner = coord_left_corner
        self.coord_right_corner = coord_right_corner
        self.mesh_size = mesh_size
        self.gmsh_options = gmsh_options

    def generateGmshModel(self, visualize_mesh=False):
        """
        Generates a 3D block.

        Parameters
        ----------
        visualize_mesh : boolean
            a booelan value to show the mesh using Gmsh or not
        Returns
        -------
        gmsh_model: Object
            gmsh model
        """

        # Parameters
        x0 = self.coord_left_corner[0]
        y0 = self.coord_left_corner[1]
        z0 = self.coord_left_corner[2]
        x1 = self.coord_right_corner[0]
        y1 = self.coord_right_corner[1]
        z1 = self.coord_right_corner[2]
        assert x1 > x0
        assert y1 > y0
        assert z1 > z0
        l = x1 - x0
        h = y1 - y0
        w = z1 - z0
        # Mesh size.
        lcar = self.mesh_size * min(h, l, w)

        # create gmsh model instance
        gmsh_model = gmsh.model
        factory = gmsh_model.occ

        # initialize gmsh
        gmsh.initialize(sys.argv)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lcar)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lcar)

        if self.gmsh_options:
            for command, value in self.gmsh_options.items():
                if type(value).__name__ == "str":
                    gmsh.option.setString(command, value)
                else:
                    gmsh.option.setNumber(command, value)

        # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1);

        gmsh_model.add("Box")

        factory.addBox(x0, y0, z0, l, h, w)

        gmsh_model.occ.synchronize()

        # generate mesh
        gmsh_model.mesh.generate(3)

        if visualize_mesh:
            if "-nopopup" not in sys.argv:
                gmsh.fltk.run()

        return gmsh_model


class Rectangle_4PointBending(object):
    def __init__(
        self,
        l_beam,
        h_beam,
        region_size_dict,
        mesh_size=0.15,
        refine_factor=12,
        gmsh_options=None,
    ):
        self.l_beam = l_beam
        self.h_beam = h_beam
        self.region_size_dict = region_size_dict
        self.mesh_size = mesh_size
        self.gmsh_options = gmsh_options
        self.refine_factor = refine_factor

    def generateGmshModel(self, visualize_mesh=False):
        """
        Generates a rectangle with partitioned mesh.

        Parameters
        ----------
        visualize_mesh : boolean
            a booelan value to show the mesh using Gmsh or not
        Returns
        -------
        gmsh_model: Object
            gmsh model
        """

        # Mesh size.
        lc = self.mesh_size  # * min(self.l_beam,self.h_beam)

        # create gmsh model instance
        gmsh_model = gmsh.model

        # initialize gmsh
        gmsh.initialize(sys.argv)

        if self.gmsh_options:
            for command, value in self.gmsh_options.items():
                if type(value).__name__ == "str":
                    gmsh.option.setString(command, value)
                else:
                    gmsh.option.setNumber(command, value)

        gmsh_model.add("Beam with partitioned mesh")

        #  corner points
        c1 = gmsh_model.geo.addPoint(0, 0, 0, lc)  # left bottom, origin
        c2 = gmsh_model.geo.addPoint(self.l_beam, 0, 0, lc)  # right bottom
        c3 = gmsh_model.geo.addPoint(self.l_beam, self.h_beam, 0, lc)  # right top
        c4 = gmsh_model.geo.addPoint(0, self.h_beam, 0, lc)  # left top

        # location points
        p1 = gmsh_model.geo.addPoint(
            self.region_size_dict["r1"]["start"], 0, 0, lc
        )  # p1
        p2 = gmsh_model.geo.addPoint(
            self.region_size_dict["r1"]["start"]
            + self.region_size_dict["r1"]["increment"],
            0,
            0,
            lc,
        )  # p2
        p3 = gmsh_model.geo.addPoint(
            self.region_size_dict["r2"]["start"], 0, 0, lc
        )  # p3
        p4 = gmsh_model.geo.addPoint(
            self.region_size_dict["r2"]["start"]
            + self.region_size_dict["r2"]["increment"],
            0,
            0,
            lc,
        )  # p4
        p5 = gmsh_model.geo.addPoint(
            self.region_size_dict["r3"]["start"], self.h_beam, 0, lc
        )  # p5
        p6 = gmsh_model.geo.addPoint(
            self.region_size_dict["r3"]["start"]
            + self.region_size_dict["r3"]["increment"],
            self.h_beam,
            0,
            lc,
        )  # p6
        p7 = gmsh_model.geo.addPoint(
            self.region_size_dict["r4"]["start"], self.h_beam, 0, lc
        )  # p7
        p8 = gmsh_model.geo.addPoint(
            self.region_size_dict["r4"]["start"]
            + self.region_size_dict["r4"]["increment"],
            self.h_beam,
            0,
            lc,
        )  # p8

        # generate lines (use counter-clockwise direction)
        gmsh_model.geo.addLine(c1, p1)
        gmsh_model.geo.addLine(p1, p2)
        gmsh_model.geo.addLine(p2, p3)
        gmsh_model.geo.addLine(p3, p4)
        gmsh_model.geo.addLine(p4, c2)
        gmsh_model.geo.addLine(c2, c3)
        gmsh_model.geo.addLine(c3, p8)
        gmsh_model.geo.addLine(p8, p7)
        gmsh_model.geo.addLine(p7, p6)
        gmsh_model.geo.addLine(p6, p5)
        gmsh_model.geo.addLine(p5, c4)
        gmsh_model.geo.addLine(c4, c1)

        # The third elementary entity is the surface. In order to define a simple
        # rectangular surface from the four curves defined above, a curve loop has first
        # to be defined.
        curve_loop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        gmsh_model.geo.addCurveLoop(curve_loop, 1)

        # We can then define the surface as a list of curve loops (only one here,
        # representing the external contour, since there are no holes--see `t4.py' for
        # an example of a surface with a hole):
        gmsh_model.geo.addPlaneSurface([1], 1)

        # Before they can be meshed (and, more generally, before they can be used by API
        # functions outside of the built-in CAD kernel functions), the CAD entities must
        # be synchronized with the Gmsh model
        gmsh_model.geo.synchronize()

        if self.refine_factor:
            gmsh_model.mesh.field.add("Distance", 1)
            gmsh_model.mesh.field.setNumbers(1, "CurvesList", [2, 4, 8, 10])
            gmsh_model.mesh.field.setNumber(1, "Sampling", 100)

            gmsh_model.mesh.field.add("Threshold", 2)
            gmsh_model.mesh.field.setNumber(2, "InField", 1)
            gmsh_model.mesh.field.setNumber(2, "SizeMin", lc / self.refine_factor)
            gmsh_model.mesh.field.setNumber(2, "SizeMax", lc)
            gmsh_model.mesh.field.setNumber(2, "DistMin", 0.01)
            gmsh_model.mesh.field.setNumber(2, "DistMax", 0.02)

            gmsh_model.mesh.field.add("Min", 3)
            gmsh_model.mesh.field.setNumbers(3, "FieldsList", [2, 3])

            gmsh_model.mesh.field.setAsBackgroundMesh(3)

        # generate mesh
        gmsh_model.mesh.generate(2)

        if visualize_mesh:
            if "-nopopup" not in sys.argv:
                gmsh.fltk.run()

        return gmsh_model


class Rectangle_4PointBendingCentered(object):
    def __init__(
        self,
        coord_left_corner,
        coord_right_corner,
        region_size_dict,
        mesh_size=0.15,
        refine_factor=12,
        gmsh_options=None,
    ):
        self.coord_left_corner = coord_left_corner
        self.coord_right_corner = coord_right_corner
        self.region_size_dict = region_size_dict
        self.mesh_size = mesh_size
        self.gmsh_options = gmsh_options
        self.refine_factor = refine_factor

    def generateGmshModel(self, visualize_mesh=False):
        """
        Generates a rectangle with partitioned mesh.

        Parameters
        ----------
        visualize_mesh : boolean
            a booelan value to show the mesh using Gmsh or not
        Returns
        -------
        gmsh_model: Object
            gmsh model
        """

        # Mesh size.
        lc = self.mesh_size  # * min(self.l_beam,self.h_beam)

        # create gmsh model instance
        gmsh_model = gmsh.model

        # initialize gmsh
        gmsh.initialize(sys.argv)

        if self.gmsh_options:
            for command, value in self.gmsh_options.items():
                if type(value).__name__ == "str":
                    gmsh.option.setString(command, value)
                else:
                    gmsh.option.setNumber(command, value)

        gmsh_model.add("Beam with partitioned mesh")

        x0 = self.coord_left_corner[0]
        y0 = self.coord_left_corner[1]
        x1 = self.coord_right_corner[0]
        y1 = self.coord_right_corner[1]

        #  corner points
        c1 = gmsh_model.geo.addPoint(x0, y0, 0, lc)  # left bottom, origin
        c2 = gmsh_model.geo.addPoint(x1, y0, 0, lc)  # right bottom
        c3 = gmsh_model.geo.addPoint(x1, y1, 0, lc)  # right top
        c4 = gmsh_model.geo.addPoint(x0, y1, 0, lc)  # left top

        # location points
        p1 = gmsh_model.geo.addPoint(
            self.region_size_dict["r1"]["center"]
            - self.region_size_dict["r1"]["deviation"],
            y0,
            0,
            lc,
        )  # p1
        p2 = gmsh_model.geo.addPoint(
            self.region_size_dict["r1"]["center"]
            + self.region_size_dict["r1"]["deviation"],
            y0,
            0,
            lc,
        )  # p2
        p3 = gmsh_model.geo.addPoint(
            self.region_size_dict["r2"]["center"]
            - self.region_size_dict["r2"]["deviation"],
            y0,
            0,
            lc,
        )  # p3
        p4 = gmsh_model.geo.addPoint(
            self.region_size_dict["r2"]["center"]
            + self.region_size_dict["r2"]["deviation"],
            y0,
            0,
            lc,
        )  # p4
        p5 = gmsh_model.geo.addPoint(
            self.region_size_dict["r3"]["center"]
            - self.region_size_dict["r3"]["deviation"],
            y1,
            0,
            lc,
        )  # p5
        p6 = gmsh_model.geo.addPoint(
            self.region_size_dict["r3"]["center"]
            + self.region_size_dict["r3"]["deviation"],
            y1,
            0,
            lc,
        )  # p6
        p7 = gmsh_model.geo.addPoint(
            self.region_size_dict["r4"]["center"]
            - self.region_size_dict["r4"]["deviation"],
            y1,
            0,
            lc,
        )  # p7
        p8 = gmsh_model.geo.addPoint(
            self.region_size_dict["r4"]["center"]
            + self.region_size_dict["r4"]["deviation"],
            y1,
            0,
            lc,
        )  # p8

        # generate lines (use counter-clockwise direction)
        gmsh_model.geo.addLine(c1, p1)
        gmsh_model.geo.addLine(p1, p2)
        gmsh_model.geo.addLine(p2, p3)
        gmsh_model.geo.addLine(p3, p4)
        gmsh_model.geo.addLine(p4, c2)
        gmsh_model.geo.addLine(c2, c3)
        gmsh_model.geo.addLine(c3, p8)
        gmsh_model.geo.addLine(p8, p7)
        gmsh_model.geo.addLine(p7, p6)
        gmsh_model.geo.addLine(p6, p5)
        gmsh_model.geo.addLine(p5, c4)
        gmsh_model.geo.addLine(c4, c1)

        # The third elementary entity is the surface. In order to define a simple
        # rectangular surface from the four curves defined above, a curve loop has first
        # to be defined.
        curve_loop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        gmsh_model.geo.addCurveLoop(curve_loop, 1)

        # We can then define the surface as a list of curve loops (only one here,
        # representing the external contour, since there are no holes--see `t4.py' for
        # an example of a surface with a hole):
        gmsh_model.geo.addPlaneSurface([1], 1)

        # Before they can be meshed (and, more generally, before they can be used by API
        # functions outside of the built-in CAD kernel functions), the CAD entities must
        # be synchronized with the Gmsh model
        gmsh_model.geo.synchronize()

        if self.refine_factor:
            gmsh_model.mesh.field.add("Distance", 1)
            gmsh_model.mesh.field.setNumbers(1, "CurvesList", [2, 4, 8, 10])
            gmsh_model.mesh.field.setNumber(1, "Sampling", 100)

            gmsh_model.mesh.field.add("Threshold", 2)
            gmsh_model.mesh.field.setNumber(2, "InField", 1)
            gmsh_model.mesh.field.setNumber(2, "SizeMin", lc / self.refine_factor)
            gmsh_model.mesh.field.setNumber(2, "SizeMax", lc)
            gmsh_model.mesh.field.setNumber(2, "DistMin", 0.01)
            gmsh_model.mesh.field.setNumber(2, "DistMax", 0.02)

            gmsh_model.mesh.field.add("Min", 3)
            gmsh_model.mesh.field.setNumbers(3, "FieldsList", [2, 3])

            gmsh_model.mesh.field.setAsBackgroundMesh(3)

        # generate mesh
        gmsh_model.mesh.generate(2)

        if visualize_mesh:
            if "-nopopup" not in sys.argv:
                gmsh.fltk.run()

        return gmsh_model


class QuarterDisc(object):
    def __init__(
        self,
        radius,
        center,
        angle=None,
        refine_times=None,
        mesh_size=0.15,
        gmsh_options=None,
    ):
        self.radius = radius
        self.center = center
        if angle:
            self.angle_rad = np.pi * angle / 180
        else:
            self.angle_rad = angle
        self.refine_times = refine_times
        self.mesh_size = mesh_size
        self.gmsh_options = gmsh_options

    def generateGmshModel(self, visualize_mesh=False):
        """
        Generates a quarter disc with partition or without it.

        Parameters
        ----------
        visualize_mesh : boolean
            a booelan value to show the mesh using Gmsh or not
        Returns
        -------
        gmsh_model: Object
            gmsh model
        """

        # Mesh size.
        lcar = self.mesh_size * self.radius

        # create gmsh model instance
        gmsh_model = gmsh.model

        # initialize gmsh
        gmsh.initialize(sys.argv)

        if self.gmsh_options:
            for command, value in self.gmsh_options.items():
                if type(value).__name__ == "str":
                    gmsh.option.setString(command, value)
                else:
                    gmsh.option.setNumber(command, value)

        x_loc_p3 = None
        y_loc_p3 = None

        gmsh_model.add("Rectangle")

        p0 = gmsh_model.geo.addPoint(self.center[0], self.center[1], 0, lcar, 1)
        p1 = gmsh_model.geo.addPoint(
            self.center[0] - self.radius, self.center[1], 0, lcar, 2
        )
        p2 = gmsh_model.geo.addPoint(
            self.center[0], self.center[1] - self.radius, 0, lcar, 3
        )
        if self.angle_rad:
            p3 = gmsh_model.geo.addPoint(
                self.radius * np.cos(self.angle_rad),
                self.radius * np.sin(self.angle_rad),
                0,
                lcar,
                4,
            )
            x_loc_p3 = self.radius * np.cos(self.angle_rad)
            y_loc_p3 = self.radius * np.sin(self.angle_rad)

        c1 = gmsh_model.geo.addLine(p0, p1)
        c2 = gmsh_model.geo.addLine(p2, p0)
        if self.angle_rad:
            c3 = gmsh_model.geo.addCircleArc(p1, p0, p3)
            c4 = gmsh_model.geo.addCircleArc(p3, p0, p2)

            gmsh_model.geo.addCurveLoop([c1, c2, c3, c4], 1)
        else:
            c3 = gmsh_model.geo.addCircleArc(p1, p0, p2)
            gmsh_model.geo.addCurveLoop([c1, c2, c3], 1)

        gmsh_model.geo.addPlaneSurface([1], 1)

        gmsh_model.geo.synchronize()

        if self.refine_times and self.angle_rad:
            gmsh_model.mesh.field.add("Distance", 1)
            gmsh_model.mesh.field.setNumbers(1, "CurvesList", [c4])
            gmsh_model.mesh.field.setNumber(1, "Sampling", 100)

            gmsh_model.mesh.field.add("Threshold", 2)
            gmsh_model.mesh.field.setNumber(2, "InField", 1)
            gmsh_model.mesh.field.setNumber(2, "SizeMin", lcar / self.refine_times)
            gmsh_model.mesh.field.setNumber(2, "SizeMax", lcar)
            gmsh_model.mesh.field.setNumber(2, "DistMin", self.radius / 10000)
            gmsh_model.mesh.field.setNumber(2, "DistMax", self.radius / 1000)

            gmsh_model.mesh.field.add("Min", 3)
            gmsh_model.mesh.field.setNumbers(3, "FieldsList", [2, 3])

            gmsh_model.mesh.field.setAsBackgroundMesh(3)

        # generate mesh
        gmsh_model.mesh.generate(2)

        if visualize_mesh:
            if "-nopopup" not in sys.argv:
                gmsh.fltk.run()

        return gmsh_model, x_loc_p3, y_loc_p3
