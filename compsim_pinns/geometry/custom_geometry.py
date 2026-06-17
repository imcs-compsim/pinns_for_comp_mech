"""Custom Gmsh geometry wrappers for DeepXDE sampling and element data."""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from deepxde import config
from deepxde.geometry.geometry import Geometry


def _is_in_tolerance(provided, target, tolerance=1e-5):
    """Check whether provided points match any target point within tolerance."""

    contained_rows = np.all(
        np.isclose(provided[:, None, :], target, rtol=tolerance, atol=tolerance),
        axis=2,
    )
    return np.any(contained_rows, axis=1)


def _compute_tangentials(normal_boundaries):
    """Compute two unit tangential vectors for each 3D boundary normal."""

    normals = np.asarray(normal_boundaries, dtype=float).reshape(-1, 3)
    if normals.shape[0] == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(
        normals,
        normal_norms,
        out=np.zeros_like(normals),
        where=normal_norms > 0,
    )

    t_1 = np.zeros_like(normals)
    xy_norms = np.linalg.norm(normals[:, :2], axis=1)
    z_axis_mask = xy_norms <= 1e-12
    general_mask = ~z_axis_mask

    t_1[z_axis_mask] = np.array([0.0, 1.0, 0.0])
    t_1[general_mask, 0] = normals[general_mask, 1] / xy_norms[general_mask]
    t_1[general_mask, 1] = -normals[general_mask, 0] / xy_norms[general_mask]

    t_2 = np.cross(normals, t_1)
    t_2_norms = np.linalg.norm(t_2, axis=1, keepdims=True)
    t_2 = np.divide(
        t_2,
        t_2_norms,
        out=np.zeros_like(t_2),
        where=t_2_norms > 0,
    )

    if not np.allclose(np.sum(normals * t_1, axis=1), 0.0):
        raise ValueError(
            "Normal vectors and first tangential vectors are not orthogonal."
        )
    if not np.allclose(np.sum(normals * t_2, axis=1), 0.0):
        raise ValueError(
            "Normal vectors and second tangential vectors are not orthogonal."
        )
    if not np.allclose(np.sum(t_1 * t_2, axis=1), 0.0):
        raise ValueError("First and second tangential vectors are not orthogonal.")

    return t_1, t_2


class GmshGeometry3D(Geometry):
    """Represent a 3D Gmsh geometry for DeepXDE boundary and mesh queries."""

    def __init__(self, gmsh_model, external_dim_size=None, target_surface_ids=None):
        """Initialize a 3D Gmsh geometry wrapper."""

        self.gmsh_model = gmsh_model
        self.external_dim_size = external_dim_size
        self.target_surface_ids = target_surface_ids
        self.boundary_normal_global = self.fun_boundary_normal_global()
        self.bbox = np.array([1, 1, 1])
        super(GmshGeometry3D, self).__init__(3, self.bbox, 1)

    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

        node_tag, node_coords_all, parametricCoord = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )[0]

        _, _, node_coords_xyz_inside = self.order_coordinates(
            node_coords_all, node_tag, node_tag_inside=node_tag_inside
        )

        if self.external_dim_size:
            node_coords_xyz_inside = self.add_external_dim(node_coords_xyz_inside)

        return _is_in_tolerance(x, node_coords_xyz_inside)

    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

        node_tag, node_coords_all, _ = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        _, node_coords_xyz_boundary, _ = self.order_coordinates(
            node_coords_all, node_tag, node_tag_boundary, node_tag_inside
        )

        if self.external_dim_size:
            node_coords_xyz_boundary = self.add_external_dim(node_coords_xyz_boundary)

        return _is_in_tolerance(x, node_coords_xyz_boundary)

    def boundary_normal(self, x):
        """Slice the unit normal at x for Neumann or Robin boundary conditions."""

        n, _, _, uniq = self.boundary_normal_global

        if self.external_dim_size:
            x = np.delete(x, -1, 1)

        mask = []
        for x_i in x:
            mask.extend(np.where(np.all(np.isclose(x_i, uniq), axis=1))[0].tolist())

        return n[mask]

    def boundary_tangential_1(self, x):
        """Slice the first unit tangential vector at x for Neumann or Robin boundary conditions."""

        _, t_1, _, uniq = self.boundary_normal_global

        if self.external_dim_size:
            x = np.delete(x, -1, 1)

        mask = []
        for x_i in x:
            mask.extend(np.where(np.all(np.isclose(x_i, uniq), axis=1))[0].tolist())

        return t_1[mask]

    def boundary_tangential_2(self, x):
        """Slice the second unit tangential vector at x for Neumann or Robin boundary conditions."""

        _, _, t_2, uniq = self.boundary_normal_global

        if self.external_dim_size:
            x = np.delete(x, -1, 1)

        mask = []
        for x_i in x:
            mask.extend(np.where(np.all(np.isclose(x_i, uniq), axis=1))[0].tolist())

        return t_2[mask]

    def fun_boundary_normal_global(self):
        """Compute geometry boundary normals and tangential vectors."""

        node_tag_boundary = []
        node_coords_xyz_boundary = []
        normal_boundary = []
        surface_id = []
        border = {}
        start = 0

        for geometry_entity_pair in self.gmsh_model.getEntities():
            if geometry_entity_pair[0] == 2:  # if it is a surface
                s_tag = geometry_entity_pair[1]
                surface_name = "surface_" + str(s_tag)

                # get node tag, coordinates and parametric coordinates form geometry
                node_tag, node_coords, parametric_coords = (
                    self.gmsh_model.mesh.getNodes(
                        dim=2, tag=s_tag, includeBoundary=True
                    )
                )  # dim, curve tag, includeBoundary
                # get normals
                normals = self.gmsh_model.getNormal(s_tag, parametric_coords)

                # reshape coordinates and first derivative
                node_coords = node_coords.reshape(-1, 3)
                normals = normals.reshape(-1, 3)

                surface_id_intermediate = [s_tag] * node_coords.shape[0]
                surface_id.extend(surface_id_intermediate)

                # store intermediate quantities in the global variables
                node_tag_boundary.extend(node_tag.tolist())
                node_coords_xyz_boundary.extend(node_coords.tolist())
                normal_boundary.extend(normals.tolist())

        # convert them into numpy array
        node_tag_boundary = np.array(node_tag_boundary)
        node_coords_xyz_boundary = np.array(node_coords_xyz_boundary)
        normal_boundary = np.array(normal_boundary)
        surface_id = np.array(surface_id)

        # calculate the tangential vector components.
        tangential_boundary_1, tangential_boundary_2 = self.compute_tangentials(
            normal_boundary
        )

        # get the unique nodes
        u, idx, c = np.unique(node_tag_boundary, return_counts=True, return_index=True)
        # get the repeated nodes that have more than 1 boundary normal
        repeated_node_tag = u[c > 1]

        if self.target_surface_ids:
            for repeated_node_id in repeated_node_tag:
                # create mask to get the repeated nodes in the global boundary node list
                repeated_node_mask = node_tag_boundary == repeated_node_id
                # get the neighboring surfaces which includes this point
                neighbor_surface_ids = surface_id[repeated_node_mask]
                # choose the target face that is one of the neighboring surfaces
                target_surface_position = None
                for target_surface in self.target_surface_ids:
                    if np.isin(target_surface, neighbor_surface_ids):
                        target_surface_position = np.where(
                            neighbor_surface_ids == target_surface
                        )[0][0]
                        break
                if target_surface_position is None:
                    continue
                determined_boundary_normal = normal_boundary[repeated_node_mask][
                    target_surface_position
                ]
                determined_tangential_boundary_1 = tangential_boundary_1[
                    repeated_node_mask
                ][target_surface_position]
                determined_tangential_boundary_2 = tangential_boundary_2[
                    repeated_node_mask
                ][target_surface_position]
                ids_of_repeated_node = np.where(repeated_node_mask)[0]
                for id in ids_of_repeated_node:
                    normal_boundary[id] = determined_boundary_normal
                    tangential_boundary_1[id] = determined_tangential_boundary_1
                    tangential_boundary_2[id] = determined_tangential_boundary_2

        # get the unique coordinates and corresponding unit boundary normals of the geometry
        uniq = node_coords_xyz_boundary[sorted(idx)]
        normal_boundary = normal_boundary[sorted(idx)]
        tangential_boundary_1 = tangential_boundary_1[sorted(idx)]
        tangential_boundary_2 = tangential_boundary_2[sorted(idx)]

        return normal_boundary, tangential_boundary_1, tangential_boundary_2, uniq

    def compute_tangentials(self, normal_boundaries):
        """Compute two unit tangential vectors for each 3D boundary normal."""

        return _compute_tangentials(normal_boundaries)

    def random_points(self, n, random="pseudo"):
        """Get collocation points from geometry"""
        np.random.seed(42)

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )

        node_coords_xyz, _, _ = self.order_coordinates(node_coords, node_tag)

        if self.external_dim_size:
            node_coords_xyz = self.add_external_dim(node_coords_xyz)

        if not (n == 1):
            if n > node_coords_xyz.shape[0]:
                raise Warning(
                    f"The number o desired samples (num_domain={n}) cannot be larger than total number of total points inside of the domain ({node_coords_xyz.shape[0]})"
                )
            random_indices = np.random.choice(
                node_coords_xyz.shape[0], size=n, replace=False
            )
            node_coords_xyz = node_coords_xyz[random_indices]

        return node_coords_xyz.astype(config.real(np))

    def random_boundary_points(self, n, random="pseudo"):
        """Get boundary points from geometry"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        _, node_coords_xyz_boundary, _ = self.order_coordinates(
            node_coords, node_tag, node_tag_boundary, node_tag_inside
        )

        if self.external_dim_size:
            node_coords_xyz_boundary = self.add_external_dim(node_coords_xyz_boundary)

        return node_coords_xyz_boundary.astype(config.real(np))

    def order_coordinates(
        self, node_coords, node_tag, node_tag_boundary=None, node_tag_inside=None
    ):
        """Get sorted coordinates and node tags"""

        node_tag -= 1  # Gmsh node numbering start with 1 but we need 0

        node_coords_xyz = node_coords.reshape(-1, 3)
        node_coords_xyz = node_coords_xyz[node_tag.argsort()][:, 0:3]

        node_coords_xyz_boundary = None
        node_coords_xyz_inside = None

        if node_tag_boundary is not None:
            node_tag_boundary -= 1
            node_coords_xyz_boundary = node_coords_xyz[node_tag_boundary]
        if node_tag_inside is not None:
            node_tag_inside -= 1
            node_coords_xyz_inside = node_coords_xyz[node_tag_inside]

        return node_coords_xyz, node_coords_xyz_boundary, node_coords_xyz_inside

    def get_mesh(self):
        """Get the mesh for post-processing"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            self.dim, -1, includeBoundary=True
        )

        node_coords_xyz, node_coords_xyz_boundary, node_coords_xyz_inside = (
            self.order_coordinates(node_coords, node_tag)
        )

        element_types, element_tags, node_tags = self.gmsh_model.mesh.getElements(
            self.dim, -1
        )

        if element_types[0] == 4:  # 4-node tetrahedron
            tets = node_tags[0].reshape(-1, 4) - 1
            offset = np.arange(4, tets.shape[0] * tets.shape[1] + 1, tets.shape[1])
            cell_types = (
                np.ones(tets.shape[0]) * 10
            )  # https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html check cell types, for tets it is 10
            elements = tets

        elif element_types[0] == 5:  # 8-node hexahedron
            quads = node_tags[0].reshape(-1, 8) - 1
            offset = np.arange(8, quads.shape[0] * quads.shape[1] + 1, quads.shape[1])
            cell_types = (
                np.ones(quads.shape[0]) * 12
            )  # https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html check cell types, for hexahedron it is 12
            elements = quads

        if self.external_dim_size:
            node_coords_xyz = self.add_external_dim(node_coords_xyz)

        return node_coords_xyz, offset, cell_types, elements


class GmshGeometry2D(Geometry):
    """Represent a 2D Gmsh geometry for DeepXDE boundary and mesh queries."""

    def __init__(
        self,
        gmsh_model,
        external_dim_size=None,
        borders=None,
        revert_curve_list=None,
        revert_normal_dir_list=None,
    ):
        """Initialize a 2D Gmsh geometry wrapper."""

        self.gmsh_model = gmsh_model
        self.revert_curve_list = revert_curve_list
        self.revert_normal_dir_list = revert_normal_dir_list
        self.boundary_normal_global = self.fun_boundary_normal_global()
        self.external_dim_size = external_dim_size
        self.borders = borders
        if external_dim_size:
            self.external_dim = (
                np.linspace(self.borders[0], self.borders[1], self.external_dim_size)
                .reshape(-1, 1)
                .astype(np.dtype("f8"))
            )
        self.bbox = (
            np.array(
                [
                    self.boundary_normal_global[1][:, 0].min(),
                    self.boundary_normal_global[1][:, 1].min(),
                ]
            ),
            np.array(
                [
                    self.boundary_normal_global[1][:, 0].max(),
                    self.boundary_normal_global[1][:, 1].max(),
                ]
            ),
        )
        super(GmshGeometry2D, self).__init__(2, self.bbox, 1)

    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

        node_tag, node_coords_all, parametricCoord = self.gmsh_model.mesh.getNodes(
            2, -1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=False)[0]

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = (
            self.order_coordinates(
                node_coords_all, node_tag, node_tag_inside=node_tag_inside
            )
        )

        if self.external_dim_size:
            node_coords_xy_inside = self.add_external_dim(node_coords_xy_inside)

        return np.all(np.isin(x, node_coords_xy_inside), axis=1)

    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

        node_tag, node_coords_all, _ = self.gmsh_model.mesh.getNodes(
            2, -1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=False)[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = (
            self.order_coordinates(
                node_coords_all, node_tag, node_tag_boundary, node_tag_inside
            )
        )

        if self.external_dim_size:
            node_coords_xy_boundary = self.add_external_dim(node_coords_xy_boundary)

        return np.all(np.isin(x, node_coords_xy_boundary), axis=1)

    def boundary_normal(self, x):
        """Slice the unit normal at x for Neumann or Robin boundary conditions."""

        n, uniq = self.boundary_normal_global

        if self.external_dim_size:
            x = np.delete(x, -1, 1)

        mask = []
        for x_i in x:
            mask.extend(np.where(np.all(np.isclose(x_i, uniq), axis=1))[0].tolist())

        return n[mask]

    def fun_boundary_normal_global(self):
        """Compute unit normals on the geometry boundary."""

        fig = plt.figure(figsize=(8, 8), dpi=80)

        node_tag_boundary = []
        node_coords_x_boundary = []
        node_coords_y_boundary = []
        n_x_boundary = []
        n_y_boundary = []
        border = {}
        start = 0

        for geometry_entity_pair in self.gmsh_model.getEntities():
            if geometry_entity_pair[0] == 1:  # if it is a curve
                curve_name = "curve_" + str(geometry_entity_pair[1])
                # get node tag, coordinates and parametric coordinates form geometry
                node_tag, node_coords, parametric_coords = (
                    self.gmsh_model.mesh.getNodes(
                        1, geometry_entity_pair[1], includeBoundary=True
                    )
                )  # dim, curve tag, includeBoundary
                # calculate the first derivative
                dx_dy = self.gmsh_model.getDerivative(
                    1, geometry_entity_pair[1], parametric_coords
                )  # dim, curvetag, parametricCoord

                # reshape coordinates and first derivative
                node_coords = node_coords.reshape(-1, 3)
                dx_dy = dx_dy.reshape(-1, 3)
                # since it is a 2D problem, choose only x and y terms
                node_coords = node_coords[:, 0:2]
                dx_dy = dx_dy[:, 0:2]

                # normalize the derivative terms (unit)
                dx_dy = dx_dy / np.sqrt(dx_dy[:, 0] ** 2 + dx_dy[:, 1] ** 2)[:, None]
                # get the unit normals
                n_x = dx_dy[:, 1]
                n_y = -dx_dy[:, 0]

                # store intermediate quantities in the global variables
                node_tag_boundary.extend(node_tag.tolist())
                node_coords_x_boundary.extend(node_coords[:, 0].tolist())
                node_coords_y_boundary.extend(node_coords[:, 1].tolist())
                n_x_boundary.extend(n_x.tolist())
                n_y_boundary.extend(n_y.tolist())

                # distinguish start/end positio for each curve
                end = start + node_coords.shape[0]
                border[curve_name] = [start, end]
                start = end

                plt.scatter(node_coords[:, 0], node_coords[:, 1], label=curve_name)
                plt.quiver(node_coords[:, 0], node_coords[:, 1], n_x, n_y)
                plt.legend()

        # convert them into numpy array
        node_tag_boundary = np.array(node_tag_boundary)
        node_coords_x_boundary = np.array(node_coords_x_boundary)
        node_coords_y_boundary = np.array(node_coords_y_boundary)
        n_x_boundary = np.array(n_x_boundary)
        n_y_boundary = np.array(n_y_boundary)

        # save the initial unit normals
        normals_picture = os.path.join(os.getcwd(), "normal_directions.png")
        fig.savefig(normals_picture)
        fig = plt.figure(figsize=(8, 8), dpi=80)

        # ask user if the normal direction is correct
        print(f"Check the plot normal directions.png in {os.getcwd()}")
        if self.revert_curve_list is not None:
            curve_list = self.revert_curve_list
        else:
            curve_list = list(
                map(
                    str,
                    input(
                        "If the reverting of direction is desired, give the curve/s name with space , e.g curve_1 curve_2: "
                    ).split(),
                )
            )
            print("\n")

        # if the user gives any curve id, revert the normal direction
        for curve in curve_list:
            if curve in border.keys():
                slice_part = border[curve]
                n_x_boundary[slice_part[0] : slice_part[1]] = (
                    n_x_boundary[slice_part[0] : slice_part[1]] * -1
                )
                n_y_boundary[slice_part[0] : slice_part[1]] = (
                    n_y_boundary[slice_part[0] : slice_part[1]] * -1
                )

        # for some nodes, there are two normals available (if it is an intersection point, i.e. sharp edges)
        # thus, one normal direction has to be chosen by user
        # get the unique nodes
        u, idx, c = np.unique(node_tag_boundary, return_counts=True, return_index=True)
        # get the repeated nodes that have more than 1 boundary normal
        repeated_node_tag = u[c > 1]

        coords_xy = np.hstack(
            (
                node_coords_x_boundary.reshape(-1, 1),
                node_coords_y_boundary.reshape(-1, 1),
            )
        )
        n = np.hstack((n_x_boundary.reshape(-1, 1), n_y_boundary.reshape(-1, 1)))

        # eliminate one of the normals by asking user
        count = 0
        for node in repeated_node_tag:
            location = coords_xy[node_tag_boundary == node]
            if not np.allclose(location, location[0]):
                raise ValueError(
                    f"Repeated node {node} has inconsistent coordinates: {location}"
                )
            normals = n[node_tag_boundary == node]
            print(
                f"At location {location[0]} for node {node}, two normals exist such as {normals[0]} and {normals[1]}"
            )
            if self.revert_normal_dir_list:
                desired_normal_id = self.revert_normal_dir_list[count]
                count += 1
            else:
                desired_normal_id = int(
                    input("Please indicate the correct normal: 1 or 2 --> ")
                )
            print(
                f"At location {location[0]} for node {node} corrected normal is: {normals[desired_normal_id - 1]}"
            )
            print("\n")
            if (desired_normal_id == 1) or (desired_normal_id == 2):
                n[node_tag_boundary == node] = normals[desired_normal_id - 1]
            else:
                n[node_tag_boundary == node] = normals[0]
                warnings.warn(
                    f"Not correct id is given, the first normal is set: {normals[0]}"
                )

        # save the updated unit normals
        plt.scatter(coords_xy[:, 0], coords_xy[:, 1])
        plt.quiver(coords_xy[:, 0], coords_xy[:, 1], n[:, 0], n[:, 1])
        normals_picture = os.path.join(os.getcwd(), "normal_directions_updated.png")
        fig.savefig(normals_picture)
        plt.close("all")

        # get the unique coordinates and corresponding unit boundary normals of the geometry
        uniq = coords_xy[sorted(idx)]
        n = n[sorted(idx)]

        return n, uniq

    def random_points(self, n, random="pseudo"):
        """Get collocation points from geometry"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            2, -1, includeBoundary=False
        )

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = (
            self.order_coordinates(node_coords, node_tag)
        )

        if self.external_dim_size:
            node_coords_xy = self.add_external_dim(node_coords_xy)

        return node_coords_xy

    def get_mesh(self):
        """Get the mesh for post-processing"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            2, -1, includeBoundary=True
        )

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = (
            self.order_coordinates(node_coords, node_tag)
        )

        element_types, element_tags, node_tags = self.gmsh_model.mesh.getElements(2, -1)

        if element_types == 2:
            dol_triangles = node_tags[0].reshape(-1, 3) - 1

            offset = np.arange(
                3,
                dol_triangles.shape[0] * dol_triangles.shape[1] + 1,
                dol_triangles.shape[1],
            )
            cell_types = np.ones(dol_triangles.shape[0]) * 5

        if self.external_dim_size:
            node_coords_xy = self.add_external_dim(node_coords_xy)

        return node_coords_xy, offset, cell_types, dol_triangles

    def random_boundary_points(self, n, random="pseudo"):
        """Get boundary points from geometry"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            2, -1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=False)[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = (
            self.order_coordinates(
                node_coords, node_tag, node_tag_boundary, node_tag_inside
            )
        )

        if self.external_dim_size:
            node_coords_xy_boundary = self.add_external_dim(node_coords_xy_boundary)

        return node_coords_xy_boundary

    def order_coordinates(
        self, node_coords, node_tag, node_tag_boundary=None, node_tag_inside=None
    ):
        """Get sorted coordinates and node tags"""

        node_tag -= 1  # Gmsh node numbering start with 1 but we need 0

        node_coords_xyz = node_coords.reshape(-1, 3)
        node_coords_xy = node_coords_xyz[node_tag.argsort()][:, 0:2]

        node_coords_xy_boundary = None
        node_coords_xy_inside = None

        if node_tag_boundary is not None:
            node_tag_boundary -= 1
            node_coords_xy_boundary = node_coords_xy[node_tag_boundary]
        if node_tag_inside is not None:
            node_tag_inside -= 1
            node_coords_xy_inside = node_coords_xy[node_tag_inside]

        return node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside

    def add_external_dim(self, node_coords_xy):
        """Append repeated external dimension coordinates to 2D node coordinates."""

        node_coords_xy_rp = np.tile(node_coords_xy, (self.external_dim_size, 1))
        external_dim_rp = np.repeat(self.external_dim, node_coords_xy.shape[0], axis=0)
        node_coords_xy = np.hstack((node_coords_xy_rp, external_dim_rp))

        return node_coords_xy


class GmshGeometry1D(Geometry):
    """Represent a 1D Gmsh geometry with quadrature-based element data."""

    def __init__(
        self,
        gmsh_model,
        coord_left,
        coord_right,
        coord_quadrature=None,
        weight_quadrature=None,
        test_function=None,
        test_function_derivative=None,
        n_test_func=None,
        ele_func=None,
        external_dim_size=None,
        borders=None,
    ):
        """Initialize a 1D Gmsh geometry wrapper."""

        self.gmsh_model = gmsh_model
        self.coord_left = coord_left
        self.coord_right = coord_right
        self.coord_quadrature = coord_quadrature
        self.weight_quadrature = weight_quadrature
        self.test_function = test_function
        self.test_function_derivative = test_function_derivative
        self.ele_func = ele_func
        self.n_test_func = n_test_func
        self.n_gp = self.weight_quadrature.shape[0]
        # obtain element information
        self.get_element_info()
        self.external_dim_size = external_dim_size
        self.borders = borders
        if external_dim_size:
            self.external_dim = (
                np.linspace(self.borders[0], self.borders[1], self.external_dim_size)
                .reshape(-1, 1)
                .astype(np.dtype("f8"))
            )
        self.bbox = (np.array([self.coord_left]), np.array([self.coord_left]))
        super(GmshGeometry1D, self).__init__(1, self.bbox, 1)

    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

        node_tag, node_coords_all, parametricCoord = self.gmsh_model.mesh.getNodes(
            dim=1, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=1, tag=-1, includeBoundary=False
        )[0]

        node_coords_x, node_coords_x_boundary, node_coords_x_inside = (
            self.order_coordinates(
                node_coords_all, node_tag, node_tag_inside=node_tag_inside
            )
        )

        if self.external_dim_size:
            node_coords_x_inside = self.add_external_dim(node_coords_x_inside)

        return np.all(np.isin(x, node_coords_x_inside), axis=1)

    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

        node_tag, node_coords_all, _ = self.gmsh_model.mesh.getNodes(
            dim=1, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=1, tag=-1, includeBoundary=False
        )[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = (
            self.order_coordinates(
                node_coords_all, node_tag, node_tag_boundary, node_tag_inside
            )
        )

        if self.external_dim_size:
            node_coords_xy_boundary = self.add_external_dim(node_coords_xy_boundary)

        return np.all(np.isin(x, node_coords_xy_boundary), axis=1)

    def boundary_normal(self, x):
        """Slice the unit normal at x for Neumann or Robin boundary conditions."""

        n, uniq = self.boundary_normal_global

        if self.external_dim_size:
            x = np.delete(x, -1, 1)

        mask = []
        for x_i in x:
            mask.extend(np.where(np.all(np.isclose(x_i, uniq), axis=1))[0].tolist())

        return n[mask]

    def random_points(self, n, random="pseudo"):
        """Get collocation points from geometry"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            dim=1, tag=-1, includeBoundary=False
        )

        node_coords_x, _, _ = self.order_coordinates(node_coords, node_tag)

        if self.external_dim_size:
            node_coords_x = self.add_external_dim(node_coords_x)

        return node_coords_x.astype(config.real(np))

    def get_mesh(self):
        """Get the mesh for post-processing"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            2, -1, includeBoundary=True
        )

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = (
            self.order_coordinates(node_coords, node_tag)
        )

        element_types, element_tags, node_tags = self.gmsh_model.mesh.getElements(2, -1)

        if element_types == 2:
            dol_triangles = node_tags[0].reshape(-1, 3) - 1

            offset = np.arange(
                3,
                dol_triangles.shape[0] * dol_triangles.shape[1] + 1,
                dol_triangles.shape[1],
            )
            cell_types = np.ones(dol_triangles.shape[0]) * 5

        if self.external_dim_size:
            node_coords_xy = self.add_external_dim(node_coords_xy)

        return node_coords_xy, offset, cell_types, dol_triangles

    def random_boundary_points(self, n, random="pseudo"):
        """Get boundary points from geometry"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            dim=1, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=1, tag=-1, includeBoundary=False
        )[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        _, node_coords_x_boundary, _ = self.order_coordinates(
            node_coords, node_tag, node_tag_boundary, node_tag_inside
        )

        if self.external_dim_size:
            node_coords_x_boundary = self.add_external_dim(node_coords_x_boundary)

        return node_coords_x_boundary.astype(config.real(np))

    def order_coordinates(
        self, node_coords, node_tag, node_tag_boundary=None, node_tag_inside=None
    ):
        """Get sorted coordinates and node tags"""

        node_tag -= 1  # Gmsh node numbering start with 1 but we need 0

        node_coords_xyz = node_coords.reshape(-1, 3)
        node_coords_x = node_coords_xyz[node_tag.argsort()][:, 0:1]

        node_coords_x_boundary = None
        node_coords_x_inside = None

        if node_tag_boundary is not None:
            node_tag_boundary -= 1
            node_coords_x_boundary = node_coords_x[node_tag_boundary]
        if node_tag_inside is not None:
            node_tag_inside -= 1
            node_coords_x_inside = node_coords_x[node_tag_inside]

        return node_coords_x, node_coords_x_boundary, node_coords_x_inside

    def get_element_info(self):
        """Build mapped coordinates, Jacobians, weights, and element functions."""

        self.mapped_coordinates = np.array([])
        self.jacobian = np.array([])
        self.global_element_weights = np.array([])
        self.global_element_function = np.array([])

        self.n_elements = self.gmsh_model.mesh.getElements(1, -1)[1][0].shape[0]

        self.global_test_function = np.tile(self.test_function, self.n_elements).astype(
            config.real(np)
        )
        self.global_test_function = np.expand_dims(self.global_test_function, axis=2)
        self.global_test_function_derivative = np.tile(
            self.test_function_derivative, self.n_elements
        ).astype(config.real(np))
        self.global_test_function_derivative = np.expand_dims(
            self.global_test_function_derivative, axis=2
        )

        for element_tag in self.gmsh_model.mesh.getElements(1, -1)[1][0]:
            if self.gmsh_model.mesh.getElement(element_tag)[1].shape[0] > 2:
                raise ValueError("Use linear elements.")
            id_node1 = self.gmsh_model.mesh.getElement(element_tag)[1][0]
            id_node2 = self.gmsh_model.mesh.getElement(element_tag)[1][1]

            coordinate_node1 = self.gmsh_model.mesh.getNode(id_node1)[0][0]
            coordinate_node2 = self.gmsh_model.mesh.getNode(id_node2)[0][0]

            element_mapped_coordinate = self.get_mapped_coordinates(
                coordinate_node1, coordinate_node2
            )
            self.mapped_coordinates = np.hstack(
                (self.mapped_coordinates, element_mapped_coordinate)
            )

            element_jacobian = self.get_jacobian(coordinate_node1, coordinate_node2)
            self.jacobian = np.hstack((self.jacobian, element_jacobian))

            self.global_element_weights = np.hstack(
                (self.global_element_weights, self.weight_quadrature)
            )

            self.global_element_function = np.hstack(
                (
                    self.global_element_function,
                    self.get_element_function(
                        element_mapped_coordinate, element_jacobian
                    ),
                )
            )

        self.mapped_coordinates = self.mapped_coordinates.reshape(-1, 1).astype(
            config.real(np)
        )
        self.jacobian = self.jacobian.reshape(-1, 1).astype(config.real(np))
        self.global_element_weights = self.global_element_weights.reshape(-1, 1).astype(
            config.real(np)
        )
        self.global_element_function = self.global_element_function.reshape(
            -1, 1
        ).astype(config.real(np))

    def get_mapped_coordinates(self, coordinate_node1, coordinate_node2):
        """Map 1D reference quadrature coordinates to physical coordinates."""

        # linear mapping
        # x_m = N . x --> N linear shape functions N1=1/2(1-psi) N2=1/2(1+psi)
        N1 = 1 / 2 * (1 - self.coord_quadrature)
        N2 = 1 / 2 * (1 + self.coord_quadrature)
        mapped_coordinate = N1 * coordinate_node1 + N2 * coordinate_node2

        return mapped_coordinate

    def get_jacobian(self, coordinate_node1, coordinate_node2):
        """Compute the 1D element Jacobian."""

        DN1 = -1 / 2
        DN2 = 1 / 2
        jacob = DN1 * coordinate_node1 + DN2 * coordinate_node2

        return jacob

    def get_element_function(self, element_mapped_coordinate, element_jacobian):
        """Evaluate the weighted element function at quadrature points."""

        f_quad_element = self.ele_func(element_mapped_coordinate)

        element_func = element_jacobian * np.asarray(
            [
                sum(self.weight_quadrature * f_quad_element * t)
                for t in self.test_function
            ]
        )

        return element_func


class GmshGeometryElement(Geometry):
    """Represent a Gmsh geometry with element-level quadrature data."""

    def __init__(
        self,
        gmsh_model,
        dimension=1,
        coord_quadrature=None,
        weight_quadrature=None,
        test_function=None,
        test_function_derivative=None,
        n_test_func=None,
        external_dim_size=None,
        borders=None,
        revert_curve_list=None,
        revert_normal_dir_list=None,
        only_get_mesh=False,
    ):
        """Initialize an element-based Gmsh geometry wrapper."""

        self.gmsh_model = gmsh_model
        self.coord_quadrature = coord_quadrature
        self.weight_quadrature = weight_quadrature
        self.test_function = test_function
        self.test_function_derivative = test_function_derivative
        self.n_test_func = n_test_func
        self.dim = dimension
        # obtain element information
        if self.coord_quadrature is not None:
            self.n_gp = self.weight_quadrature.shape[0]
            self.get_element_info()

        self.revert_curve_list = revert_curve_list
        self.revert_normal_dir_list = revert_normal_dir_list
        self.only_get_mesh = only_get_mesh
        if not self.only_get_mesh:
            if self.dim > 1:
                self.boundary_normal_global = self.fun_boundary_normal_global()
        self.external_dim_size = external_dim_size
        self.borders = borders
        if external_dim_size:
            self.external_dim = (
                np.linspace(self.borders[0], self.borders[1], self.external_dim_size)
                .reshape(-1, 1)
                .astype(np.dtype("f8"))
            )
        self.bbox = [1, 1]
        super(GmshGeometryElement, self).__init__(self.dim, self.bbox, 1)

    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

        node_tag, node_coords_all, parametricCoord = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )[0]

        node_coords_x, node_coords_x_boundary, node_coords_x_inside = (
            self.order_coordinates(
                node_coords_all, node_tag, node_tag_inside=node_tag_inside
            )
        )

        if self.external_dim_size:
            node_coords_x_inside = self.add_external_dim(node_coords_x_inside)

        return _is_in_tolerance(x, node_coords_x_inside)

    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

        node_tag, node_coords_all, _ = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        node_coords_xyz, node_coords_xyz_boundary, node_coords_xyz_inside = (
            self.order_coordinates(
                node_coords_all, node_tag, node_tag_boundary, node_tag_inside
            )
        )

        if self.external_dim_size:
            node_coords_xyz_boundary = self.add_external_dim(node_coords_xyz_boundary)

        return _is_in_tolerance(x, node_coords_xyz_boundary)

    def boundary_normal(self, x):
        """Slice the unit normal at x for Neumann or Robin boundary conditions."""

        n, uniq = self.boundary_normal_global

        if self.external_dim_size:
            x = np.delete(x, -1, 1)

        mask = []
        for x_i in x:
            mask.extend(np.where(np.all(np.isclose(x_i, uniq), axis=1))[0].tolist())

        return n[mask]

    def fun_boundary_normal_global(self):
        """Compute unit normals on the geometry boundary."""

        fig = plt.figure(figsize=(8, 8), dpi=80)

        node_tag_boundary = []
        node_coords_x_boundary = []
        node_coords_y_boundary = []
        n_x_boundary = []
        n_y_boundary = []
        border = {}
        start = 0

        for geometry_entity_pair in self.gmsh_model.getEntities():
            if geometry_entity_pair[0] == 1:  # if it is a curve
                curve_name = "curve_" + str(geometry_entity_pair[1])
                # get node tag, coordinates and parametric coordinates form geometry
                node_tag, node_coords, parametric_coords = (
                    self.gmsh_model.mesh.getNodes(
                        1, geometry_entity_pair[1], includeBoundary=True
                    )
                )  # dim, curve tag, includeBoundary
                # calculate the first derivative
                dx_dy = self.gmsh_model.getDerivative(
                    1, geometry_entity_pair[1], parametric_coords
                )  # dim, curvetag, parametricCoord

                # reshape coordinates and first derivative
                node_coords = node_coords.reshape(-1, 3)
                dx_dy = dx_dy.reshape(-1, 3)
                # since it is a 2D problem, choose only x and y terms
                node_coords = node_coords[:, 0:2]
                dx_dy = dx_dy[:, 0:2]

                # normalize the derivative terms (unit)
                dx_dy = dx_dy / np.sqrt(dx_dy[:, 0] ** 2 + dx_dy[:, 1] ** 2)[:, None]
                # get the unit normals
                n_x = dx_dy[:, 1]
                n_y = -dx_dy[:, 0]

                # store intermediate quantities in the global variables
                node_tag_boundary.extend(node_tag.tolist())
                node_coords_x_boundary.extend(node_coords[:, 0].tolist())
                node_coords_y_boundary.extend(node_coords[:, 1].tolist())
                n_x_boundary.extend(n_x.tolist())
                n_y_boundary.extend(n_y.tolist())

                # distinguish start/end positio for each curve
                end = start + node_coords.shape[0]
                border[curve_name] = [start, end]
                start = end

                plt.scatter(node_coords[:, 0], node_coords[:, 1], label=curve_name)
                plt.quiver(node_coords[:, 0], node_coords[:, 1], n_x, n_y)
                plt.legend()

        # convert them into numpy array
        node_tag_boundary = np.array(node_tag_boundary)
        node_coords_x_boundary = np.array(node_coords_x_boundary)
        node_coords_y_boundary = np.array(node_coords_y_boundary)
        n_x_boundary = np.array(n_x_boundary)
        n_y_boundary = np.array(n_y_boundary)

        # save the initial unit normals
        normals_picture = os.path.join(os.getcwd(), "normal_directions.png")
        fig.savefig(normals_picture)
        fig = plt.figure(figsize=(8, 8), dpi=80)

        # ask user if the normal direction is correct
        print(f"Check the plot normal directions.png in {os.getcwd()}")
        if self.revert_curve_list is not None:
            curve_list = self.revert_curve_list
        else:
            curve_list = list(
                map(
                    str,
                    input(
                        "If the reverting of direction is desired, give the curve/s name with space , e.g curve_1 curve_2: "
                    ).split(),
                )
            )
            print("\n")

        # if the user gives any curve id, revert the normal direction
        for curve in curve_list:
            if curve in border.keys():
                slice_part = border[curve]
                n_x_boundary[slice_part[0] : slice_part[1]] = (
                    n_x_boundary[slice_part[0] : slice_part[1]] * -1
                )
                n_y_boundary[slice_part[0] : slice_part[1]] = (
                    n_y_boundary[slice_part[0] : slice_part[1]] * -1
                )

        # for some nodes, there are two normals available (if it is an intersection point, i.e. sharp edges)
        # thus, one normal direction has to be chosen by user
        # get the unique nodes
        u, idx, c = np.unique(node_tag_boundary, return_counts=True, return_index=True)
        # get the repeated nodes that have more than 1 boundary normal
        repeated_node_tag = u[c > 1]

        coords_xy = np.hstack(
            (
                node_coords_x_boundary.reshape(-1, 1),
                node_coords_y_boundary.reshape(-1, 1),
            )
        )
        n = np.hstack((n_x_boundary.reshape(-1, 1), n_y_boundary.reshape(-1, 1)))

        # eliminate one of the normals by asking user
        count = 0
        for node in repeated_node_tag:
            location = coords_xy[node_tag_boundary == node]
            if not np.allclose(location, location[0]):
                raise ValueError(
                    f"Repeated node {node} has inconsistent coordinates: {location}"
                )
            normals = n[node_tag_boundary == node]
            print(
                f"At location {location[0]} for node {node}, two normals exist such as {normals[0]} and {normals[1]}"
            )
            if self.revert_normal_dir_list:
                desired_normal_id = self.revert_normal_dir_list[count]
                count += 1
            else:
                desired_normal_id = int(
                    input("Please indicate the correct normal: 1 or 2 --> ")
                )
            print(
                f"At location {location[0]} for node {node} corrected normal is: {normals[desired_normal_id - 1]}"
            )
            print("\n")
            if (desired_normal_id == 1) or (desired_normal_id == 2):
                n[node_tag_boundary == node] = normals[desired_normal_id - 1]
            else:
                n[node_tag_boundary == node] = normals[0]
                warnings.warn(
                    f"Not correct id is given, the first normal is set: {normals[0]}"
                )

        # save the updated unit normals
        plt.scatter(coords_xy[:, 0], coords_xy[:, 1])
        plt.quiver(coords_xy[:, 0], coords_xy[:, 1], n[:, 0], n[:, 1])
        normals_picture = os.path.join(os.getcwd(), "normal_directions_updated.png")
        fig.savefig(normals_picture)
        plt.close("all")

        # get the unique coordinates and corresponding unit boundary normals of the geometry
        uniq = coords_xy[sorted(idx)]
        n = n[sorted(idx)]

        return n, uniq

    def random_points(self, n, random="pseudo"):
        """Get collocation points from geometry"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )

        node_coords_x, _, _ = self.order_coordinates(node_coords, node_tag)

        if self.external_dim_size:
            node_coords_x = self.add_external_dim(node_coords_x)
        # If not entire mesh is not desired to be used for calculations
        if not (n == 1):
            if n > node_coords_x.shape[0]:
                raise Warning(
                    f"The number o desired samples (num_domain={n}) cannot be larger than total number of total points inside of the domain ({node_coords_x.shape[0]})"
                )
            np.random.seed(42)
            random_indices = np.random.choice(
                node_coords_x.shape[0], size=n, replace=False
            )
            node_coords_x = node_coords_x[random_indices]

        return node_coords_x.astype(config.real(np))

    def get_mesh(self):
        """Get the mesh for post-processing"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            self.dim, -1, includeBoundary=True
        )

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = (
            self.order_coordinates(node_coords, node_tag)
        )

        element_types, element_tags, node_tags = self.gmsh_model.mesh.getElements(
            self.dim, -1
        )

        if element_types[0] == 2:  # triangle with 3 nodes
            dol_triangles = node_tags[0].reshape(-1, 3) - 1

            offset = np.arange(
                3,
                dol_triangles.shape[0] * dol_triangles.shape[1] + 1,
                dol_triangles.shape[1],
            )
            cell_types = np.ones(dol_triangles.shape[0]) * 5
            elements = dol_triangles

        elif element_types[0] == 3:  # quad with 4 nodes
            quads = node_tags[0].reshape(-1, 4) - 1
            offset = np.arange(4, quads.shape[0] * quads.shape[1] + 1, quads.shape[1])
            cell_types = np.ones(quads.shape[0]) * 9
            elements = quads

        if self.external_dim_size:
            node_coords_xy = self.add_external_dim(node_coords_xy)

        return node_coords_xy, offset, cell_types, elements

    def random_boundary_points(self, n, random="pseudo"):
        """Get boundary points from geometry"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        _, node_coords_x_boundary, _ = self.order_coordinates(
            node_coords, node_tag, node_tag_boundary, node_tag_inside
        )

        if self.external_dim_size:
            node_coords_x_boundary = self.add_external_dim(node_coords_x_boundary)

        return node_coords_x_boundary.astype(config.real(np))

    def order_coordinates(
        self, node_coords, node_tag, node_tag_boundary=None, node_tag_inside=None
    ):
        """Get sorted coordinates and node tags"""

        node_tag -= 1  # gmsh node numbering start with 1 but we need 0

        node_coords_xyz = node_coords.reshape(-1, 3)
        node_coords_x = node_coords_xyz[node_tag.argsort()][:, 0 : self.dim]

        node_coords_x_boundary = None
        node_coords_x_inside = None

        if node_tag_boundary is not None:
            node_tag_boundary -= 1
            node_coords_x_boundary = node_coords_x[node_tag_boundary]
        if node_tag_inside is not None:
            node_tag_inside -= 1
            node_coords_x_inside = node_coords_x[node_tag_inside]

        return node_coords_x, node_coords_x_boundary, node_coords_x_inside

    def get_element_info(self):
        """Build element-level quadrature data for mapped coordinates and weights."""

        self.n_elements = self.gmsh_model.mesh.getElements(self.dim, -1)[1][0].shape[0]

        self.mapped_coordinates = np.empty((self.n_elements * self.n_gp, self.dim))
        self.global_test_function = np.empty(
            (self.n_test_func, self.n_elements, self.n_gp, self.dim)
        )
        self.global_test_function_derivative = np.empty(
            (self.n_test_func, self.n_elements, self.n_gp, self.dim)
        )
        self.jacobian = np.empty((self.n_elements, self.n_gp, 1))
        self.global_element_weights = np.empty((self.n_elements, self.n_gp, self.dim))

        element_id = 0

        for element_tag in self.gmsh_model.mesh.getElements(self.dim, -1)[1][0]:
            if self.gmsh_model.mesh.getElement(element_tag)[1].shape[0] > self.dim * 2:
                raise ValueError("Use linear elements.")

            coordinate_list = []
            for dof in range(self.dim * 2):
                node_id = self.gmsh_model.mesh.getElement(element_tag)[1][dof]
                coordinate_list.append(
                    self.gmsh_model.mesh.getNode(node_id)[0][0 : self.dim]
                )

            element_mapped_coordinate = self.get_mapped_coordinates(coordinate_list)
            self.mapped_coordinates[
                element_id * self.n_gp : (element_id + 1) * self.n_gp, :
            ] = element_mapped_coordinate

            element_jacobian = self.get_jacobian(coordinate_list)
            self.jacobian[element_id] = element_jacobian

            self.global_test_function[:, element_id] = self.test_function.copy()
            self.global_test_function_derivative[:, element_id] = (
                self.test_function_derivative.copy()
            )
            self.global_element_weights[element_id] = self.weight_quadrature.copy()

            element_id += 1

        self.jacobian = self.jacobian.reshape(
            self.jacobian.shape[0] * self.jacobian.shape[1], self.jacobian.shape[2]
        )
        self.global_element_weights = self.global_element_weights.reshape(
            self.global_element_weights.shape[0] * self.global_element_weights.shape[1],
            self.global_element_weights.shape[2],
        )
        self.global_test_function = self.global_test_function.reshape(
            self.global_test_function.shape[0],
            self.global_test_function.shape[1] * self.global_test_function.shape[2],
            self.global_test_function.shape[3],
        )
        self.global_test_function_derivative = (
            self.global_test_function_derivative.reshape(
                self.global_test_function_derivative.shape[0],
                self.global_test_function_derivative.shape[1]
                * self.global_test_function_derivative.shape[2],
                self.global_test_function_derivative.shape[3],
            )
        )

        self.mapped_coordinates = self.mapped_coordinates.astype(config.real(np))
        self.jacobian = self.jacobian.astype(config.real(np))
        self.global_element_weights = self.global_element_weights.astype(
            config.real(np)
        )
        self.global_test_function = self.global_test_function.astype(config.real(np))
        self.global_test_function_derivative = (
            self.global_test_function_derivative.astype(config.real(np))
        )

    def get_mapped_coordinates(self, coordinate_list):
        """Map reference quadrature coordinates to physical element coordinates."""

        # linear mapping
        # x_m = N . x --> N linear shape functions N1=1/2(1-psi) N2=1/2(1+psi)
        if self.dim == 1:
            N1 = 1 / 2 * (1 - self.coord_quadrature)
            N2 = 1 / 2 * (1 + self.coord_quadrature)
            mapped_coordinate = N1 * coordinate_list[0] + N2 * coordinate_list[1]

            return mapped_coordinate

        elif self.dim == 2:
            # linear mapping
            # x_m = N . x --> N linear shape functions N1=1/2(1-psi) N2=1/2(1+psi)
            psi_x = self.coord_quadrature[:, 0:1]
            psi_y = self.coord_quadrature[:, 1:2]

            N1 = 1 / 4 * (1 - psi_x) * (1 - psi_y)
            N2 = 1 / 4 * (1 + psi_x) * (1 - psi_y)
            N3 = 1 / 4 * (1 + psi_x) * (1 + psi_y)
            N4 = 1 / 4 * (1 - psi_x) * (1 + psi_y)

            N_stack = np.hstack((N1, N2, N3, N4))
            x_stack = np.vstack(
                (
                    coordinate_list[0][0:1],
                    coordinate_list[1][0:1],
                    coordinate_list[2][0:1],
                    coordinate_list[3][0:1],
                )
            )
            y_stack = np.vstack(
                (
                    coordinate_list[0][1:2],
                    coordinate_list[1][1:2],
                    coordinate_list[2][1:2],
                    coordinate_list[3][1:2],
                )
            )

            mapped_coordinate_x = np.matmul(N_stack, x_stack)
            mapped_coordinate_y = np.matmul(N_stack, y_stack)

            return np.hstack((mapped_coordinate_x, mapped_coordinate_y))

    def get_jacobian(self, coordinate_list):
        """Compute element Jacobians at quadrature points."""

        if self.dim == 1:
            DN1 = -1 / 2 * np.ones((self.n_gp, 1))
            DN2 = 1 / 2 * np.ones((self.n_gp, 1))
            jacob = DN1 * coordinate_list[0] + DN2 * coordinate_list[1]

        if self.dim == 2:
            psi_x = self.coord_quadrature[:, 0:1]
            psi_y = self.coord_quadrature[:, 1:2]

            DN1_psi_x = -1 / 4 * (1 - psi_y)
            DN2_psi_x = 1 / 4 * (1 - psi_y)
            DN3_psi_x = 1 / 4 * (1 + psi_y)
            DN4_psi_x = -1 / 4 * (1 + psi_y)

            DN1_psi_y = -1 / 4 * (1 - psi_x)
            DN2_psi_y = -1 / 4 * (1 + psi_x)
            DN3_psi_y = 1 / 4 * (1 + psi_x)
            DN4_psi_y = 1 / 4 * (1 - psi_x)

            DN_stack_upper = np.hstack((DN1_psi_x, DN2_psi_x, DN3_psi_x, DN4_psi_x))
            DN_stack_lower = np.hstack((DN1_psi_y, DN2_psi_y, DN3_psi_y, DN4_psi_y))

            x_stack = np.vstack(
                (
                    coordinate_list[0][0:1],
                    coordinate_list[1][0:1],
                    coordinate_list[2][0:1],
                    coordinate_list[3][0:1],
                )
            )
            y_stack = np.vstack(
                (
                    coordinate_list[0][1:2],
                    coordinate_list[1][1:2],
                    coordinate_list[2][1:2],
                    coordinate_list[3][1:2],
                )
            )

            J11 = np.matmul(DN_stack_upper, x_stack)
            J12 = np.matmul(DN_stack_upper, y_stack)
            J21 = np.matmul(DN_stack_lower, x_stack)
            J22 = np.matmul(DN_stack_lower, y_stack)

            jacob = J11 * J22 - J12 * J21

        return jacob


class GmshGeometryElementDeepEnergy(Geometry):
    """Represent a Gmsh geometry with Deep Energy element and boundary data."""

    def __init__(
        self,
        gmsh_model,
        dimension=1,
        coord_quadrature=None,
        weight_quadrature=None,
        external_dim_size=None,
        borders=None,
        target_surface_ids=None,
        revert_curve_list=None,
        revert_normal_dir_list=None,
        only_get_mesh=False,
        boundary_dim=1,
        coord_quadrature_boundary=None,
        weight_quadrature_boundary=None,
        boundary_selection_map=None,
        lagrange_method=False,
        use_geometry_normals=False,
    ):
        """Initialize a Deep Energy Gmsh geometry wrapper."""

        self.gmsh_model = gmsh_model
        self.dim = dimension
        self.coord_quadrature = coord_quadrature
        self.weight_quadrature = weight_quadrature
        self.external_dim_size = external_dim_size
        self.revert_curve_list = revert_curve_list
        self.revert_normal_dir_list = revert_normal_dir_list
        self.target_surface_ids = target_surface_ids
        self.only_get_mesh = only_get_mesh
        self.boundary_dim = boundary_dim
        self.coord_quadrature_boundary = coord_quadrature_boundary
        self.weight_quadrature_boundary = weight_quadrature_boundary
        self.lagrange_method = lagrange_method

        self.mapped_coordinates_boundary = None
        self.mapped_normal_boundary = None
        self.jacobian_boundary = None
        self.global_weights_boundary = None
        self.n_gp_boundary = None
        self.boundary_elements = None
        self.boundary_selection_map = boundary_selection_map
        self.boundary_selection_tag = None
        self.lagrange_parameter = None
        self.use_geometry_normals = use_geometry_normals
        self.boundary_normal_global = None

        # obtain element information
        if self.coord_quadrature is not None:
            self.n_gp = self.weight_quadrature.shape[0]
            print(
                "--------------------------------------------------------------------------------------------------------------"
            )
            print("Building element-based information...")
            t_start = time.time()
            self.get_element_info()
            t_end = time.time()
            elapsed_time = t_end - t_start
            print(
                f"Element-based information is built and it took:{elapsed_time:.4f} seconds"
            )
            print(
                "--------------------------------------------------------------------------------------------------------------"
            )

        self.borders = borders
        if external_dim_size:
            self.external_dim = (
                np.linspace(self.borders[0], self.borders[1], self.external_dim_size)
                .reshape(-1, 1)
                .astype(np.dtype("f8"))
            )
        self.bbox = [1, 1]

        super(GmshGeometryElementDeepEnergy, self).__init__(self.dim, self.bbox, 1)

    def ensure_boundary_normal_global(self):
        """Compute pointwise CAD normals only when boundary_normal() needs them."""

        if self.boundary_normal_global is not None:
            return
        if self.only_get_mesh:
            raise ValueError(
                "Boundary normals are not available when only_get_mesh=True."
            )
        if self.dim == 2:
            self.boundary_normal_global = self.fun_boundary_normal_global()
        elif self.dim == 3:
            self.boundary_normal_global = self.fun_boundary_normal_global_3d()
        else:
            raise NotImplementedError(
                f"Boundary normals are not implemented for dimension {self.dim}."
            )

    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

        node_tag, node_coords_all, parametricCoord = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )[0]

        node_coords_x, node_coords_x_boundary, node_coords_x_inside = (
            self.order_coordinates(
                node_coords_all, node_tag, node_tag_inside=node_tag_inside
            )
        )

        if self.external_dim_size:
            node_coords_x_inside = self.add_external_dim(node_coords_x_inside)

        return _is_in_tolerance(x, node_coords_x_inside)

    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

        node_tag, node_coords_all, _ = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        node_coords_xyz, node_coords_xyz_boundary, node_coords_xyz_inside = (
            self.order_coordinates(
                node_coords_all, node_tag, node_tag_boundary, node_tag_inside
            )
        )

        if self.external_dim_size:
            node_coords_xyz_boundary = self.add_external_dim(node_coords_xyz_boundary)

        return _is_in_tolerance(x, node_coords_xyz_boundary)

    def boundary_normal(self, x):
        """Slice the unit normal at x for Neumann or Robin boundary conditions."""

        self.ensure_boundary_normal_global()

        if self.dim == 2:
            n, uniq = self.boundary_normal_global
        elif self.dim == 3:
            n, _, _, uniq = self.boundary_normal_global

        if self.external_dim_size:
            x = np.delete(x, -1, 1)

        mask = []
        for x_i in x:
            mask.extend(np.where(np.all(np.isclose(x_i, uniq), axis=1))[0].tolist())

        return n[mask]

    def fun_boundary_normal_global(self):
        """Compute unit normals on the geometry boundary."""

        fig = plt.figure(figsize=(8, 8), dpi=80)

        node_tag_boundary = []
        node_coords_x_boundary = []
        node_coords_y_boundary = []
        n_x_boundary = []
        n_y_boundary = []
        border = {}
        start = 0

        for geometry_entity_pair in self.gmsh_model.getEntities():
            if geometry_entity_pair[0] == 1:  # if it is a curve
                curve_name = "curve_" + str(geometry_entity_pair[1])
                # get node tag, coordinates and parametric coordinates form geometry
                node_tag, node_coords, parametric_coords = (
                    self.gmsh_model.mesh.getNodes(
                        1, geometry_entity_pair[1], includeBoundary=True
                    )
                )  # dim, curve tag, includeBoundary
                # calculate the first derivative
                dx_dy = self.gmsh_model.getDerivative(
                    1, geometry_entity_pair[1], parametric_coords
                )  # dim, curvetag, parametricCoord

                # reshape coordinates and first derivative
                node_coords = node_coords.reshape(-1, 3)
                dx_dy = dx_dy.reshape(-1, 3)
                # since it is a 2D problem, choose only x and y terms
                node_coords = node_coords[:, 0:2]
                dx_dy = dx_dy[:, 0:2]

                # normalize the derivative terms (unit)
                dx_dy = dx_dy / np.sqrt(dx_dy[:, 0] ** 2 + dx_dy[:, 1] ** 2)[:, None]
                # get the unit normals
                n_x = dx_dy[:, 1]
                n_y = -dx_dy[:, 0]

                # store intermediate quantities in the global variables
                node_tag_boundary.extend(node_tag.tolist())
                node_coords_x_boundary.extend(node_coords[:, 0].tolist())
                node_coords_y_boundary.extend(node_coords[:, 1].tolist())
                n_x_boundary.extend(n_x.tolist())
                n_y_boundary.extend(n_y.tolist())

                # distinguish start/end positio for each curve
                end = start + node_coords.shape[0]
                border[curve_name] = [start, end]
                start = end

                plt.scatter(node_coords[:, 0], node_coords[:, 1], label=curve_name)
                plt.quiver(node_coords[:, 0], node_coords[:, 1], n_x, n_y)
                plt.legend()

        # convert them into numpy array
        node_tag_boundary = np.array(node_tag_boundary)
        node_coords_x_boundary = np.array(node_coords_x_boundary)
        node_coords_y_boundary = np.array(node_coords_y_boundary)
        n_x_boundary = np.array(n_x_boundary)
        n_y_boundary = np.array(n_y_boundary)

        # save the initial unit normals
        normals_picture = os.path.join(os.getcwd(), "normal_directions.png")
        fig.savefig(normals_picture)
        fig = plt.figure(figsize=(8, 8), dpi=80)

        # ask user if the normal direction is correct
        print(f"Check the plot normal directions.png in {os.getcwd()}")
        if self.revert_curve_list is not None:
            curve_list = self.revert_curve_list
        else:
            curve_list = list(
                map(
                    str,
                    input(
                        "If the reverting of direction is desired, give the curve/s name with space , e.g curve_1 curve_2: "
                    ).split(),
                )
            )
            print("\n")

        # if the user gives any curve id, revert the normal direction
        for curve in curve_list:
            if curve in border.keys():
                slice_part = border[curve]
                n_x_boundary[slice_part[0] : slice_part[1]] = (
                    n_x_boundary[slice_part[0] : slice_part[1]] * -1
                )
                n_y_boundary[slice_part[0] : slice_part[1]] = (
                    n_y_boundary[slice_part[0] : slice_part[1]] * -1
                )

        # for some nodes, there are two normals available (if it is an intersection point, i.e. sharp edges)
        # thus, one normal direction has to be chosen by user
        # get the unique nodes
        u, idx, c = np.unique(node_tag_boundary, return_counts=True, return_index=True)
        # get the repeated nodes that have more than 1 boundary normal
        repeated_node_tag = u[c > 1]

        coords_xy = np.hstack(
            (
                node_coords_x_boundary.reshape(-1, 1),
                node_coords_y_boundary.reshape(-1, 1),
            )
        )
        n = np.hstack((n_x_boundary.reshape(-1, 1), n_y_boundary.reshape(-1, 1)))

        # eliminate one of the normals by asking user
        count = 0
        for node in repeated_node_tag:
            location = coords_xy[node_tag_boundary == node]
            if not np.allclose(location, location[0]):
                raise ValueError(
                    f"Repeated node {node} has inconsistent coordinates: {location}"
                )
            normals = n[node_tag_boundary == node]
            print(
                f"At location {location[0]} for node {node}, two normals exist such as {normals[0]} and {normals[1]}"
            )
            if self.revert_normal_dir_list:
                desired_normal_id = self.revert_normal_dir_list[count]
                count += 1
            else:
                desired_normal_id = int(
                    input("Please indicate the correct normal: 1 or 2 --> ")
                )
            print(
                f"At location {location[0]} for node {node} corrected normal is: {normals[desired_normal_id - 1]}"
            )
            print("\n")
            if (desired_normal_id == 1) or (desired_normal_id == 2):
                n[node_tag_boundary == node] = normals[desired_normal_id - 1]
            else:
                n[node_tag_boundary == node] = normals[0]
                warnings.warn(
                    f"Not correct id is given, the first normal is set: {normals[0]}"
                )

        # save the updated unit normals
        plt.scatter(coords_xy[:, 0], coords_xy[:, 1])
        plt.quiver(coords_xy[:, 0], coords_xy[:, 1], n[:, 0], n[:, 1])
        normals_picture = os.path.join(os.getcwd(), "normal_directions_updated.png")
        fig.savefig(normals_picture)
        plt.close("all")

        # get the unique coordinates and corresponding unit boundary normals of the geometry
        uniq = coords_xy[sorted(idx)]
        n = n[sorted(idx)]

        return n, uniq

    def fun_boundary_normal_global_3d(self):
        """Compute geometry boundary normals and tangential vectors."""

        node_tag_boundary = []
        node_coords_xyz_boundary = []
        normal_boundary = []
        surface_id = []
        border = {}
        start = 0

        for geometry_entity_pair in self.gmsh_model.getEntities():
            if geometry_entity_pair[0] == 2:  # if it is a surface
                s_tag = geometry_entity_pair[1]
                surface_name = "surface_" + str(s_tag)

                # get node tag, coordinates and parametric coordinates form geometry
                node_tag, node_coords, parametric_coords = (
                    self.gmsh_model.mesh.getNodes(
                        dim=2, tag=s_tag, includeBoundary=True
                    )
                )  # dim, curve tag, includeBoundary
                # get normals
                normals = self.gmsh_model.getNormal(s_tag, parametric_coords)

                # reshape coordinates and first derivative
                node_coords = node_coords.reshape(-1, 3)
                normals = normals.reshape(-1, 3)

                surface_id_intermediate = [s_tag] * node_coords.shape[0]
                surface_id.extend(surface_id_intermediate)

                # store intermediate quantities in the global variables
                node_tag_boundary.extend(node_tag.tolist())
                node_coords_xyz_boundary.extend(node_coords.tolist())
                normal_boundary.extend(normals.tolist())

        # convert them into numpy array
        node_tag_boundary = np.array(node_tag_boundary)
        node_coords_xyz_boundary = np.array(node_coords_xyz_boundary)
        normal_boundary = np.array(normal_boundary)
        surface_id = np.array(surface_id)

        # calculate the tangential vector components.
        tangential_boundary_1, tangential_boundary_2 = self.compute_tangentials(
            normal_boundary
        )

        # get the unique nodes
        u, idx, c = np.unique(node_tag_boundary, return_counts=True, return_index=True)
        # get the repeated nodes that have more than 1 boundary normal
        repeated_node_tag = u[c > 1]

        if self.target_surface_ids:
            for repeated_node_id in repeated_node_tag:
                # create mask to get the repeated nodes in the global boundary node list
                repeated_node_mask = node_tag_boundary == repeated_node_id
                # get the neighboring surfaces which includes this point
                neighbor_surface_ids = surface_id[repeated_node_mask]
                # choose the target face that is one of the neighboring surfaces
                target_surface_position = None
                for target_surface in self.target_surface_ids:
                    if np.isin(target_surface, neighbor_surface_ids):
                        target_surface_position = np.where(
                            neighbor_surface_ids == target_surface
                        )[0][0]
                        break
                if target_surface_position is None:
                    continue
                determined_boundary_normal = normal_boundary[repeated_node_mask][
                    target_surface_position
                ]
                determined_tangential_boundary_1 = tangential_boundary_1[
                    repeated_node_mask
                ][target_surface_position]
                determined_tangential_boundary_2 = tangential_boundary_2[
                    repeated_node_mask
                ][target_surface_position]
                ids_of_repeated_node = np.where(repeated_node_mask)[0]
                for id in ids_of_repeated_node:
                    normal_boundary[id] = determined_boundary_normal
                    tangential_boundary_1[id] = determined_tangential_boundary_1
                    tangential_boundary_2[id] = determined_tangential_boundary_2

        # get the unique coordinates and corresponding unit boundary normals of the geometry
        uniq = node_coords_xyz_boundary[sorted(idx)]
        normal_boundary = normal_boundary[sorted(idx)]
        tangential_boundary_1 = tangential_boundary_1[sorted(idx)]
        tangential_boundary_2 = tangential_boundary_2[sorted(idx)]

        return normal_boundary, tangential_boundary_1, tangential_boundary_2, uniq

    def compute_tangentials(self, normal_boundaries):
        """Compute two unit tangential vectors for each 3D boundary normal."""

        return _compute_tangentials(normal_boundaries)

    def random_points(self, n, random="pseudo"):
        """Get collocation points from geometry"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )

        node_coords_x, _, _ = self.order_coordinates(node_coords, node_tag)

        if self.external_dim_size:
            node_coords_x = self.add_external_dim(node_coords_x)

        return node_coords_x.astype(config.real(np))

    def get_mesh(self):
        """Get the mesh for post-processing"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            self.dim, -1, includeBoundary=True
        )

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = (
            self.order_coordinates(node_coords, node_tag)
        )

        element_types, element_tags, node_tags = self.gmsh_model.mesh.getElements(
            self.dim, -1
        )

        if element_types[0] == 2:  # triangle with 3 nodes
            dol_triangles = node_tags[0].reshape(-1, 3) - 1

            offset = np.arange(
                3,
                dol_triangles.shape[0] * dol_triangles.shape[1] + 1,
                dol_triangles.shape[1],
            )
            cell_types = np.ones(dol_triangles.shape[0]) * 5
            elements = dol_triangles

        elif element_types[0] == 3:  # quad with 4 nodes
            quads = node_tags[0].reshape(-1, 4) - 1
            offset = np.arange(4, quads.shape[0] * quads.shape[1] + 1, quads.shape[1])
            cell_types = np.ones(quads.shape[0]) * 9
            elements = quads

        elif element_types[0] == 4:  # 4-node tetrahedron
            tets = node_tags[0].reshape(-1, 4) - 1
            offset = np.arange(4, tets.shape[0] * tets.shape[1] + 1, tets.shape[1])
            cell_types = (
                np.ones(tets.shape[0]) * 10
            )  # https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html check cell types, for tets it is 10
            elements = tets

        elif element_types[0] == 5:  # 8-node hexahedron
            quads = node_tags[0].reshape(-1, 8) - 1
            offset = np.arange(8, quads.shape[0] * quads.shape[1] + 1, quads.shape[1])
            cell_types = (
                np.ones(quads.shape[0]) * 12
            )  # https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html check cell types, for hexahedron it is 12
            elements = quads

        if self.external_dim_size:
            node_coords_xy = self.add_external_dim(node_coords_xy)

        return node_coords_xy, offset, cell_types, elements

    def random_boundary_points(self, n, random="pseudo"):
        """Get boundary points from geometry"""

        node_tag, node_coords, _ = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=True
        )
        node_tag_inside = self.gmsh_model.mesh.getNodes(
            dim=self.dim, tag=-1, includeBoundary=False
        )[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        _, node_coords_x_boundary, _ = self.order_coordinates(
            node_coords, node_tag, node_tag_boundary, node_tag_inside
        )

        if self.external_dim_size:
            node_coords_x_boundary = self.add_external_dim(node_coords_x_boundary)

        return node_coords_x_boundary.astype(config.real(np))

    def order_coordinates(
        self, node_coords, node_tag, node_tag_boundary=None, node_tag_inside=None
    ):
        """Get sorted coordinates and node tags"""

        node_tag -= 1  # gmsh node numbering start with 1 but we need 0

        node_coords_xyz = node_coords.reshape(-1, 3)
        node_coords_x = node_coords_xyz[node_tag.argsort()][:, 0 : self.dim]

        node_coords_x_boundary = None
        node_coords_x_inside = None

        if node_tag_boundary is not None:
            node_tag_boundary -= 1
            node_coords_x_boundary = node_coords_x[node_tag_boundary]
        if node_tag_inside is not None:
            node_tag_inside -= 1
            node_coords_x_inside = node_coords_x[node_tag_inside]

        return node_coords_x, node_coords_x_boundary, node_coords_x_inside

    def get_gmsh_normals_at_boundary_gauss_points(self, boundary_mapped_coordinate):
        """Get Gmsh normals at mapped boundary quadrature points."""

        boundary_mapped_coordinate = np.asarray(boundary_mapped_coordinate)
        n_points = boundary_mapped_coordinate.shape[0]
        coord_xyz = np.zeros((boundary_mapped_coordinate.shape[0], 3))
        coord_xyz[:, : self.dim] = boundary_mapped_coordinate[:, : self.dim]
        coord = coord_xyz.reshape(-1).tolist()
        boundary_entity_dim = self.dim - 1
        best_distance = np.full(
            n_points, np.inf
        )  # distance from the closest point on the boundary entity to the mapped boundary quadrature point
        best_normals = np.zeros(
            (n_points, self.dim)
        )  # normal vector corresponding to the closest point on the boundary entity for each mapped boundary quadrature point
        found_normals = np.zeros(
            n_points, dtype=bool
        )  # boolean array to track whether a normal vector has been found for each mapped boundary quadrature point

        for entity_dim, entity_tag in self.gmsh_model.getEntities():
            if entity_dim != boundary_entity_dim:
                continue

            try:
                closest_point, parametric_coord = self.gmsh_model.getClosestPoint(
                    entity_dim, entity_tag, coord
                )
            except AttributeError:
                closest_point = boundary_mapped_coordinate
                parametric_coord = self.gmsh_model.getParametrization(
                    entity_dim, entity_tag, coord
                )
            except Exception:
                closest_point = None
                parametric_coord = None

            if parametric_coord is None:
                continue

            try:
                if entity_dim == 1:
                    tangent = np.array(
                        self.gmsh_model.getDerivative(1, entity_tag, parametric_coord)
                    ).reshape(-1, 3)[:, :2]
                    normals = np.column_stack((tangent[:, 1], -tangent[:, 0]))
                else:
                    normals = np.array(
                        self.gmsh_model.getNormal(entity_tag, parametric_coord)
                    ).reshape(-1, 3)[:, : self.dim]
            except Exception:
                normals = None

            if normals is None:
                continue

            closest_point = np.asarray(closest_point)
            if closest_point.shape == boundary_mapped_coordinate.shape:
                closest_point = closest_point.reshape(-1, self.dim)
            else:
                closest_point = closest_point.reshape(-1, 3)[:, : self.dim]
            distance = np.linalg.norm(
                closest_point - boundary_mapped_coordinate, axis=1
            )
            # make sure to update the normal only if the closest point on the boundary entity is closer than any previously found closest point (mostly found on other entities) for each mapped boundary quadrature point
            update_mask = distance < best_distance

            best_distance[update_mask] = distance[update_mask]
            best_normals[update_mask] = normals[update_mask]
            found_normals[update_mask] = True

        if not np.any(found_normals):
            return None

        normal_norm = np.linalg.norm(best_normals, axis=1, keepdims=True)
        best_normals = np.divide(
            best_normals,
            normal_norm,
            out=np.zeros_like(best_normals),
            where=normal_norm > 0,
        )

        return best_normals, found_normals

    def orient_boundary_normals_outward(
        self, normals, boundary_mapped_coordinate, element_centers
    ):
        """Flip boundary normals so they point away from element centers."""

        direction = boundary_mapped_coordinate - element_centers
        # compute the dot product between normals and direction vectors, and flip normals where the dot product is negative
        flip = np.sum(normals * direction, axis=1, keepdims=True) < 0

        return np.where(flip, -normals, normals)

    def get_flat_boundary_face_normals(self, edge_coordinate_list, coordinate_list):
        """Compute repeated outward normals for a flat boundary edge or face."""

        # Collect face and element centers
        face_coordinates = np.vstack(edge_coordinate_list)
        element_center = np.mean(np.vstack(coordinate_list), axis=0)
        face_center = np.mean(face_coordinates, axis=0)

        # Build a normal from the boundary edge or face orientation
        if self.dim == 2:
            tangent = face_coordinates[1] - face_coordinates[0]
            normal = np.array([tangent[1], -tangent[0]])
        else:
            normal = np.cross(
                face_coordinates[1] - face_coordinates[0],
                face_coordinates[2] - face_coordinates[0],
            )

        # Skip degenerate edges or faces
        normal_norm = np.linalg.norm(normal)
        if np.isclose(normal_norm, 0):
            return np.zeros((self.n_gp_boundary, self.dim))

        # Normalize and orient away from the element center
        normal = normal / normal_norm
        if np.dot(normal, face_center - element_center) < 0:
            normal = -normal

        # Use the same flat normal at every boundary Gauss point
        return np.repeat(normal.reshape(1, -1), self.n_gp_boundary, axis=0)

    def get_element_info(self):
        """Build element and optional boundary quadrature data."""

        self.n_elements = self.gmsh_model.mesh.getElements(self.dim, -1)[1][0].shape[0]

        self.mapped_coordinates = np.empty((self.n_elements * self.n_gp, self.dim))
        self.jacobian = np.empty((self.n_elements * self.n_gp, 1))
        self.global_element_weights = np.empty(
            (self.n_elements * self.n_gp, self.weight_quadrature.shape[1])
        )

        # If the boundary integral information is desired
        if self.coord_quadrature_boundary is not None:
            n_boundary_points = self.random_boundary_points(1).shape[0]
            self.n_gp_boundary = self.weight_quadrature_boundary.shape[0]
            if self.dim == 2:
                boundary_multiplier = 2  #
            elif self.dim == 3:
                boundary_multiplier = 12
            # allocate the mapped_coordinates_boundary using the maximum possible number of boundary elements (edges in 2D and surface is 3D) using boundary_multiplier*n_boundary_points and self.n_gp_boundary
            # Later we will reduce the sizes
            self.mapped_coordinates_boundary = np.empty(
                (n_boundary_points * boundary_multiplier * self.n_gp_boundary, self.dim)
            )
            self.mapped_normal_boundary = np.empty(
                (n_boundary_points * boundary_multiplier * self.n_gp_boundary, self.dim)
            )
            self.jacobian_boundary = np.empty(
                (n_boundary_points * boundary_multiplier * self.n_gp_boundary, 1)
            )
            self.global_weights_boundary = np.empty(
                (
                    n_boundary_points * boundary_multiplier * self.n_gp_boundary,
                    self.weight_quadrature_boundary.shape[1],
                )
            )
            boundary_element_centers = np.empty(
                (n_boundary_points * boundary_multiplier * self.n_gp_boundary, self.dim)
            )

            if self.boundary_selection_map:
                tag_list = [
                    selection["tag"] for selection in self.boundary_selection_map
                ]
                self.boundary_selection_tag = {
                    tag: np.empty(
                        (n_boundary_points * boundary_multiplier * self.n_gp_boundary,),
                        dtype=bool,
                    )
                    for tag in tag_list
                }
        else:
            boundary_element_centers = None

        element_id = 0
        boundary_element_id = 0

        for element_tag in self.gmsh_model.mesh.getElements(self.dim, -1)[1][0]:
            if self.gmsh_model.mesh.getElement(element_tag)[1].shape[0] > 2**self.dim:
                raise ValueError("Use linear elements.")

            coordinate_list = []
            for node_id in self.gmsh_model.mesh.getElement(element_tag)[1]:
                coordinate_list.append(
                    self.gmsh_model.mesh.getNode(node_id)[0][0 : self.dim]
                )

            element_mapped_coordinate = self.get_mapped_coordinates(
                self.dim, self.coord_quadrature, coordinate_list
            )
            self.mapped_coordinates[
                element_id * self.n_gp : (element_id + 1) * self.n_gp, :
            ] = element_mapped_coordinate

            element_jacobian = self.get_jacobian(
                self.dim, self.coord_quadrature, coordinate_list
            )
            self.jacobian[element_id * self.n_gp : (element_id + 1) * self.n_gp, :] = (
                element_jacobian
            )

            self.global_element_weights[
                element_id * self.n_gp : (element_id + 1) * self.n_gp, :
            ] = self.weight_quadrature.copy()

            element_id += 1

            if self.coord_quadrature_boundary is not None:
                # https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
                element_type = self.gmsh_model.mesh.getElement(element_tag)[0]
                if element_type == 2:  # triangle with 3 nodes
                    edge_list = [
                        [coordinate_list[0], coordinate_list[1]],
                        [coordinate_list[1], coordinate_list[2]],
                        [coordinate_list[2], coordinate_list[0]],
                    ]
                elif element_type == 3:  # quad with 4 nodes
                    edge_list = [
                        [coordinate_list[0], coordinate_list[1]],
                        [coordinate_list[1], coordinate_list[2]],
                        [coordinate_list[2], coordinate_list[3]],
                        [coordinate_list[3], coordinate_list[0]],
                    ]
                elif (
                    element_type == 4
                ):  # tets with 4 nodes. The following edge_list represents the surface_list, same variable name is used for easiness.
                    edge_list = [
                        [
                            coordinate_list[0],
                            coordinate_list[2],
                            coordinate_list[1],
                        ],  # Face 1
                        [
                            coordinate_list[0],
                            coordinate_list[1],
                            coordinate_list[3],
                        ],  # Face 2
                        [
                            coordinate_list[1],
                            coordinate_list[2],
                            coordinate_list[3],
                        ],  # Face 3
                        [
                            coordinate_list[2],
                            coordinate_list[0],
                            coordinate_list[3],
                        ],  # Face 4
                    ]
                elif (
                    element_type == 5
                ):  # hexa with 8 nodes. The following edge_list represents the surface_list, same variable name is used for easiness.
                    edge_list = [
                        [
                            coordinate_list[0],
                            coordinate_list[3],
                            coordinate_list[2],
                            coordinate_list[1],
                        ],  # Face 1 (back)
                        [
                            coordinate_list[4],
                            coordinate_list[5],
                            coordinate_list[6],
                            coordinate_list[7],
                        ],  # Face 2 (front)
                        [
                            coordinate_list[0],
                            coordinate_list[1],
                            coordinate_list[5],
                            coordinate_list[4],
                        ],  # Face 3 (bottom)
                        [
                            coordinate_list[1],
                            coordinate_list[2],
                            coordinate_list[6],
                            coordinate_list[5],
                        ],  # Face 4 (right)
                        [
                            coordinate_list[2],
                            coordinate_list[3],
                            coordinate_list[7],
                            coordinate_list[6],
                        ],  # Face 5 (top)
                        [
                            coordinate_list[3],
                            coordinate_list[0],
                            coordinate_list[4],
                            coordinate_list[7],
                        ],  # Face 6 (left)
                    ]
                else:
                    raise NotImplementedError(
                        f"Element type: {element_type}, not implemented! Use only linear triangle, quad, tetrahedral or hexahedral elements"
                    )

                for edge_coordinate_list in edge_list:
                    on_boundary = all(
                        self.on_boundary(coord.reshape(-1, self.dim))
                        for coord in edge_coordinate_list
                    )
                    if on_boundary:
                        boundary_mapped_coordinate = self.get_mapped_coordinates(
                            self.boundary_dim,
                            self.coord_quadrature_boundary,
                            edge_coordinate_list,
                        )
                        boundary_slice = slice(
                            boundary_element_id * self.n_gp_boundary,
                            (boundary_element_id + 1) * self.n_gp_boundary,
                        )
                        self.mapped_coordinates_boundary[boundary_slice, :] = (
                            boundary_mapped_coordinate
                        )

                        if self.dim in [2, 3]:
                            boundary_normal_mapped = (
                                self.get_flat_boundary_face_normals(
                                    edge_coordinate_list, coordinate_list
                                )
                            )
                            boundary_element_centers[boundary_slice, :] = np.mean(
                                np.vstack(coordinate_list), axis=0
                            )
                        else:
                            boundary_normal_coordinate_list = [
                                self.boundary_normal(
                                    edge_coords.reshape(-1, self.dim)
                                ).ravel()
                                for edge_coords in edge_coordinate_list
                            ]
                            boundary_normal_mapped = self.get_mapped_coordinates(
                                self.boundary_dim,
                                self.coord_quadrature_boundary,
                                boundary_normal_coordinate_list,
                            )
                        self.mapped_normal_boundary[boundary_slice, :] = (
                            boundary_normal_mapped
                        )

                        boundary_jacobian = self.get_jacobian(
                            self.boundary_dim,
                            self.coord_quadrature_boundary,
                            edge_coordinate_list,
                        )
                        self.jacobian_boundary[boundary_slice, :] = boundary_jacobian

                        self.global_weights_boundary[boundary_slice, :] = (
                            self.weight_quadrature_boundary.copy()
                        )

                        if self.boundary_selection_map:
                            for selection_map in self.boundary_selection_map:
                                boundary_function = selection_map["boundary_function"]
                                tag = selection_map["tag"]
                                check_cond = all(
                                    boundary_function(coord)
                                    for coord in edge_coordinate_list
                                )
                                check_cond = np.repeat(
                                    check_cond, self.n_gp_boundary, axis=0
                                )
                                self.boundary_selection_tag[tag][boundary_slice] = (
                                    check_cond
                                )

                        boundary_element_id += 1

        if (
            self.use_geometry_normals
            and self.coord_quadrature_boundary is not None
            and self.dim in [2, 3]
            and boundary_element_id > 0
        ):
            used_boundary_slice = slice(0, boundary_element_id * self.n_gp_boundary)
            gmsh_normal_result = self.get_gmsh_normals_at_boundary_gauss_points(
                self.mapped_coordinates_boundary[used_boundary_slice, :]
            )

            if gmsh_normal_result is not None:
                gmsh_normals, gmsh_normal_mask = gmsh_normal_result
                if np.any(gmsh_normal_mask):
                    gmsh_normals[gmsh_normal_mask] = (
                        self.orient_boundary_normals_outward(
                            gmsh_normals[gmsh_normal_mask],
                            self.mapped_coordinates_boundary[used_boundary_slice, :][
                                gmsh_normal_mask
                            ],
                            boundary_element_centers[used_boundary_slice, :][
                                gmsh_normal_mask
                            ],
                        )
                    )
                    boundary_normals = self.mapped_normal_boundary[
                        used_boundary_slice, :
                    ]
                    boundary_normals[gmsh_normal_mask] = gmsh_normals[gmsh_normal_mask]
                    self.mapped_normal_boundary[used_boundary_slice, :] = (
                        boundary_normals
                    )

        self.mapped_coordinates = self.mapped_coordinates.astype(config.real(np))
        self.jacobian = self.jacobian.astype(config.real(np))
        self.global_element_weights = self.global_element_weights.astype(
            config.real(np)
        )

        if self.coord_quadrature_boundary is not None:
            self.boundary_elements = boundary_element_id
            self.mapped_coordinates_boundary = self.mapped_coordinates_boundary[
                : boundary_element_id * self.n_gp_boundary, :
            ].astype(config.real(np))
            self.mapped_normal_boundary = self.mapped_normal_boundary[
                : boundary_element_id * self.n_gp_boundary, :
            ].astype(config.real(np))
            self.jacobian_boundary = self.jacobian_boundary[
                : boundary_element_id * self.n_gp_boundary, :
            ].astype(config.real(np))
            self.global_weights_boundary = self.global_weights_boundary[
                : boundary_element_id * self.n_gp_boundary, :
            ].astype(config.real(np))
            if self.boundary_selection_map:
                for tag in self.boundary_selection_tag.keys():
                    self.boundary_selection_tag[tag] = self.boundary_selection_tag[tag][
                        : boundary_element_id * self.n_gp_boundary
                    ]
            if self.lagrange_method:
                self.lagrange_parameter = np.zeros_like(self.jacobian_boundary)

    def get_mapped_coordinates(self, dimension, quadrature_coordinate, coordinate_list):
        """Map reference quadrature coordinates to physical coordinates."""

        # linear mapping
        # x_m = N . x --> N linear shape functions N1=1/2(1-psi) N2=1/2(1+psi)
        if dimension == 1:
            N1 = 1 / 2 * (1 - quadrature_coordinate)
            N2 = 1 / 2 * (1 + quadrature_coordinate)
            mapped_coordinate = N1 * coordinate_list[0] + N2 * coordinate_list[1]

            return mapped_coordinate
        # 2D biliniear mapping (quad4 and tri3)
        elif dimension == 2:
            xi = quadrature_coordinate[:, 0:1]
            eta = quadrature_coordinate[:, 1:2]
            coords = np.vstack([coord for coord in coordinate_list])

            if len(coordinate_list) == 4:  # quad element logic
                N1 = 1 / 4 * (1 - xi) * (1 - eta)
                N2 = 1 / 4 * (1 + xi) * (1 - eta)
                N3 = 1 / 4 * (1 + xi) * (1 + eta)
                N4 = 1 / 4 * (1 - xi) * (1 + eta)

                N_stack = np.hstack((N1, N2, N3, N4))
            elif len(coordinate_list) == 3:  # triangular element logic
                N1 = 1.0 - xi - eta
                N2 = xi
                N3 = eta

                N_stack = np.hstack((N1, N2, N3))  # shape (n_gp, 3)

            mapped_coordinate = np.matmul(N_stack, coords)  # shape (n_gp, dim)

            return mapped_coordinate

        # 3D trilinear mapping (hex8)
        elif dimension == 3:
            xi = quadrature_coordinate[:, 0:1]
            eta = quadrature_coordinate[:, 1:2]
            zeta = quadrature_coordinate[:, 2:3]
            coords = np.vstack([coord for coord in coordinate_list])

            if len(coordinate_list) == 8:  # hex element logic
                N1 = 0.125 * (1 - xi) * (1 - eta) * (1 - zeta)
                N2 = 0.125 * (1 + xi) * (1 - eta) * (1 - zeta)
                N3 = 0.125 * (1 + xi) * (1 + eta) * (1 - zeta)
                N4 = 0.125 * (1 - xi) * (1 + eta) * (1 - zeta)
                N5 = 0.125 * (1 - xi) * (1 - eta) * (1 + zeta)
                N6 = 0.125 * (1 + xi) * (1 - eta) * (1 + zeta)
                N7 = 0.125 * (1 + xi) * (1 + eta) * (1 + zeta)
                N8 = 0.125 * (1 - xi) * (1 + eta) * (1 + zeta)

                N_stack = np.hstack((N1, N2, N3, N4, N5, N6, N7, N8))
            elif len(coordinate_list) == 4:  # tet element logic
                N1 = 1.0 - xi - eta - zeta
                N2 = xi
                N3 = eta
                N4 = zeta
                N_stack = np.hstack((N1, N2, N3, N4))

            mapped_coordinate = np.matmul(N_stack, coords)  # shape (n_gp, dim)

            return mapped_coordinate

    def get_jacobian(self, dimension, quadrature_coordinate, coordinate_list):
        """Compute Jacobians for mapped 1D, 2D, or 3D elements."""

        if dimension == 1:
            # Extract x and y coordinates
            x1, y1 = coordinate_list[0]  # First node (x1, y1)
            x2, y2 = coordinate_list[1]  # Second node (x2, y2)
            # Compute the Jacobian as the Euclidean distance
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            jacob = length / 2 * np.ones((quadrature_coordinate.shape[0], 1))

        if dimension == 2:
            xi = quadrature_coordinate[:, 0:1]
            eta = quadrature_coordinate[:, 1:2]
            num_gp = quadrature_coordinate.shape[0]
            jacob = np.zeros((quadrature_coordinate.shape[0], 1))
            coords = np.vstack([coord for coord in coordinate_list])

            if len(coordinate_list) == 4:
                # Shape function derivatives w.r.t natural coords
                DN1_dxi = -0.25 * (1 - eta)
                DN2_dxi = 0.25 * (1 - eta)
                DN3_dxi = 0.25 * (1 + eta)
                DN4_dxi = -0.25 * (1 + eta)

                DN1_deta = -0.25 * (1 - xi)
                DN2_deta = -0.25 * (1 + xi)
                DN3_deta = 0.25 * (1 + xi)
                DN4_deta = 0.25 * (1 - xi)

                dN_dxi = np.hstack(
                    (DN1_dxi, DN2_dxi, DN3_dxi, DN4_dxi)
                )  # shape (num_gp, 4)
                dN_deta = np.hstack(
                    (DN1_deta, DN2_deta, DN3_deta, DN4_deta)
                )  # shape (num_gp, 4)
            elif len(coordinate_list) == 3:
                dN_dxi = np.hstack(
                    [
                        -np.ones_like(xi),  # ∂N1/∂ξ
                        np.ones_like(xi),  # ∂N2/∂ξ
                        np.zeros_like(xi),  # ∂N3/∂ξ
                    ]
                )

                dN_deta = np.hstack(
                    [
                        -np.ones_like(eta),  # ∂N1/∂η
                        np.zeros_like(eta),  # ∂N2/∂η
                        np.ones_like(eta),  # ∂N3/∂η
                    ]
                )
            else:
                raise ValueError(
                    "2D elements must have 3 (triangle) or 4 (quad) nodes."
                )

            for i in range(num_gp):
                dx_dxi = (
                    dN_dxi[i, :] @ coords
                )  # shape (3,) — tangent vector in ξ direction
                dx_deta = (
                    dN_deta[i, :] @ coords
                )  # shape (3,) — tangent vector in η direction

                normal_vec = np.cross(dx_dxi, dx_deta)
                jacob[i] = np.linalg.norm(normal_vec)  # area scale factor

        elif dimension == 3:
            xi = quadrature_coordinate[:, 0:1]
            eta = quadrature_coordinate[:, 1:2]
            zeta = quadrature_coordinate[:, 2:3]
            num_gp = quadrature_coordinate.shape[0]
            jacob = np.zeros((quadrature_coordinate.shape[0], 1))
            coords = np.vstack([coord for coord in coordinate_list])

            if len(coordinate_list) == 8:  # hex element logic
                # Derivatives of shape functions w.r.t xi, eta, zeta (for each node)
                dN_dxi = [
                    -0.125 * (1 - eta) * (1 - zeta),
                    0.125 * (1 - eta) * (1 - zeta),
                    0.125 * (1 + eta) * (1 - zeta),
                    -0.125 * (1 + eta) * (1 - zeta),
                    -0.125 * (1 - eta) * (1 + zeta),
                    0.125 * (1 - eta) * (1 + zeta),
                    0.125 * (1 + eta) * (1 + zeta),
                    -0.125 * (1 + eta) * (1 + zeta),
                ]

                dN_deta = [
                    -0.125 * (1 - xi) * (1 - zeta),
                    -0.125 * (1 + xi) * (1 - zeta),
                    0.125 * (1 + xi) * (1 - zeta),
                    0.125 * (1 - xi) * (1 - zeta),
                    -0.125 * (1 - xi) * (1 + zeta),
                    -0.125 * (1 + xi) * (1 + zeta),
                    0.125 * (1 + xi) * (1 + zeta),
                    0.125 * (1 - xi) * (1 + zeta),
                ]

                dN_dzeta = [
                    -0.125 * (1 - xi) * (1 - eta),
                    -0.125 * (1 + xi) * (1 - eta),
                    -0.125 * (1 + xi) * (1 + eta),
                    -0.125 * (1 - xi) * (1 + eta),
                    0.125 * (1 - xi) * (1 - eta),
                    0.125 * (1 + xi) * (1 - eta),
                    0.125 * (1 + xi) * (1 + eta),
                    0.125 * (1 - xi) * (1 + eta),
                ]

                dN_dxi_stack = np.hstack(dN_dxi)
                dN_deta_stack = np.hstack(dN_deta)
                dN_dzeta_stack = np.hstack(dN_dzeta)

                x_stack = np.vstack([coord[0:1] for coord in coordinate_list])
                y_stack = np.vstack([coord[1:2] for coord in coordinate_list])
                z_stack = np.vstack([coord[2:3] for coord in coordinate_list])

                # Compute J at all quadrature points
                J11 = np.matmul(dN_dxi_stack, x_stack)
                J12 = np.matmul(dN_dxi_stack, y_stack)
                J13 = np.matmul(dN_dxi_stack, z_stack)

                J21 = np.matmul(dN_deta_stack, x_stack)
                J22 = np.matmul(dN_deta_stack, y_stack)
                J23 = np.matmul(dN_deta_stack, z_stack)

                J31 = np.matmul(dN_dzeta_stack, x_stack)
                J32 = np.matmul(dN_dzeta_stack, y_stack)
                J33 = np.matmul(dN_dzeta_stack, z_stack)

                # Assemble Jacobians and compute determinants for each point
                for i in range(num_gp):
                    J = np.array(
                        [
                            [J11[i, 0], J12[i, 0], J13[i, 0]],
                            [J21[i, 0], J22[i, 0], J23[i, 0]],
                            [J31[i, 0], J32[i, 0], J33[i, 0]],
                        ]
                    )
                    jacob[i] = np.linalg.det(J)

            elif len(coordinate_list) == 4:  # Tet element logic
                # Derivatives of shape functions (∂N/∂ξ) for linear tetrahedron (constant)
                dN_dxi_eta_zeta = np.array(
                    [
                        [-1, -1, -1],  # ∂N1
                        [1, 0, 0],  # ∂N2
                        [0, 1, 0],  # ∂N3
                        [0, 0, 1],  # ∂N4
                    ]
                )  # shape (4, 3)

                # Compute Jacobian matrix (3x3): J = X^T @ dN_dξ
                J = coords.T @ dN_dxi_eta_zeta  # (3x4) @ (4x3) = (3x3)

                # Constant determinant over all GPs
                detJ = np.linalg.det(J)
                jacob[:] = abs(detJ)

        return jacob
