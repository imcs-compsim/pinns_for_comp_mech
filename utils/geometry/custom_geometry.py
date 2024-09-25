import warnings
from deepxde.geometry.geometry import Geometry
from deepxde import config
import numpy as np
import matplotlib.pyplot as plt
import os

class GmshGeometry3D(Geometry):
    def __init__(self, gmsh_model, external_dim_size=None):
        self.gmsh_model = gmsh_model
        self.external_dim_size = external_dim_size
        self.boundary_normal_global = self.fun_boundary_normal_global()
        self.bbox = (np.array([1,1,1]))
        super(GmshGeometry3D, self).__init__(
            3, self.bbox, 1
        )
    
    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

        node_tag, node_coords_all, parametricCoord  = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=False)[0]

        _, _, node_coords_xyz_inside = self.order_coordinates(node_coords_all, node_tag, node_tag_inside=node_tag_inside)

        if self.external_dim_size:
            node_coords_xyz_inside = self.add_external_dim(node_coords_xyz_inside)

        return self.is_in_tolerance(x, node_coords_xyz_inside)

    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

        node_tag, node_coords_all, _  = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=False)[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        _, node_coords_xyz_boundary, _ = self.order_coordinates(node_coords_all, node_tag, node_tag_boundary, node_tag_inside)

        if self.external_dim_size:
            node_coords_xyz_boundary = self.add_external_dim(node_coords_xyz_boundary)
        
        return self.is_in_tolerance(x, node_coords_xyz_boundary)
    
    def is_in_tolerance(self, provided, target):
            
        tolerance = 1e-5
        
        contained_rows = np.all(np.isclose(provided[:, None, :], target, rtol=tolerance, atol=tolerance), axis=2)
        contained_indices_boolean = np.any(contained_rows, axis=1)
        return contained_indices_boolean
        
    def boundary_normal(self, x):
        """Slice the unit normal at x for Neumann or Robin boundary conditions."""

        n, _, _, uniq = self.boundary_normal_global

        if self.external_dim_size:
            x = np.delete(x, -1, 1)

        mask = []
        for x_i in x:
            mask.extend(np.where(np.all(np.isclose(x_i,uniq),axis=1))[0].tolist()) 
        
        return n[mask]
    
    def boundary_tangential_1(self, x):
        """Slice the unit tangentil vector 1 at x for Neumann or Robin boundary conditions."""

        _, t_1, _, uniq = self.boundary_normal_global

        if self.external_dim_size:
            x = np.delete(x, -1, 1)

        mask = []
        for x_i in x:
            mask.extend(np.where(np.all(np.isclose(x_i,uniq),axis=1))[0].tolist()) 
        
        return t_1[mask]
    
    def boundary_tangential_2(self, x):
        """Slice the unit tangentil vector 2 at x for Neumann or Robin boundary conditions."""

        _, _, t_2, uniq = self.boundary_normal_global

        if self.external_dim_size:
            x = np.delete(x, -1, 1)

        mask = []
        for x_i in x:
            mask.extend(np.where(np.all(np.isclose(x_i,uniq),axis=1))[0].tolist()) 
        
        return t_2[mask]
        
    def fun_boundary_normal_global(self):
        """Compute the unit normal on the geometry boundary"""

        # fig = plt.figure(figsize=(8, 8), dpi=80)

        node_tag_boundary, node_coords_xyz_boundary, normal_boundary = [],[],[]
        border = {}
        start = 0

        for geometry_entitiy_pair in self.gmsh_model.getEntities():
            if geometry_entitiy_pair[0] == 2: # if it is a surface
                s_tag = geometry_entitiy_pair[1]
                surface_name = "surface_" + str(s_tag)
                
                # get node tag, coordinates and parametric coordinates form geometry
                node_tag, node_coords, parameteric_coords = self.gmsh_model.mesh.getNodes(dim=2, tag = s_tag, includeBoundary=True) # dim, curve tag, includeBoundary
                # get normals
                normals = self.gmsh_model.getNormal(s_tag, parameteric_coords)
                
                
                # reshape coordinates and first derivative
                node_coords = node_coords.reshape(-1,3)
                normals = normals.reshape(-1,3)

                # store intermediate quantities in the global variables
                node_tag_boundary.extend(node_tag.tolist())
                node_coords_xyz_boundary.extend(node_coords.tolist())
                normal_boundary.extend(normals.tolist())

                # # distinguish start/end positio for each surface 
                # end = start+node_coords.shape[0]
                # border[surface_name] = [start,end]
                # start = end

        # convert them into numpy array
        node_tag_boundary = np.array(node_tag_boundary)
        node_coords_xyz_boundary = np.array(node_coords_xyz_boundary)
        normal_boundary = np.array(normal_boundary)
        
        # calculate the tangential vector components.
        tangential_boundary_1, tangential_boundary_2 = self.compute_tangentials(normal_boundary)

        # get the unique nodes
        u, idx, c = np.unique(node_tag_boundary, return_counts=True, return_index=True)
        # get the repeated nodes that have more than 1 boundary normal
        # repeated_node_tag = u[c>1]
        
        # get the unique coordinates and corresponding unit boundary normals of the geometry
        uniq = node_coords_xyz_boundary[sorted(idx)]
        normal_boundary = normal_boundary[sorted(idx)] 
        tangential_boundary_1 = tangential_boundary_1[sorted(idx)] 
        tangential_boundary_2 = tangential_boundary_2[sorted(idx)] 

        return normal_boundary, tangential_boundary_1, tangential_boundary_2, uniq
    
    def compute_tangentials(self, normal_boundaries):
        # calculate the tangential vector components here.
        # Tangential of normal vector in 3D is a plane. Thus, we need two tangetial vectors, let's call them t_1 and t_2.
        t_1_list = []
        t_2_list = []

        # Loop over each normal vector in the input array
        for n in normal_boundaries:
            n_x, n_y, n_z = n

            # case 1 Handle [0, 0, 1]
            if np.isclose(n_z, 1):
                t_1 = np.array([0, 1, 0])
                t_2 = np.array([-1, 0, 0])
            # case 2 Handle [0, 0, -1]
            elif np.isclose(n_z, -1):
                t_1 = np.array([0, 1, 0])
                t_2 = np.array([1, 0, 0])
            else:
                t_1 = np.array([n_y, -n_x, 0]) / np.sqrt(n_x**2 + n_y**2)
                t_2 = np.cross(n, t_1)

            # check if all normal_boundary (let's call it as n), t1 and t2 are perpedicular to each, meaning 
            # n . t_1 = 0, n . t_2 = 0, and t_1 . t-2 = 0
            assert(np.isclose(np.dot(n,t_1), 0))
            assert(np.isclose(np.dot(n,t_2), 0))
            assert(np.isclose(np.dot(t_1,t_2), 0))

            # Append results to the list
            t_1_list.append(t_1)
            t_2_list.append(t_2)

        return np.array(t_1_list), np.array(t_2_list)    
    
    def random_points(self, n, random="pseudo"):
        """Get collocation points from geometry""" 
        np.random.seed(42)
        
        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=False)

        node_coords_xyz, _, _ = self.order_coordinates(node_coords, node_tag)

        if self.external_dim_size:
            node_coords_xyz = self.add_external_dim(node_coords_xyz)
            
        if not (n==1):
            if n>node_coords_xyz.shape[0]:
                raise Warning(f"The number o desired samples (num_domain={n}) cannot be larger than total number of total points inside of the domain ({node_coords_xyz.shape[0]})")
            random_indices = np.random.choice(node_coords_xyz.shape[0], size=n, replace=False)
            node_coords_xyz = node_coords_xyz[random_indices]
            
        return node_coords_xyz.astype(config.real(np))
    
    def random_boundary_points(self, n, random="pseudo"):
        """Get boundary points from geometry"""

        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=False)[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        _, node_coords_xyz_boundary, _ = self.order_coordinates(node_coords, node_tag, node_tag_boundary, node_tag_inside)

        if self.external_dim_size:
            node_coords_xyz_boundary = self.add_external_dim(node_coords_xyz_boundary)

        return node_coords_xyz_boundary.astype(config.real(np))
    
    def order_coordinates(self, node_coords, node_tag, node_tag_boundary=None, node_tag_inside=None):
        '''Get sorted coordinates and node tags'''

        node_tag -= 1 # gmsh node numbering start with 1 but we need 0         

        node_coords_xyz = node_coords.reshape(-1,3)
        node_coords_xyz = node_coords_xyz[node_tag.argsort()][:,0:3]
        
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

        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(self.dim, -1, includeBoundary=True)

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = self.order_coordinates(node_coords, node_tag)

        element_types, element_tags, node_tags = self.gmsh_model.mesh.getElements(self.dim,-1)

        if element_types[0] == 4: # 4-node tetrahedron
            raise NotImplemented("Elemental info for 4-node tetrahedron not implemented!")
        
        elif element_types[0] == 5: # 8-node hexahedron
            quads = node_tags[0].reshape(-1,8) - 1
            offset = np.arange(8,quads.shape[0]*quads.shape[1]+1, quads.shape[1])
            cell_types = np.ones(quads.shape[0])*12 #https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html check cell types, for hexahedron it is 12
            elements = quads
        
        if self.external_dim_size:
            node_coords_xy = self.add_external_dim(node_coords_xy)

        return node_coords_xy, offset, cell_types, elements
        
class GmshGeometry2D(Geometry):
    def __init__(self, gmsh_model, external_dim_size=None, borders=None, revert_curve_list=None, revert_normal_dir_list=None):
        self.gmsh_model = gmsh_model
        self.revert_curve_list = revert_curve_list
        self.revert_normal_dir_list = revert_normal_dir_list
        self.boundary_normal_global = self.fun_boundary_normal_global()
        self.external_dim_size = external_dim_size
        self.borders=borders
        if external_dim_size:
            self.external_dim = np.linspace(self.borders[0],self.borders[1],self.external_dim_size).reshape(-1,1).astype(np.dtype('f8'))
        self.bbox = (np.array([self.boundary_normal_global[1][:,0].min(),self.boundary_normal_global[1][:,1].min()]), np.array([self.boundary_normal_global[1][:,0].max(),self.boundary_normal_global[1][:,1].max()]))
        super(GmshGeometry2D, self).__init__(
            2, self.bbox, 1
        )

    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

        node_tag, node_coords_all, parametricCoord  = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=False)[0]

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = self.order_coordinates(node_coords_all, node_tag, node_tag_inside=node_tag_inside)

        if self.external_dim_size:
            node_coords_xy_inside = self.add_external_dim(node_coords_xy_inside)

        return np.all(np.isin(x, node_coords_xy_inside), axis=1)

    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

        node_tag, node_coords_all, _  = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=False)[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = self.order_coordinates(node_coords_all, node_tag, node_tag_boundary, node_tag_inside)

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
            mask.extend(np.where(np.all(np.isclose(x_i,uniq),axis=1))[0].tolist()) 
        
        return n[mask]

    def fun_boundary_normal_global(self):
        """Compute the unit normal on the geometry boundary"""

        fig = plt.figure(figsize=(8, 8), dpi=80)

        node_tag_boundary, node_coords_x_boundary, node_coords_y_boundary, n_x_boundary, n_y_boundary = [],[],[],[],[]
        border = {}
        start = 0

        for geometry_entitiy_pair in self.gmsh_model.getEntities():
            if geometry_entitiy_pair[0] == 1: # if it is a curve
                curve_name = "curve_" + str(geometry_entitiy_pair[1])
                # get node tag, coordinates and parametric coordinates form geometry
                node_tag, node_coords, parameteric_coords = self.gmsh_model.mesh.getNodes(1,geometry_entitiy_pair[1], includeBoundary=True) # dim, curve tag, includeBoundary
                # calculate the first derivative
                dx_dy = self.gmsh_model.getDerivative(1, geometry_entitiy_pair[1], parameteric_coords) # dim, curvetag, parametricCoord
                
                # reshape coordinates and first derivative
                node_coords = node_coords.reshape(-1,3)
                dx_dy = dx_dy.reshape(-1,3)
                # since it is a 2D problem, choose only x and y terms
                node_coords = node_coords[:,0:2]
                dx_dy = dx_dy[:,0:2]

                # normalize the derivative terms (unit)
                dx_dy = dx_dy/np.sqrt(dx_dy[:,0]**2 + dx_dy[:,1]**2)[:,None]
                # get the unit normals
                n_x = dx_dy[:,1]
                n_y = -dx_dy[:,0]

                # store intermediate quantities in the global variables
                node_tag_boundary.extend(node_tag.tolist())
                node_coords_x_boundary.extend(node_coords[:,0].tolist())
                node_coords_y_boundary.extend(node_coords[:,1].tolist())
                n_x_boundary.extend(n_x.tolist())
                n_y_boundary.extend(n_y.tolist())

                # distinguish start/end positio for each curve 
                end = start+node_coords.shape[0]
                border[curve_name] = [start,end]
                start = end
                
                plt.scatter(node_coords[:,0], node_coords[:,1], label=curve_name)
                plt.quiver(node_coords[:,0], node_coords[:,1], n_x, n_y)
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
            curve_list = list(map(str, input("If the reverting of direction is desired, give the curve/s name with space , e.g curve_1 curve_2: ").split()))
            print("\n")

        # if the user gives any curve id, revert the normal direction
        for curve in curve_list:
            if curve in border.keys():
                slice_part = border[curve]
                n_x_boundary[slice_part[0]:slice_part[1]] = n_x_boundary[slice_part[0]:slice_part[1]]*-1
                n_y_boundary[slice_part[0]:slice_part[1]] = n_y_boundary[slice_part[0]:slice_part[1]]*-1 

        # for some nodes, there are two normals available (if it is an intersection point, i.e. sharp edges)
        # thus, one normal direction has to be chosen by user
        # get the unique nodes
        u, idx, c = np.unique(node_tag_boundary, return_counts=True, return_index=True)
        # get the repeated nodes that have more than 1 boundary normal
        repeated_node_tag = u[c>1]

        coords_xy = np.hstack((node_coords_x_boundary.reshape(-1,1),node_coords_y_boundary.reshape(-1,1)))
        n = np.hstack((n_x_boundary.reshape(-1,1),n_y_boundary.reshape(-1,1)))

        # eleminate one of the normals by asking user
        count = 0
        for node in repeated_node_tag:
            location = coords_xy[node_tag_boundary == node]
            assert(np.all(location[0] == location[1]))
            normals = n[node_tag_boundary == node]
            print(f"At location {location[0]} for node {node}, two normals exist such as {normals[0]} and {normals[1]}")
            if self.revert_normal_dir_list:
                desired_normal_id = self.revert_normal_dir_list[count]
                count += 1
            else:
                desired_normal_id = int(input("Please indicate the correct normal: 1 or 2 --> "))
            print(f"At location {location[0]} for node {node} corrected normal is: {normals[desired_normal_id-1]}")
            print("\n")
            if (desired_normal_id == 1) or (desired_normal_id == 2):
                n[node_tag_boundary == node] = normals[desired_normal_id - 1]
            else:
                n[node_tag_boundary == node] = normals[0]
                warnings.warn(f"Not correct id is given, the first normal is set: {normals[0]}")
        
        # save the updated unit normals 
        plt.scatter(coords_xy[:,0], coords_xy[:,1])
        plt.quiver(coords_xy[:,0], coords_xy[:,1], n[:,0], n[:,1])
        normals_picture = os.path.join(os.getcwd(), "normal_directions_updated.png")
        fig.savefig(normals_picture)
        plt.close('all')
        
        # get the unique coordinates and corresponding unit boundary normals of the geometry
        uniq = coords_xy[sorted(idx)]
        n = n[sorted(idx)] 

        return n, uniq

    def random_points(self, n, random="pseudo"):
        """Get collocation points from geometry"""

        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=False)

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = self.order_coordinates(node_coords, node_tag)

        if self.external_dim_size:
            node_coords_xy = self.add_external_dim(node_coords_xy)

        return node_coords_xy
    
    def get_mesh(self):
        """Get the mesh for post-processing"""

        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=True)

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = self.order_coordinates(node_coords, node_tag)

        element_types, element_tags, node_tags = self.gmsh_model.mesh.getElements(2,-1)

        if element_types == 2:
            dol_triangles = node_tags[0].reshape(-1,3) - 1

            offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1, dol_triangles.shape[1])
            cell_types = np.ones(dol_triangles.shape[0])*5
        
        if self.external_dim_size:
            node_coords_xy = self.add_external_dim(node_coords_xy)

        return node_coords_xy, offset, cell_types, dol_triangles

    def random_boundary_points(self, n, random="pseudo"):
        """Get boundary points from geometry"""

        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=False)[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = self.order_coordinates(node_coords, node_tag, node_tag_boundary, node_tag_inside)

        if self.external_dim_size:
            node_coords_xy_boundary = self.add_external_dim(node_coords_xy_boundary)

        return node_coords_xy_boundary
    
    def order_coordinates(self, node_coords, node_tag, node_tag_boundary=None, node_tag_inside=None):
        '''Get sorted coordinates and node tags'''

        node_tag -= 1 # gmsh node numbering start with 1 but we need 0         

        node_coords_xyz = node_coords.reshape(-1,3)
        node_coords_xy = node_coords_xyz[node_tag.argsort()][:,0:2]
        
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

        node_coords_xy_rp = np.tile(node_coords_xy,(self.external_dim_size,1))
        external_dim_rp = np.repeat(self.external_dim,node_coords_xy.shape[0],axis=0)
        node_coords_xy = np.hstack((node_coords_xy_rp,external_dim_rp))

        return node_coords_xy
    

class GmshGeometry1D(Geometry):
    def __init__(self, gmsh_model, coord_left, coord_right, coord_quadrature= None, weight_quadrature = None, test_function=None, test_function_derivative= None, n_test_func=None, ele_func=None, external_dim_size=None, borders=None):
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
        #self.boundary_normal_global = self.fun_boundary_normal_global()
        self.external_dim_size = external_dim_size
        self.borders = borders
        if external_dim_size:
            self.external_dim = np.linspace(self.borders[0],self.borders[1],self.external_dim_size).reshape(-1,1).astype(np.dtype('f8'))
        self.bbox = (np.array([self.coord_left]), np.array([self.coord_left]))
        super(GmshGeometry1D, self).__init__(
            1, self.bbox, 1
        )

    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

        node_tag, node_coords_all, parametricCoord  = self.gmsh_model.mesh.getNodes(dim=1, tag=-1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(dim=1, tag=-1, includeBoundary=False)[0]

        node_coords_x, node_coords_x_boundary, node_coords_x_inside = self.order_coordinates(node_coords_all, node_tag, node_tag_inside=node_tag_inside)

        if self.external_dim_size:
            node_coords_x_inside = self.add_external_dim(node_coords_x_inside)

        return np.all(np.isin(x, node_coords_x_inside), axis=1)

    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

        node_tag, node_coords_all, _  = self.gmsh_model.mesh.getNodes(dim=1, tag=-1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(dim=1, tag=-1, includeBoundary=False)[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = self.order_coordinates(node_coords_all, node_tag, node_tag_boundary, node_tag_inside)

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
            mask.extend(np.where(np.all(np.isclose(x_i,uniq),axis=1))[0].tolist()) 
        
        return n[mask]

    def random_points(self, n, random="pseudo"):
        """Get collocation points from geometry"""

        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(dim=1, tag=-1, includeBoundary=False)

        node_coords_x, _, _ = self.order_coordinates(node_coords, node_tag)

        if self.external_dim_size:
            node_coords_x = self.add_external_dim(node_coords_x)

        return node_coords_x.astype(config.real(np))
    
    def get_mesh(self):
        """Get the mesh for post-processing"""

        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(2, -1, includeBoundary=True)

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = self.order_coordinates(node_coords, node_tag)

        element_types, element_tags, node_tags = self.gmsh_model.mesh.getElements(2,-1)

        if element_types == 2:
            dol_triangles = node_tags[0].reshape(-1,3) - 1

            offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1, dol_triangles.shape[1])
            cell_types = np.ones(dol_triangles.shape[0])*5
        
        if self.external_dim_size:
            node_coords_xy = self.add_external_dim(node_coords_xy)

        return node_coords_xy, offset, cell_types, dol_triangles

    def random_boundary_points(self, n, random="pseudo"):
        """Get boundary points from geometry"""

        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(dim=1, tag=-1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(dim=1, tag=-1, includeBoundary=False)[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        _, node_coords_x_boundary, _ = self.order_coordinates(node_coords, node_tag, node_tag_boundary, node_tag_inside)

        if self.external_dim_size:
            node_coords_x_boundary = self.add_external_dim(node_coords_x_boundary)

        return node_coords_x_boundary.astype(config.real(np))
    
    def order_coordinates(self, node_coords, node_tag, node_tag_boundary=None, node_tag_inside=None):
        '''Get sorted coordinates and node tags'''

        node_tag -= 1 # gmsh node numbering start with 1 but we need 0         

        node_coords_xyz = node_coords.reshape(-1,3)
        node_coords_x = node_coords_xyz[node_tag.argsort()][:,0:1]
        
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
        #self.mapped_coordinates = []
        #self.jacobian = []
        #self.element_weights = []
        self.mapped_coordinates = np.array([])
        self.jacobian = np.array([])
        self.global_element_weights = np.array([])
        self.global_element_function = np.array([])
        
        self.n_elements = self.gmsh_model.mesh.getElements(1, -1)[1][0].shape[0]
        
        self.global_test_function = np.tile(self.test_function,self.n_elements).astype(config.real(np))
        self.global_test_function = np.expand_dims(self.global_test_function, axis=2)
        self.global_test_function_derivative = np.tile(self.test_function_derivative,self.n_elements).astype(config.real(np))
        self.global_test_function_derivative = np.expand_dims(self.global_test_function_derivative, axis=2)
        
        for element_tag in self.gmsh_model.mesh.getElements(1, -1)[1][0]:
            if self.gmsh_model.mesh.getElement(element_tag)[1].shape[0] > 2:
                raise ValueError("Use linear elements.")
            id_node1 = self.gmsh_model.mesh.getElement(element_tag)[1][0]
            id_node2 = self.gmsh_model.mesh.getElement(element_tag)[1][1]
            
            coordinate_node1 = self.gmsh_model.mesh.getNode(id_node1)[0][0]
            coordinate_node2 = self.gmsh_model.mesh.getNode(id_node2)[0][0]
            
            element_mapped_coordinate = self.get_mapped_coordinates(coordinate_node1, coordinate_node2)
            self.mapped_coordinates = np.hstack((self.mapped_coordinates, element_mapped_coordinate))
            
            element_jacobian = self.get_jacobian(coordinate_node1, coordinate_node2)
            self.jacobian = np.hstack((self.jacobian, element_jacobian))
            
            self.global_element_weights = np.hstack((self.global_element_weights, self.weight_quadrature))
            
            self.global_element_function = np.hstack((self.global_element_function, self.get_element_function(element_mapped_coordinate, element_jacobian)))
            # list version
            #self.mapped_coordinates.append(self.get_mapped_coordinates(coordinate_node1, coordinate_node2))
            #self.jacobian.append(self.get_jacobian(coordinate_node1, coordinate_node2))
            #self.element_weights.append(self.weight_quadrature)
        
        self.mapped_coordinates = self.mapped_coordinates.reshape(-1,1).astype(config.real(np))
        self.jacobian = self.jacobian.reshape(-1,1).astype(config.real(np))
        self.global_element_weights = self.global_element_weights.reshape(-1,1).astype(config.real(np))
        self.global_element_function = self.global_element_function.reshape(-1,1).astype(config.real(np))
            
    def get_mapped_coordinates(self, coordinate_node1, coordinate_node2):
        # linear mapping
        # x_m = N . x --> N linear shape functions N1=1/2(1-psi) N2=1/2(1+psi)
        N1 = 1/2*(1-self.coord_quadrature)
        N2 = 1/2*(1+self.coord_quadrature) 
        mapped_coordinate = N1*coordinate_node1 + N2*coordinate_node2
        
        return mapped_coordinate
    
    def get_jacobian(self, coordinate_node1, coordinate_node2):
        DN1 = -1/2
        DN2 = 1/2
        jacob = DN1*coordinate_node1 + DN2*coordinate_node2
        
        return jacob
    
    def get_element_function(self, element_mapped_coordinate, element_jacobian):
        
        f_quad_element = self.ele_func(element_mapped_coordinate)
        
        element_func = element_jacobian * np.asarray([sum(self.weight_quadrature * f_quad_element * t) for t in self.test_function])
        
        return element_func
 
class GmshGeometryElement(Geometry):
    def __init__(self, 
                 gmsh_model,
                 dimension=1, 
                 coord_quadrature= None, 
                 weight_quadrature = None, 
                 test_function=None, 
                 test_function_derivative= None, 
                 n_test_func=None, 
                 external_dim_size=None, 
                 borders=None,
                 revert_curve_list=None,
                 revert_normal_dir_list=None,
                 only_get_mesh = False):
    
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
            self.external_dim = np.linspace(self.borders[0],self.borders[1],self.external_dim_size).reshape(-1,1).astype(np.dtype('f8'))
        self.bbox = [1,1]
        super(GmshGeometryElement, self).__init__(
            self.dim, self.bbox, 1
        )

    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

        node_tag, node_coords_all, parametricCoord  = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=False)[0]

        node_coords_x, node_coords_x_boundary, node_coords_x_inside = self.order_coordinates(node_coords_all, node_tag, node_tag_inside=node_tag_inside)

        if self.external_dim_size:
            node_coords_x_inside = self.add_external_dim(node_coords_x_inside)

        return self.is_in_tolerance(x, node_coords_x_inside)

    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

        node_tag, node_coords_all, _  = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=False)[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        node_coords_xyz, node_coords_xyz_boundary, node_coords_xyz_inside = self.order_coordinates(node_coords_all, node_tag, node_tag_boundary, node_tag_inside)

        if self.external_dim_size:
            node_coords_xyz_boundary = self.add_external_dim(node_coords_xyz_boundary)
        
        return self.is_in_tolerance(x, node_coords_xyz_boundary)
    
    def is_in_tolerance(self, provided, target):
        
        tolerance = 1e-5
        
        contained_rows = np.all(np.isclose(provided[:, None, :], target, rtol=tolerance, atol=tolerance), axis=2)
        contained_indices_boolean = np.any(contained_rows, axis=1)
        return contained_indices_boolean
    
    def boundary_normal(self, x):
        """Slice the unit normal at x for Neumann or Robin boundary conditions."""

        n, uniq = self.boundary_normal_global

        if self.external_dim_size:
            x = np.delete(x, -1, 1)

        mask = []
        for x_i in x:
            mask.extend(np.where(np.all(np.isclose(x_i,uniq),axis=1))[0].tolist()) 
        
        return n[mask]

    def fun_boundary_normal_global(self):
        """Compute the unit normal on the geometry boundary"""

        fig = plt.figure(figsize=(8, 8), dpi=80)

        node_tag_boundary, node_coords_x_boundary, node_coords_y_boundary, n_x_boundary, n_y_boundary = [],[],[],[],[]
        border = {}
        start = 0

        for geometry_entitiy_pair in self.gmsh_model.getEntities():
            if geometry_entitiy_pair[0] == 1: # if it is a curve
                curve_name = "curve_" + str(geometry_entitiy_pair[1])
                # get node tag, coordinates and parametric coordinates form geometry
                node_tag, node_coords, parameteric_coords = self.gmsh_model.mesh.getNodes(1,geometry_entitiy_pair[1], includeBoundary=True) # dim, curve tag, includeBoundary
                # calculate the first derivative
                dx_dy = self.gmsh_model.getDerivative(1, geometry_entitiy_pair[1], parameteric_coords) # dim, curvetag, parametricCoord
                
                # reshape coordinates and first derivative
                node_coords = node_coords.reshape(-1,3)
                dx_dy = dx_dy.reshape(-1,3)
                # since it is a 2D problem, choose only x and y terms
                node_coords = node_coords[:,0:2]
                dx_dy = dx_dy[:,0:2]

                # normalize the derivative terms (unit)
                dx_dy = dx_dy/np.sqrt(dx_dy[:,0]**2 + dx_dy[:,1]**2)[:,None]
                # get the unit normals
                n_x = dx_dy[:,1]
                n_y = -dx_dy[:,0]

                # store intermediate quantities in the global variables
                node_tag_boundary.extend(node_tag.tolist())
                node_coords_x_boundary.extend(node_coords[:,0].tolist())
                node_coords_y_boundary.extend(node_coords[:,1].tolist())
                n_x_boundary.extend(n_x.tolist())
                n_y_boundary.extend(n_y.tolist())

                # distinguish start/end positio for each curve 
                end = start+node_coords.shape[0]
                border[curve_name] = [start,end]
                start = end
                
                plt.scatter(node_coords[:,0], node_coords[:,1], label=curve_name)
                plt.quiver(node_coords[:,0], node_coords[:,1], n_x, n_y)
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
            curve_list = list(map(str, input("If the reverting of direction is desired, give the curve/s name with space , e.g curve_1 curve_2: ").split()))
            print("\n")

        # if the user gives any curve id, revert the normal direction
        for curve in curve_list:
            if curve in border.keys():
                slice_part = border[curve]
                n_x_boundary[slice_part[0]:slice_part[1]] = n_x_boundary[slice_part[0]:slice_part[1]]*-1
                n_y_boundary[slice_part[0]:slice_part[1]] = n_y_boundary[slice_part[0]:slice_part[1]]*-1 

        # for some nodes, there are two normals available (if it is an intersection point, i.e. sharp edges)
        # thus, one normal direction has to be chosen by user
        # get the unique nodes
        u, idx, c = np.unique(node_tag_boundary, return_counts=True, return_index=True)
        # get the repeated nodes that have more than 1 boundary normal
        repeated_node_tag = u[c>1]

        coords_xy = np.hstack((node_coords_x_boundary.reshape(-1,1),node_coords_y_boundary.reshape(-1,1)))
        n = np.hstack((n_x_boundary.reshape(-1,1),n_y_boundary.reshape(-1,1)))

        # eleminate one of the normals by asking user
        count = 0
        for node in repeated_node_tag:
            location = coords_xy[node_tag_boundary == node]
            assert(np.all(location[0] == location[1]))
            normals = n[node_tag_boundary == node]
            print(f"At location {location[0]} for node {node}, two normals exist such as {normals[0]} and {normals[1]}")
            if self.revert_normal_dir_list:
                desired_normal_id = self.revert_normal_dir_list[count]
                count += 1
            else:
                desired_normal_id = int(input("Please indicate the correct normal: 1 or 2 --> "))
            print(f"At location {location[0]} for node {node} corrected normal is: {normals[desired_normal_id-1]}")
            print("\n")
            if (desired_normal_id == 1) or (desired_normal_id == 2):
                n[node_tag_boundary == node] = normals[desired_normal_id - 1]
            else:
                n[node_tag_boundary == node] = normals[0]
                warnings.warn(f"Not correct id is given, the first normal is set: {normals[0]}")
        
        # save the updated unit normals 
        plt.scatter(coords_xy[:,0], coords_xy[:,1])
        plt.quiver(coords_xy[:,0], coords_xy[:,1], n[:,0], n[:,1])
        normals_picture = os.path.join(os.getcwd(), "normal_directions_updated.png")
        fig.savefig(normals_picture)
        plt.close('all')
        
        # get the unique coordinates and corresponding unit boundary normals of the geometry
        uniq = coords_xy[sorted(idx)]
        n = n[sorted(idx)] 

        return n, uniq

    def random_points(self, n, random="pseudo"):
        """Get collocation points from geometry"""

        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=False)

        node_coords_x, _, _ = self.order_coordinates(node_coords, node_tag)

        if self.external_dim_size:
            node_coords_x = self.add_external_dim(node_coords_x)

        return node_coords_x.astype(config.real(np))
    
    def get_mesh(self):
        """Get the mesh for post-processing"""

        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(self.dim, -1, includeBoundary=True)

        node_coords_xy, node_coords_xy_boundary, node_coords_xy_inside = self.order_coordinates(node_coords, node_tag)

        element_types, element_tags, node_tags = self.gmsh_model.mesh.getElements(self.dim,-1)

        if element_types[0] == 2: # triangle with 3 nodes
            dol_triangles = node_tags[0].reshape(-1,3) - 1

            offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1, dol_triangles.shape[1])
            cell_types = np.ones(dol_triangles.shape[0])*5
            elements = dol_triangles
        
        elif element_types[0] == 3: # quad with 4 nodes
            quads = node_tags[0].reshape(-1,4) - 1
            offset = np.arange(4,quads.shape[0]*quads.shape[1]+1, quads.shape[1])
            cell_types = np.ones(quads.shape[0])*9
            elements = quads
        
        if self.external_dim_size:
            node_coords_xy = self.add_external_dim(node_coords_xy)

        return node_coords_xy, offset, cell_types, elements

    def random_boundary_points(self, n, random="pseudo"):
        """Get boundary points from geometry"""


        node_tag, node_coords, _  = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=True)
        node_tag_inside = self.gmsh_model.mesh.getNodes(dim=self.dim, tag=-1, includeBoundary=False)[0]
        node_tag_boundary = np.setdiff1d(node_tag, node_tag_inside)

        _, node_coords_x_boundary, _ = self.order_coordinates(node_coords, node_tag, node_tag_boundary, node_tag_inside)

        if self.external_dim_size:
            node_coords_x_boundary = self.add_external_dim(node_coords_x_boundary)


        return node_coords_x_boundary.astype(config.real(np))
    
    def order_coordinates(self, node_coords, node_tag, node_tag_boundary=None, node_tag_inside=None):
        '''Get sorted coordinates and node tags'''

        node_tag -= 1 # gmsh node numbering start with 1 but we need 0         

        node_coords_xyz = node_coords.reshape(-1,3)
        node_coords_x = node_coords_xyz[node_tag.argsort()][:,0:self.dim]
        
        node_coords_x_boundary = None
        node_coords_x_inside = None

        if node_tag_boundary is not None:
            node_tag_boundary -= 1
            node_coords_x_boundary = node_coords_x[node_tag_boundary]
        if node_tag_inside is not None:
            node_tag_inside -= 1 
            node_coords_x_inside = node_coords_x[node_tag_inside]

        return node_coords_x, node_coords_x_boundary, node_coords_x_inside
        # return node_coords_x.astype(config.real(np)) if node_coords_x is not None else node_coords_x, \
        #         node_coords_x_boundary.astype(config.real(np)) if node_coords_x_boundary is not None else node_coords_x_boundary, \
        #         node_coords_x_inside.astype(config.real(np)) if node_coords_x_inside is not None else node_coords_x_inside
    
    def get_element_info(self):
        #self.mapped_coordinates = []
        #self.jacobian = []
        #self.element_weights = []
        
        self.n_elements = self.gmsh_model.mesh.getElements(self.dim, -1)[1][0].shape[0]
        
        self.mapped_coordinates = np.empty((self.n_elements*self.n_gp,self.dim))
        self.global_test_function = np.empty((self.n_test_func, self.n_elements, self.n_gp, self.dim))
        self.global_test_function_derivative = np.empty((self.n_test_func, self.n_elements, self.n_gp, self.dim))
        self.jacobian = np.empty((self.n_elements,self.n_gp,1))
        self.global_element_weights = np.empty((self.n_elements, self.n_gp, self.dim))
        
        element_id = 0
        
        for element_tag in self.gmsh_model.mesh.getElements(self.dim, -1)[1][0]:
            if self.gmsh_model.mesh.getElement(element_tag)[1].shape[0] > self.dim*2:
                raise ValueError("Use linear elements.")
            
            coordinate_list = []
            for dof in range(self.dim*2):
                node_id = self.gmsh_model.mesh.getElement(element_tag)[1][dof]
                coordinate_list.append(self.gmsh_model.mesh.getNode(node_id)[0][0:self.dim])
            
            element_mapped_coordinate = self.get_mapped_coordinates(coordinate_list)
            self.mapped_coordinates[element_id*self.n_gp:(element_id+1)*self.n_gp,:] = element_mapped_coordinate
            
            element_jacobian = self.get_jacobian(coordinate_list)
            self.jacobian[element_id] = element_jacobian
            
            self.global_test_function[:,element_id] = self.test_function.copy()
            self.global_test_function_derivative[:,element_id] = self.test_function_derivative.copy()
            self.global_element_weights[element_id] = self.weight_quadrature.copy()
            
            element_id += 1

        self.jacobian = self.jacobian.reshape(self.jacobian.shape[0]*self.jacobian.shape[1],
                                              self.jacobian.shape[2])
        self.global_element_weights = self.global_element_weights.reshape(self.global_element_weights.shape[0]*self.global_element_weights.shape[1],
                                                                          self.global_element_weights.shape[2])
        self.global_test_function = self.global_test_function.reshape(self.global_test_function.shape[0],
                                                                      self.global_test_function.shape[1]*self.global_test_function.shape[2],
                                                                      self.global_test_function.shape[3])
        self.global_test_function_derivative = self.global_test_function_derivative.reshape(self.global_test_function_derivative.shape[0],
                                                                                            self.global_test_function_derivative.shape[1]*self.global_test_function_derivative.shape[2],
                                                                                            self.global_test_function_derivative.shape[3])
                                
        self.mapped_coordinates = self.mapped_coordinates.astype(config.real(np))
        self.jacobian = self.jacobian.astype(config.real(np))
        self.global_element_weights = self.global_element_weights.astype(config.real(np))
        self.global_test_function = self.global_test_function.astype(config.real(np))
        self.global_test_function_derivative = self.global_test_function_derivative.astype(config.real(np))
            

    def get_mapped_coordinates(self, coordinate_list):
        # linear mapping
        # x_m = N . x --> N linear shape functions N1=1/2(1-psi) N2=1/2(1+psi)
        if self.dim == 1:
            N1 = 1/2*(1-self.coord_quadrature)
            N2 = 1/2*(1+self.coord_quadrature) 
            mapped_coordinate = N1*coordinate_list[0] + N2*coordinate_list[1]
            
            return mapped_coordinate
        
        elif self.dim == 2:
            # linear mapping
            # x_m = N . x --> N linear shape functions N1=1/2(1-psi) N2=1/2(1+psi)
            psi_x = self.coord_quadrature[:,0:1]
            psi_y = self.coord_quadrature[:,1:2]
            
            N1 = 1/4*(1-psi_x)*(1-psi_y)
            N2 = 1/4*(1+psi_x)*(1-psi_y)
            N3 = 1/4*(1+psi_x)*(1+psi_y)
            N4 = 1/4*(1-psi_x)*(1+psi_y)
            
            N_stack = np.hstack((N1,N2,N3,N4))
            x_stack = np.vstack((coordinate_list[0][0:1],coordinate_list[1][0:1],coordinate_list[2][0:1],coordinate_list[3][0:1]))
            y_stack = np.vstack((coordinate_list[0][1:2],coordinate_list[1][1:2],coordinate_list[2][1:2],coordinate_list[3][1:2]))
            
            mapped_coordinate_x = np.matmul(N_stack, x_stack)
            mapped_coordinate_y = np.matmul(N_stack, y_stack)
        
            return np.hstack((mapped_coordinate_x, mapped_coordinate_y))
    
    def get_jacobian(self, coordinate_list):
        if self.dim == 1:
            DN1 = -1/2*np.ones((self.n_gp,1))
            DN2 = 1/2*np.ones((self.n_gp,1))
            jacob = DN1*coordinate_list[0] + DN2*coordinate_list[1]
        
        if self.dim == 2:
            psi_x = self.coord_quadrature[:,0:1]
            psi_y = self.coord_quadrature[:,1:2]
            
            DN1_psi_x = -1/4*(1-psi_y)
            DN2_psi_x = 1/4*(1-psi_y)
            DN3_psi_x = 1/4*(1+psi_y)
            DN4_psi_x = -1/4*(1+psi_y)
            

            DN1_psi_y = -1/4*(1-psi_x)
            DN2_psi_y = -1/4*(1+psi_x)
            DN3_psi_y = 1/4*(1+psi_x)
            DN4_psi_y = 1/4*(1-psi_x)
            
            DN_stack_upper = np.hstack((DN1_psi_x,DN2_psi_x,DN3_psi_x,DN4_psi_x))
            DN_stack_lower = np.hstack((DN1_psi_y,DN2_psi_y,DN3_psi_y,DN4_psi_y))
            

            x_stack = np.vstack((coordinate_list[0][0:1],coordinate_list[1][0:1],coordinate_list[2][0:1],coordinate_list[3][0:1]))
            y_stack = np.vstack((coordinate_list[0][1:2],coordinate_list[1][1:2],coordinate_list[2][1:2],coordinate_list[3][1:2]))
            
            J11 = np.matmul(DN_stack_upper, x_stack)
            J12 = np.matmul(DN_stack_upper, y_stack)
            J21 = np.matmul(DN_stack_lower, x_stack)
            J22 = np.matmul(DN_stack_lower, y_stack)
            
            jacob = J11*J22 - J12*J21
            
            #print(jacob)
 
        return jacob