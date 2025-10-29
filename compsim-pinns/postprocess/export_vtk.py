import numpy as np

def meshGeometry(geom, numberOfPointsOnBoundary=100):
    
    import triangle as tr
    
    boun = geom.uniform_boundary_points(numberOfPointsOnBoundary)  
    
    area = 1.0 / (numberOfPointsOnBoundary * numberOfPointsOnBoundary)
    
    mesh = tr.triangulate(dict(vertices=boun), 'qa'+str(area))
    
    return mesh['vertices'], mesh['triangles']

def solutionFieldOnMeshToVtk(X, triangles, pointData, file_path):
    
    from pyevtk.hl import unstructuredGridToVTK
    
    x = X[:,0].flatten()
    y = X[:,1].flatten()
    z = np.zeros(y.shape)
    
    dol_triangles = triangles
    offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1]).astype(dol_triangles.dtype)
    cell_types = np.ones(dol_triangles.shape[0])*5
    
    unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                          cell_types, pointData=pointData)
    
    