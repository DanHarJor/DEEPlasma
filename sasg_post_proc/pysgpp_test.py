import sys
sys.path.append('./')
sys.path.append('./SGpp_local')
# sys.path.append('SGpp_local/datadriven')
# from uq.learner.builder.StopPolicyDescriptor import StopPolicyDescriptor
# from datadriven.python.learner.LearnerBuilder import StopPolicyDescriptor
# from datadriven.python.learner.LearnerBuilder import StopPolicyDescriptor
# from SGpp_local.datadriven.python.uq.operations.sparse_grid import insertTruncatedBorder
import numpy as np
import pysgpp
# from pysgpp.extensions.datadriven.learner.LearnerBuilder import LearnerBuilder
# StopPolicyDescriptor = LearnerBuilder.StopPolicyDescriptor

import pysgpp.extensions.datadriven.uq.analysis.asgc

# import SGpp_local.datadriven.python.uq.analysis.asgc
#anova.hdmrAnalytic import HDMRAnalytic


for i in dir(pysgpp):
    if 'extensions' in i:
       print(i)
# from pysgpp.extensions.datadriven.uq.analysis.asgc.stop import StopPolicyDescriptor


# from datadriven.python.learner.LearnerBuilder import LearnerBuilder

# StopPolicyDescriptor = LearnerBuilder.StopPolicyDescriptor


# from pysgpp.extensions.datadriven.learner.StopPolicyDescriptor import StopPolicyDescriptor


# hdmr = HDMRAnalytic()
# from pysgpp.datadriven.python.uq.operations.sparse_grid import *
# import deepcopy
# from pysgpp import GridPoint
# from hdmr import HDMRAnalytic
# from pysgpp.extensions.datadriven.uq.analysis import SensitivityAnalyzer

# dim=12
# grid = pysgpp.Grid.createPolyBoundaryGrid(dim,3)
# gs = grid.getStorage()
# level=1
# print('making static grid, level', level)
# grid.getGenerator().regular(level)
# # alpha = pysgpp.DataVector(grid.getStorage().getSize())

# def f(point):
#     pol = []
#     for d in point:
#         pol.append(2*d**d)
#     return np.sum(pol) 

# for i in range(grid.getStorage().getSize()):
#     gp = grid.getStorage().getPoint(i)    
#     unit_point = ()
#     for j in range(dim):
#         unit_point = unit_point + (gp.getStandardCoordinate(j),)
        
#     alpha[i] = f(unit_point)

# pysgpp.createOperationHierarchisation(grid).doHierarchisation(alpha)

# print('before add boarder',gs.getSize())

# def add_boundary_points(grid):
#     storage = grid.getStorage()
#     generator = grid.getGenerator()
#     dim = storage.getDimension()

#     set
#     for i in range(storage.getSize()):
#         gp = storage.getPoint(i)
#         for d in range(dim):
#             level = gp.getLevel(d)
#             if level==1:
#                 # Left boundary
#                 left_gp = pysgpp.HashGridPoint(gp)
#                 left_gp.set(d, level, 0)
#                 if not storage.isContaining(left_gp):
#                     storage.insert(left_gp)

#                 # Right boundary
#                 right_gp = pysgpp.HashGridPoint(gp)
#                 right_gp.set(d, level, 2 ** level)
#                 if not storage.isContaining(right_gp):
#                     storage.insert(right_gp)
                    
# add_boundary_points(grid)
# print('after add boarder',gs.getSize())


# def quadrature_integral(grid, alpha):
#     op_quad = pysgpp.createOperationQuadrature(grid)
#     unit_integral = op_quad.doQuadrature(alpha)
#     return unit_integral

# def compute_basis_function_volumes(grid):
#     grid_storage = grid.getStorage()
#     basis = grid.getBasis()
#     volumes = []

#     for i in range(grid_storage.getSize()):
#         gp = grid_storage.getPoint(i)
#         volume = 1.0
#         for d in range(grid_storage.getDimension()):
#             level = gp.getLevel(d)
#             index = gp.getIndex(d)
#             volume *= basis.getIntegral(level, index)
#         volumes.append(volume)
#     return volumes

# print(dir(gp))
        
# volumes = compute_basis_function_volumes(grid)
# print(volumes)

# print(quadrature_integral(grid, alpha))
# analyzer = pysgpp.SensitivityAnalyzer(grid, alpha)


# gp = grid.getStorage().getPoint(0)

# new_grid = pysgpp.Grid.createModPolyGrid(12, 3)

# print('Size_before',new_grid.getStorage().getSize())
# new_grid.getStorage().insert(pysgpp.HashGridPoint(gp)) #Hash maybe not needed 
# print('Size_after',new_grid.getStorage().getSize())


# levels = [gp.getLevel(d) for d in range(gp.getDimension())]
# print(levels)