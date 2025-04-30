import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
import os
        
def plot_slices(function=None, dimension_labels=None, bounds=None, grid_size=200, nominals=None, not_vectorised=False):
    if type(bounds) == type(None):
        bounds = np.repeat((0,1), len(dimension_labels))

    if type(nominals) == type(None):
        nominals = [np.mean(b) for b in bounds]
    h=3
    w=3
    r=1
    c=len(bounds)
    fig, AX = plt.subplots(r,c,figsize=(w*c, h*r), dpi = 200)
    for i, b in enumerate(bounds):
        p = np.stack([nominals for i in range(grid_size)])
        x = np.linspace(b[0],b[1], grid_size)
        p[:,i] = x
        if not_vectorised:
            y = []
            for pi in p:
                y.append(function(pi))
        else:
            y = function(p)
        AX[i].plot(x,y, color ='black')
        # AX[i].legend()
        if type(dimension_labels) != type(None):
            AX[i].set_xlabel(dimension_labels[i])
        else:
            AX[i].set_xlabel(f'dimension {i}')
        AX[i].set_ylabel('function value')
        fig.tight_layout()
        fig.show()
    return fig

def plot_matrix_contour(function, bounds, points=None, dimension_labels=None):
    num_dim = len(bounds)
    w=2
    h=2
    figure, AX = plt.subplots(num_dim, num_dim, figsize=(w*num_dim, h*num_dim), sharex='col', sharey='row')
    Z_all = []
    axij = []
    for i in range(num_dim):
        for j in range(num_dim):
            if i==j:
                axij.append(AX[i,j])
            if j>=i:
                figure.delaxes(AX[i,j])
                # break
            else:
                Zi, contour = plot_2D_of_many(which2=(j,i), bounds=bounds, points=points ,ax=AX[i,j], style='contour', grid_size=50, function=function)
                Z_all.append(Zi)
                if j==0:
                    if type(dimension_labels) != type(None):
                        AX[i,j].set_ylabel(dimension_labels[i])
                    else:
                        AX[i,j].set_ylabel(f'{i}')
                if i==num_dim-1:
                    if type(dimension_labels) != type(None):
                        AX[i,j].set_xlabel(dimension_labels[j])
                    else:
                        AX[i,j].set_xlabel(f'{j}')
    Z_all = np.array(Z_all)
    # print(type(Z_all), Z_all.shape, Z_all)
    # print('Z_all', Z_all)
    # print('Z all', np.min(Z_all.flatten()), Z_all.flatten().min())
    # print('Z all', np.max(Z_all.flatten()), Z_all.flatten().max())
    
    # Add a single color bar for all subplots, ensuring it includes values from both contour plots
    # if Z_all.min() != Z_all.max():
    #     cbar = figure.colorbar(contour, ax=AX, orientation='vertical', location='right')
    #     cbar.set_label('function evaluation')
    #     # Update the color bar to include values from both contour plots
    #     cbar.mappable.set_clim(vmin=np.min(Z_all.flatten()), vmax=np.max(Z_all.flatten()))
    figure.show()
    return figure

    
def plot_2D_of_many(which2, function, bounds, points=None, extra=0, plot_bounds=None, nominals=None, grid_size=100, style='3D', ax=None):
    # which2 is a sequence that specifies which dimensions to plot, the rest are kept nominal, example which2 = (0,2) to plot the 1st and 3rd dimensions. 
    if type(points)!=type(None):
        points = np.array(points) # assumes shape num_points,num_dim
        points_2d = np.array([points.T[which2[0]],points.T[which2[1]]])
        
    if plot_bounds == None:
        plot_bounds = bounds
    if type(nominals) == type(None):
        nominals = [np.mean(b) for b in bounds]
    
    xlow, xhigh = plot_bounds[which2[0]][0]-extra, plot_bounds[which2[0]][1]+extra
    ylow, yhigh = plot_bounds[which2[1]][0]-extra, plot_bounds[which2[1]][1]+extra
    
    x = np.linspace(xlow, xhigh, grid_size)
    y = np.linspace(ylow, yhigh, grid_size)
    X, Y = np.meshgrid(x, y)
    
    arrays2stack = []
    for i, n in enumerate(nominals):
        arrays2stack.append(np.full_like(X,n))
    arrays2stack[which2[0]] = X
    arrays2stack[which2[1]] = Y
    pos = np.dstack(arrays2stack)
    
    # print(pos.shape)
    Z = np.zeros(shape=(grid_size,grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            p = nominals
            p[which2[0]] = X[i,j]
            p[which2[1]] = Y[i,j]
            Z[i,j] = function(p)
    # print('Z',Z)
    # print(np.max(Z), np.min(Z))
    if style == '3D':
        if type(ax) == type(None):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')    
        ax.plot_surface(X, Y, Z, cmap='viridis')
    else:
        if type(ax) == type(None):
            fig = plt.figure(figsize=(2,2), dpi=200)
            ax = fig.add_subplot(111)
        # contour = ax.contourf(X,Y,Z, cmap='viridis', levels=100)
        contour = ax.contour(X,Y,Z, cmap='viridis')
        if type(points)!=type(None):
            ax.scatter(points_2d[0], points_2d[1], marker='+', color='red')
    
    if type(ax) == type(None):
        ax.set_xlabel(f'{which2[0]}')
        ax.set_ylabel(f'{which2[1]}')
        fig.show()
    return Z, contour
        
def plot_2d(function, bounds, ax, grid_size=100, onlyContour=False, plot_bounds=None, extra=0, sample_points=None, title=None):
    if plot_bounds == None:
        plot_bounds = bounds
    num_dim = len(bounds)
    if num_dim != 2:
        raise ValueError('Daniel Says: n_dim must equil 2')
    xlow, xhigh = plot_bounds[0][0]-extra, plot_bounds[0][1]+extra
    ylow, yhigh = plot_bounds[1][0]-extra, plot_bounds[1][1]+extra
    x = np.linspace(xlow, xhigh, grid_size)
    y = np.linspace(ylow, yhigh, grid_size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
            
    
    Zmax = function(pos)
    
    # z = []
    # yy = 0.5
    # x_at_y = [(xi, yy) for xi in x]
    # for g in gaussians:
    #     z.append(g.pdf(x_at_y))
    # zmax = np.max(np.stack(z), axis = 0)
    # #slice
    if not onlyContour:
        # plt.figure()
        # plt.plot(x, zmax)
        # plt.show()
        
        fig = plt.figure()
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_3d.plot_surface(X, Y, Zmax, cmap='viridis')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Function Value')
        # ax_3d.set_title('2D Surface of Multimodal Multivariate Gaussian Distribution')
        ax_3d.view_init(elev=30, azim=30-90)
                
    if title != None:
        ax.set_title(title)
    if type(sample_points) != type(None):
        ax.scatter(*sample_points, marker='.')
    
    ax.contour(X,Y,Zmax)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
