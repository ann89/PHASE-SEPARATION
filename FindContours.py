
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab
from skimage import measure
import os
from scipy.optimize import curve_fit
import pandas as pd
import time
import sys
import MDAnalysis

""" 
Use the Director parameters (n) to distinguish if the lipid is in the Lo or Ld phase.
Find the line interface between the two phases.
- We identify the location of this interface at each instant. This location is found
  with a 2D version of the 3D method described in (Willard & Chandler (2010) Instantaneous liquid interfaces
  . The journal of Physical Chemistry B 114:1954-1958). Specifically,
  the interface is here defiend as a line in the plane where the value of a coarse-grained field characterizing the local 
  director order is intermediate between those of the ordered and disordered phases.
  The field is defined as
  \begin{equation}
  \bar {n}(\mathbf{r},t) = \sum_l f(\mathbf{r} - \mathbf{r}_l ; \epsilon) n_l
  \end{equation}
  where $\mathbf{r}_l$ is the (x,y) position of the lth tail-end particle, and $ f(\mathbf{r} - \mathbf{r}_l ; \epsilon)$
  is a coarse-grained delta-like function in the two-dimensional space,
  \begin{equation}
  f(\mathbf{r} - \mathbf{r}_l ; \epsilon) =  (1/ 2\pi \epsilon^2) exp (- |\mathbf{r} -\mathbf{r}_l|^2/ 2\epsilon^2)
  \end{equation}
  The  field variable, $\mathbf{r}$, ias two-dimensional vector specifying a position in the plane of the bilayer.
  The coarse-grained width, $\epsilon$, is chosen to be the average separation between tail-end particles $l$ and $j$
  when $\left \langle ( n_l - \left \langle n_l \right \rangle) (n_j - \left \langle n_j \right \rangle ) \right \rangle / \left \langle ( n_l - \left \langle n_l \right \rangle)^2 \right \rangle $
  in the ordered phase is $1/10$. This choice yields a value of $\epsilon = 1$ nm.
  For numerics, a square lattice tiles the average plane of the bilayer, and the coarse-grained field is evaluated at each lattice node.
  For convenience, the coarse-grained function is truncated and shifted to zero at $3 \epsilon$.
  The instantaneous order-disorder interface is identified by interpolating between these adjacent lattice nodes to find the set of points
  $\mathbf{s}$ satisfying $ n(\mathbf{s}, t) = (n_d + n_o)/2. Here  $n_d$ and $n_o$ are $\left \langle n(\mathbf{r})  \right \rangle $
  evaluated in the disordered and ordered phases, respectively.
  
  Numerically, here we use a Marching Cubes algorithm (William E. Lorensen, Harvey E. Cline: Marching Cubes: A high resolution 3D surface construction algorithm. In: Computer Graphics, Vol. 21, Nr. 4, July 1987) 
  to perform the interpolation. 
  
  *** Marching Cubes algorithmas  implemented in scipy.optimize ***
  see also http://www.cs.carleton.edu/cs_comps/0405/shape/marching_cubes.html
"""

def func(x, y, p_x, p_y, e):
    f = np.zeros(x.shape)
    e2 = e**2
    r_min_p_2 = (x-p_x)**2 + (y-p_y)**2
    boolian_if_in = r_min_p_2 <= 9 * e2 
    f[boolian_if_in] = ((1./(np.pi*2.*e2)) * np.exp(-r_min_p_2[boolian_if_in]/(2.*e2))) 
    return f 


def order_vector_field(Lx, Ly, pos_xy, phi, e, box):
    start_time = time.time()
    coordsx = np.linspace(0, box[0], Lx)
    coordsy = np.linspace(0, box[1], Ly)
 
    coordsx_help_low =np.linspace(-1 * box[0], 0, Lx)
    coordsy_help_low =np.linspace(-1 * box[1], 0, Ly)
 
    coordsx_help_up =np.linspace(box[0], 2 * box[0], Lx)
    coordsy_help_up =np.linspace(box[1], 2 * box[1], Ly)        
     
    X, Y = np.meshgrid(coordsx, coordsy)
    X_help_low, Y_help_low = np.meshgrid(coordsx_help_low, coordsy_help_low)
    X_help_up, Y_help_up = np.meshgrid(coordsx_help_up, coordsy_help_up)

    Vxy = np.zeros((np.shape(X)))

     
    for i in range(len(pos_xy)):
        func_x_y = func(X, Y, pos_xy[i,0], pos_xy[i,1], e)
        func_x_y_up =  func(X, Y_help_up, pos_xy[i,0], pos_xy[i,1], e)
        func_x_y_low = func(X, Y_help_low, pos_xy[i,0], pos_xy[i,1], e)
        
        func_x_up_y_up =  func(X_help_up, Y_help_up, pos_xy[i,0], pos_xy[i,1], e)
        func_x_up_y =     func(X_help_up, Y, pos_xy[i,0], pos_xy[i,1], e)
        func_x_up_y_low = func(X_help_up, Y_help_low, pos_xy[i,0], pos_xy[i,1], e)
        
        func_x_low_y_up =  func(X_help_low, Y_help_up, pos_xy[i,0], pos_xy[i,1], e)
        func_x_low_y =     func(X_help_low, Y, pos_xy[i,0], pos_xy[i,1], e)
        func_x_low_y_low = func(X_help_low, Y_help_low, pos_xy[i,0], pos_xy[i,1], e)
        
        Vxy        += phi[i]*(func_x_up_y_up + func_x_up_y + func_x_up_y_low +
                              func_x_y + func_x_y_up + func_x_y_low +
                              func_x_low_y_up + func_x_low_y + func_x_low_y_low)
        

    Vxy_total = Vxy 
    print("--- %s seconds function ---" % (time.time() - start_time))    
    return Vxy_total

    
    


# =============================================================================
#PLOTS
def plot_scatter_order_field(pos_xy, resid, dlipc_res, dspc_res, chl_res, ssm_res, n,box, ts, side):
    # assumes that the lipids are in the following order: DLIPC, DSPC, SSM, CHL 

        img =plt.scatter(np.array(pos_xy[:, 0]), np.array(pos_xy[:, 1]), c=(n[:,0]), s=50,cmap=plt.cm.GnBu) # c=(n[:,0]), , vmin=-0.5,vmax=1.0
        cb=plt.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.1) 

        
        dlipc = np.where(np.logical_and(resid>=np.min(dlipc_res), resid<=np.max(dlipc_res)))[0] #min(dlipc_res) max(dlipc_res)
        dspc = np.where(np.logical_and(resid>=np.min(dspc_res), resid<=np.max(dspc_res)))[0] #min(dspc_res)
        ssm  = np.where(np.logical_and(resid>=np.min(ssm_res), resid<=np.max(ssm_res)))[0]
        if side == 'up':
            chl = np.where(resid>np.max(dspc_res)) [0] #max(dspc_res) np.logical_or(resid<np.max(dlipc_res),
        elif side == 'down':
            chl = np.where(np.logical_or(resid<np.min(dlipc_res), resid>np.max(dspc_res) )) [0]
        
        plt.scatter(pos_xy[dlipc,0],pos_xy[dlipc,1],facecolors='none', edgecolors='#CC0000', s=110, lw=1.5) #red
        plt.scatter(pos_xy[dspc,0],pos_xy[dspc,1],facecolors='none', edgecolors='black', s=110, lw=1.5) #black
        plt.scatter(pos_xy[ssm,0],pos_xy[ssm,1],facecolors='none', edgecolors='#00994C', s=120, lw=1.5) #green
        plt.scatter(pos_xy[chl,0],pos_xy[chl,1],facecolors='#F5C816', edgecolors='none', s=110, lw=1.5) #yellow
                    

        
        plt.xlim(0, box[0])
        plt.ylim(0, box[1])       
        plt.xlabel(r'x [$\AA$]', fontsize=20)
        plt.ylabel(r'y [$\AA$]', fontsize=20)
        cb.set_label(label='n(' r'$\theta$' ')',size=20)
        cb.ax.tick_params(labelsize=20)
        plt.tick_params(axis='x', pad=8)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.axis('equal')
        plt.axis('off')

        if side == 'up' :
            plt.savefig(output_dir + 'Directors-upper_frame-interface' + str(ts) + '.png', dpi=300)
        else :
            plt.savefig(output_dir + 'Directors-lower_frame-interface' + str(ts) + '.png', dpi=300)


def calculate_contours_fit(L_x, L_y, e, leaflet, ts, Plots, side):
    """ read the input data """
    
    n = np.load(input_dir + 'directors_'+leaflet+'_tail_'+ str(ts) + '.npy') 

    pos = np.load(input_dir + 'coordinates_'+leaflet+'_tail_' + str(ts) + '.npy')               

    resid = np.load(input_dir + 'residues_'+leaflet+'_tail_' + str(ts) + '.npy')
    box = np.load(input_dir + 'box' + str(ts) + '.npy')

    
    chl = np.load(input_dir + 'cholesterol_'+leaflet+'_tail_' + str(ts) + '.npy')
    dlipc = np.load(input_dir + 'dlipc_'+leaflet+'_tail_'  + str(ts) + '.npy')  
    dspc = np.load(input_dir + 'dspc_'+leaflet+'_tail_'  +  str(ts) + '.npy')
    ssm = np.load(input_dir + 'ssm_'+leaflet+'_tail_'  +  str(ts) + '.npy')
    
    #n= np.ones(len(pos))
    """ END: read the input data """


    field =  order_vector_field(L_x, L_y, pos, n, e, box)

    c = pd.DataFrame(data=field).mean(axis=0).rolling(50, center=True, min_periods=1).mean() #50
    c.dropna(inplace=True)
    middle = 0.5*(np.max(c) + np.min(c))  
    #middle = 0.025
    contours = measure.find_contours(field, middle) # Marching Cubes algorith
    #save contours
    fac_x = box[0] / L_x #to get the right dimensions (range_x)
    fac_y = box[1] / L_y # (range_y)
    
    contours_x = []
    contours_y = []
    contours_x_y = []
    
    contours_all = []
    for m, contour in enumerate(contours):
        contours_x.append((contour[:, 1] * fac_x))
        contours_y.append((contour[:, 0] * fac_y))
        
        
        contours_x_y = np.column_stack((contours_x[m], contours_y[m]))
        contours_all.append(contours_x_y)
        np.save(output_contours + 'contours_'+leaflet+'.' + str(ts) + '.npy', contours_all)
            

#===================================================
#To assign resids to the different phases
    phase_belonging = np.zeros((len(pos)))
    ordered =[]
    disordered = []
    for i in np.arange(len(pos)):
        
        def apply_pbc(pos, box):
            if pos >= box:
                pos -= box
            if pos < 0:
                pos += box
            return pos
        
        idx_x = int(apply_pbc(pos[i,0], box[0]) / fac_x - 1.e-5) #the  - 1.e-5 is because accuracy issue in the /
        idx_y = int(apply_pbc(pos[i,1], box[1]) / fac_y - 1.e-5) #this - 1.e-5 is because accuracy issue in the /
        #print(idx_x, idx_y)
        order= field[idx_y, idx_x]
        if (order > middle):
            ordered.append(order)
            order = 1 #ordered lipids
            
        else :
            disordered.append(order)
            order =0   #disordered lipids
        phase_belonging[i] = order
        

    resid_phases = np.column_stack((resid[:,0], phase_belonging))
    np.save(output_dir + 'resid_phases'+leaflet+'.'+ str(j) + '.npy', resid_phases)

    if Plots == True:
        plt.figure(figsize=(15,10)) 
        
        contours_sorted = sorted(contours, key=len, reverse=True)
        
        for i in range(2):
            plt.plot(contours_sorted[i][:,1]* fac_x+0.5*fac_x, contours_sorted[i][:,0]* fac_y+0.5*fac_y, linewidth=3, color='#0000FF' ) ##00CC00
            
        #for m, contour in enumerate(contours_sorted):
        #    print(contour[:,0])
        #    for contour in contours: 
                
        #        plt.plot((contour[:, 1] * fac_x+0.5*fac_x),
        #                 (contour[:, 0] * fac_y+0.5*fac_y),
        #                 linewidth=4, color='#00CC00')
              
        plt.imshow(field, interpolation='nearest',  
                   cmap=plt.cm.gray_r,
                   extent=[0, box[0], 0, box[1]], origin='lower', alpha=0.7) 
    
        plt.axis('off')
        plot_scatter_order_field(pos, resid, dlipc, dspc, chl,ssm, n , box, ts, side) #phase_belonging.reshape(-1,1)
        plt.savefig(output_dir + 'contours-'+ leaflet + str(ts) + '.png', dpi=300)  
        plt.close()            
    
    return resid_phases #, ordered, disordered   
    
                  
# MAIN
#TODO
side = sys.argv[1] #up for upper leaflet, down for lower leaflet

input_dir = "ANALYSIS/directors/"
output_dir = "ANALYSIS/directors/plots/"
output_contours = "ANALYSIS/directors/contours/"
#read the vectors
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
if not os.path.exists(output_contours):
    os.makedirs(output_contours)


start_time = time.time()

e = 10 
L_x = 274
L_y = 91

top  = 'ANALYSIS/recentered_x.gro'
traj = 'ANALYSIS/recentered_x.xtc'
u = MDAnalysis.Universe(top,traj) 
for j in range(0,u.trajectory.n_frames,1): # u.trajectory.n_frames = 21

# =============================================================================
    if side == 'up':
        resid_phases_up = calculate_contours_fit(L_x, L_y, e, leaflet="upper", ts=j, Plots =  True, side= side) #, ordered, disordered
    if side == 'down':        
        resid_phases_down = calculate_contours_fit(L_x, L_y, e, leaflet="lower", ts=j, Plots =  True, side= side) #, ordered, disordered


print("--- %s seconds ---" % (time.time() - start_time))
