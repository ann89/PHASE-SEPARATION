#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 13:20:18 2018

@author: anna

Compute the lateral self diffusion coefficients for lipid in the Lo and Ld phase
INPUT: no jump trajectory and phase assignment
IMPORTANT: the scripts consider the motion only of lipids that do not cross the interface (always in Ld or always in Lo).
Hence, the MSD is computedonly for these lipids.

We compute here also the Displacement autocorrelation function as defined  in DOI:https://doi.org/10.1103/PhysRevLett.109.188103
This is checked against a  Fractional Langevin Equation (FLE) and continous time random walks (CTRW) type of subdiffusion. 
"""

import numpy as np
import scipy
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis
import os

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


input_dir = "ANALYSIS/directors/plots/"

output_dir = "ANALYSIS/Diffusion/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
    
top = 'nojump.pdb'
traj = 'nojump.xtc' 
#read the vectors
u = MDAnalysis.Universe(top,traj)

 
def fit_linear_diffusion_data(MSD_data_array,degrees_of_freedom=2):
#     
# 
     coefficient_dictionary = {1:2.,2:4.,3:6.} #dictionary for mapping degrees_of_freedom to coefficient in fitting equation
     coefficient = coefficient_dictionary[degrees_of_freedom]
# 
     x_data_array = np.arange(len(MSD_data_array))
     y_data_array = MSD_data_array
     z = np.polyfit(x_data_array,y_data_array,1) #fit a polynomial of any degree (x,y, degree)
     slope, intercept = z
     diffusion_constant = slope / coefficient
     p = np.poly1d(z)
     #this is actually for the fit:
     sample_fitting_data_X_values_nanoseconds = np.linspace(x_data_array[0],x_data_array[-1],100)
     sample_fitting_data_Y_values_Angstroms = p(sample_fitting_data_X_values_nanoseconds)
     return (diffusion_constant, sample_fitting_data_X_values_nanoseconds, sample_fitting_data_Y_values_Angstroms)




def get_unique_ele(dic_ele, unique_ele):
    mask = np.isin(dic_ele[:, 0], unique_ele)
    return dic_ele[mask]




dict_lower = {} # Create an empty dict
for i in range(0, u.trajectory.n_frames, 1):
    lower= np.load(input_dir + 'resid_phaseslower.' + str(i) + '.npy')
    lower_unique = np.unique(lower[:,0], return_index=True) #index

    dict_lower[i] = lower[lower_unique[1]]   #fill the dictionary with vectors
#   file_dict[i] = upper[upper_unique[1]]

dict_upper = {} # Create an empty dict
for i in range(0, u.trajectory.n_frames, 1):
    upper= np.load(input_dir + 'resid_phasesupper.' + str(i) + '.npy')
    upper_unique = np.unique(upper[:,0], return_index=True) #index
    upper_lower = np.concatenate((lower[lower_unique[1]], upper[upper_unique[1]]), axis=0)
    dict_upper[i] = upper[upper_unique[1]]   #fill the dictionary with vectors
#   file_dict[i] = upper[upper_unique[1]]


#take only the common resids for all the t
from functools import reduce
common_elements_upper=reduce(np.intersect1d, ([dict_upper[ele][:,0] for ele in dict_upper]))
common_elements_lower=reduce(np.intersect1d, ([dict_lower[ele][:,0] for ele in dict_lower]))


#take only the resids that do not change phase (i.e., always ordered or always disordered)
common_mols_upper = [get_unique_ele(dict_upper[ele], common_elements_upper) for ele in dict_upper]
order_time_evo_up = np.vstack([ele[:, 1] for ele in common_mols_upper] )

common_mols_lower = [get_unique_ele(dict_lower[ele], common_elements_lower) for ele in dict_lower]
order_time_evo_down = np.vstack([ele[:, 1] for ele in common_mols_lower] )



list_always_disordered_ele_up = []
list_always_ordered_ele_up = []
for i in range(np.shape(order_time_evo_up)[1]):
    if all(order_time_evo_up[:, i] == 1):
        list_always_ordered_ele_up.append(i)
    if all(order_time_evo_up[:, i] == 0):
        list_always_disordered_ele_up.append(i)

list_always_disordered_ele_down = []
list_always_ordered_ele_down = []
for i in range(np.shape(order_time_evo_down)[1]):
    if all(order_time_evo_down[:, i] == 1):
        list_always_ordered_ele_down.append(i)
    if all(order_time_evo_down[:, i] == 0):
        list_always_disordered_ele_down.append(i)
        

common_mols_disordered_up = common_mols_upper[0][list_always_disordered_ele_up] #list of resid always in the fluid phase up
common_mols_ordered_up = common_mols_upper[0][list_always_ordered_ele_up] #list of resid always in the ordered phase up

common_mols_disordered_down = common_mols_lower[0][list_always_disordered_ele_down] #list of resid always in the fluid phase up
common_mols_ordered_down = common_mols_lower[0][list_always_ordered_ele_down] #list of resid always in the ordered phase up

common_mols_disordered = np.vstack((common_mols_disordered_up, common_mols_disordered_down))
common_mols_ordered = np.vstack((common_mols_ordered_up, common_mols_ordered_down))



#store disordered coordinates
#all_xs_dis = []
#all_ys_dis = []

all_xs_DLIP_dis_up= []
all_ys_DLIP_dis_up =[]

all_xs_CHL_dis_up = []
all_ys_CHL_dis_up =[]

all_xs_DSPC_dis_up = []
all_ys_DSPC_dis_up =[]


all_xs_DLIP_dis_down= []
all_ys_DLIP_dis_down =[]

all_xs_CHL_dis_down = []
all_ys_CHL_dis_down =[]

all_xs_DSPC_dis_down = []
all_ys_DSPC_dis_down =[]
#ordered coordinates
#all_xs_ord = []
#all_ys_ord = []

all_xs_DLIP_ord_up = []
all_ys_DLIP_ord_up =[]

all_xs_CHL_ord_up = []
all_ys_CHL_ord_up = []

all_xs_DSPC_ord_up = []
all_ys_DSPC_ord_up =[]

all_xs_DLIP_ord_down = []
all_ys_DLIP_ord_down =[]

all_xs_CHL_ord_down = []
all_ys_CHL_ord_down = []

all_xs_DSPC_ord_down = []
all_ys_DSPC_ord_down =[]


#center of mass motion of the upper and lower layers
CM_upper_x = []
CM_upper_y = []
CM_lower_x = []
CM_lower_y = []




upper_CM_str = ' '.join(str(int(x)) for x in common_elements_upper[:])
lower_CM_str = ' '.join(str(int(x)) for x in common_elements_lower[:])
for ts in u.trajectory: # every 1 ns

#center of mass of the disordered phase
    #disordered_CM_str = ' '.join(str(int(x)) for x in common_mols_disordered[:,0])
    #disordered_CM_x.append(u.select_atoms('resnum' + ' ' + disordered_CM_str).center_of_mass()[0])
    #disordered_CM_y.append(u.select_atoms('resnum' + ' ' + disordered_CM_str).center_of_mass()[1])
#center of mass of the ordered phase
    #ordered_CM_str = ' '.join(str(int(x)) for x in common_mols_ordered[:,0])
    #ordered_CM_x.append(u.select_atoms('resnum' + ' ' + ordered_CM_str).center_of_mass()[0])
    #ordered_CM_y.append(u.select_atoms('resnum' + ' ' + ordered_CM_str).center_of_mass()[1])
       
    CM_upper_x.append(u.select_atoms('resnum' + ' ' + upper_CM_str).center_of_mass()[0])
    CM_upper_y.append(u.select_atoms('resnum' + ' ' + upper_CM_str).center_of_mass()[1])
    CM_lower_x.append(u.select_atoms('resnum' + ' ' + lower_CM_str).center_of_mass()[0])
    CM_lower_y.append(u.select_atoms('resnum' + ' ' + lower_CM_str).center_of_mass()[1])
    
# =============================================================================
#     plt.plot(CM_upper_x, CM_upper_y)
#     plt.plot(CM_lower_x, CM_lower_y)
#     plt.plot(CM_upper_x)
#     plt.plot(CM_lower_x)  
#     plt.plot(CM_upper_y)
#     plt.plot(CM_lower_y) 
#     plt.axis("equal")
# =============================================================================
    
#CM of all the single residues in the disordered phase 
#    coords_disordered = []
    coords_dis_DLIP_up = []
    coords_dis_CHL_up  = []
    coords_dis_DSPC_up = []

# resname_disordered = []
    for j in np.arange(len(common_mols_disordered_up)):
        
        #residues = u.select_atoms('resnum %i'%(common_mols_disordered[j,0]))
        dis_DLIP_up = u.select_atoms('resname DLIP and resnum %i'%(common_mols_disordered_up[j,0]))
        
        #dis_CHL_up = u.select_atoms('resname CHL and resnum %i'%(common_mols_disordered_up[j,0]))
        #dis_DSPC_up = u.select_atoms('resname DSPC and resnum %i'%(common_mols_disordered_up[j,0]))
        #coords_disordered.append(list(residues.center_of_mass()))
        if dis_DLIP_up:
            coords_dis_DLIP_up.append(list(dis_DLIP_up.center_of_mass()))

        #if dis_CHL_up:
        #    coords_dis_CHL_up.append(list(dis_CHL_up.center_of_mass()))    

        #if dis_DSPC_up:
        #    coords_dis_DSPC_up.append(list(dis_DSPC_up.center_of_mass())) 
            
            
    #coords_disordered = np.array(coords_disordered)
    coords_dis_DLIP_up = np.array(coords_dis_DLIP_up)
    #coords_dis_CHL_up = np.array(coords_dis_CHL_up)
    #coords_dis_DSPC_up = np.array(coords_dis_DSPC_up)    
    
    #x_coords = coords_disordered[:, 0]
    #y_coords = coords_disordered[:, 1]
    
    x_DLIP_up  = coords_dis_DLIP_up[:, 0]
    y_DLIP_up  = coords_dis_DLIP_up[:, 1]
    
    
    #x_CHL_up  = coords_dis_CHL_up[:, 0]
    #y_CHL_up  = coords_dis_CHL_up[:, 1]
    
    #x_DSPC_up  = coords_dis_DSPC_up[:, 0]
    #y_DSPC_up  = coords_dis_DSPC_up[:, 1]
   
   # all_xs_dis.append(x_coords)
   # all_ys_dis.append(y_coords)
    
    all_xs_DLIP_dis_up.append(x_DLIP_up)
    all_ys_DLIP_dis_up.append(y_DLIP_up)

    #all_xs_CHL_dis_up.append(x_CHL_up)
    #all_ys_CHL_dis_up.append(y_CHL_up)

    #all_xs_DSPC_dis_up.append(x_DSPC_up)
    #all_ys_DSPC_dis_up.append(y_DSPC_up)    



    coords_dis_DLIP_down = []
    #coords_dis_CHL_down  = []
    #coords_dis_DSPC_down = []


    for k in np.arange(len(common_mols_disordered_down)):
        
        #residues = u.select_atoms('resnum %i'%(common_mols_disordered[j,0]))
        dis_DLIP_down = u.select_atoms('resname DLIP and resnum %i'%(common_mols_disordered_down[k,0]))
        #dis_CHL_down = u.select_atoms('resname CHL and resnum %i'%(common_mols_disordered_down[k,0]))
        #dis_DSPC_down = u.select_atoms('resname DSPC and resnum %i'%(common_mols_disordered_down[k,0]))
        #coords_disordered.append(list(residues.center_of_mass()))
        if dis_DLIP_down:
            coords_dis_DLIP_down.append(list(dis_DLIP_down.center_of_mass()))

        #if dis_CHL_down:
        #    coords_dis_CHL_down.append(list(dis_CHL_down.center_of_mass()))    

        #if dis_DSPC_down:
         #   coords_dis_DSPC_down.append(list(dis_DSPC_down.center_of_mass())) 
            
            
    #coords_disordered = np.array(coords_disordered)
    coords_dis_DLIP_down = np.array(coords_dis_DLIP_down)
    #coords_dis_CHL_down = np.array(coords_dis_CHL_down)
    #coords_dis_DSPC_down = np.array(coords_dis_DSPC_down)    
    
    #x_coords = coords_disordered[:, 0]
    #y_coords = coords_disordered[:, 1]
    
    x_DLIP_down  = coords_dis_DLIP_down[:, 0]
    y_DLIP_down  = coords_dis_DLIP_down[:, 1]
    
    #x_CHL_down  = coords_dis_CHL_down[:, 0]
    #y_CHL_down  = coords_dis_CHL_down[:, 1]
    
    #x_DSPC_down  = coords_dis_DSPC_down[:, 0]
    #y_DSPC_down  = coords_dis_DSPC_down[:, 1]
   
   # all_xs_dis.append(x_coords)
   # all_ys_dis.append(y_coords)
    
    all_xs_DLIP_dis_down.append(x_DLIP_down)
    all_ys_DLIP_dis_down.append(y_DLIP_down)

   # all_xs_CHL_dis_down.append(x_CHL_down)
   # all_ys_CHL_dis_down.append(y_CHL_down)

   # all_xs_DSPC_dis_down.append(x_DSPC_down)
   # all_ys_DSPC_dis_down.append(y_DSPC_down)    


    coords_ord_DLIP_up = []
    coords_ord_CHL_up  = []
    coords_ord_DSPC_up = []

    for l in np.arange(len(common_mols_ordered_up)):
        
        #residues = u.select_atoms('resnum %i'%(common_mols_disordered[j,0]))
       # ord_DLIP_up = u.select_atoms('resname DLIP and resnum %i'%(common_mols_ordered_up[l,0]))
        ord_CHL_up = u.select_atoms('resname CHL and resnum %i'%(common_mols_ordered_up[l,0]))
        ord_DSPC_up = u.select_atoms('resname DSPC and resnum %i'%(common_mols_ordered_up[l,0]))
        #coords_disordered.append(list(residues.center_of_mass()))
        #if ord_DLIP_up:
        #    coords_ord_DLIP_up.append(list(ord_DLIP_up.center_of_mass()))

        if ord_CHL_up:
            coords_ord_CHL_up.append(list(ord_CHL_up.center_of_mass()))    

        if ord_DSPC_up:
            coords_ord_DSPC_up.append(list(ord_DSPC_up.center_of_mass())) 
            
            
    #coords_disordered = np.array(coords_disordered)
   # coords_ord_DLIP_up = np.array(coords_ord_DLIP_up)
    coords_ord_CHL_up = np.array(coords_ord_CHL_up)
    coords_ord_DSPC_up = np.array(coords_ord_DSPC_up)    
    
    #x_coords = coords_disordered[:, 0]
    #y_coords = coords_disordered[:, 1]
    
    #x_DLIP_up  = coords_ord_DLIP_up[:, 0]
    #y_DLIP_up  = coords_ord_DLIP_up[:, 1]
    
    x_CHL_up  = coords_ord_CHL_up[:, 0]
    y_CHL_up  = coords_ord_CHL_up[:, 1]
    
    x_DSPC_up  = coords_ord_DSPC_up[:, 0]
    y_DSPC_up  = coords_ord_DSPC_up[:, 1]
   
   # all_xs_dis.append(x_coords)
   # all_ys_dis.append(y_coords)
    
    #all_xs_DLIP_ord_up.append(x_DLIP_up)
    #all_ys_DLIP_ord_up.append(y_DLIP_up)

    all_xs_CHL_ord_up.append(x_CHL_up)
    all_ys_CHL_ord_up.append(y_CHL_up)

    all_xs_DSPC_ord_up.append(x_DSPC_up)
    all_ys_DSPC_ord_up.append(y_DSPC_up)    



    coords_ord_DLIP_down = []
    coords_ord_CHL_down  = []
    coords_ord_DSPC_down = []


    for m in np.arange(len(common_mols_ordered_down)):
        
        #residues = u.select_atoms('resnum %i'%(common_mols_disordered[j,0]))
        #ord_DLIP_down = u.select_atoms('resname DLIP and resnum %i'%(common_mols_ordered_down[m,0]))
        ord_CHL_down = u.select_atoms('resname CHL and resnum %i'%(common_mols_ordered_down[m,0]))
        ord_DSPC_down = u.select_atoms('resname DSPC and resnum %i'%(common_mols_ordered_down[m,0]))
        #coords_disordered.append(list(residues.center_of_mass()))
        #if ord_DLIP_down:
        #    coords_ord_DLIP_down.append(list(ord_DLIP_down.center_of_mass()))

        if ord_CHL_down:
            coords_ord_CHL_down.append(list(ord_CHL_down.center_of_mass()))    

        if ord_DSPC_down:
            coords_ord_DSPC_down.append(list(ord_DSPC_down.center_of_mass())) 
            
            
    #coords_disordered = np.array(coords_disordered)
    #coords_ord_DLIP_down = np.array(coords_ord_DLIP_down)
    coords_ord_CHL_down = np.array(coords_ord_CHL_down)
    coords_ord_DSPC_down = np.array(coords_ord_DSPC_down)    
    
    #x_coords = coords_disordered[:, 0]
    #y_coords = coords_disordered[:, 1]
    
  #  x_DLIP_down  = coords_ord_DLIP_down[:, 0]
  #  y_DLIP_down  = coords_ord_DLIP_down[:, 1]
    
    x_CHL_down  = coords_ord_CHL_down[:, 0]
    y_CHL_down  = coords_ord_CHL_down[:, 1]
    
    x_DSPC_down  = coords_ord_DSPC_down[:, 0]
    y_DSPC_down  = coords_ord_DSPC_down[:, 1]
   
   # all_xs_dis.append(x_coords)
   # all_ys_dis.append(y_coords)
    
  #  all_xs_DLIP_ord_down.append(x_DLIP_down)
  #  all_ys_DLIP_ord_down.append(y_DLIP_down)

    all_xs_CHL_ord_down.append(x_CHL_down)
    all_ys_CHL_ord_down.append(y_CHL_down)

    all_xs_DSPC_ord_down.append(x_DSPC_down)
    all_ys_DSPC_ord_down.append(y_DSPC_down)    




#CM of the single resid in the ordered phase 
#for ts in u.trajectory :
    
    
    #all_xs_ord.append(x_coords)
    #all_ys_ord.append(y_coords)
    

#return the vectors  
#CM dis
    
CM_upper_x = np.array(CM_upper_x)   
CM_upper_y = np.array(CM_upper_y) 
CM_lower_x = np.array(CM_lower_x) 
CM_lower_y = np.array(CM_lower_y) 
#disordered_CM_x = np.array(disordered_CM_x)
#disordered_CM_y = np.array(disordered_CM_y)
#TODO 
#np.save (output_dir + 'disordered_CM_x.npy', disordered_CM_x)
#np.save (output_dir + 'disordered_CM_y.npy', disordered_CM_y)

#CM ord
#ordered_CM_x =  np.array(ordered_CM_x)
#ordered_CM_y =  np.array(ordered_CM_y)
#np.save (output_dir + 'ordered_CM_x.npy', ordered_CM_x)
#np.save (output_dir + 'ordered_CM_y.npy', ordered_CM_y)


#lipids dis
#all_xs_dis = np.array(all_xs_dis)
#all_ys_dis = np.array(all_ys_dis)
all_xs_DLIP_dis_up = np.array(all_xs_DLIP_dis_up)
all_ys_DLIP_dis_up = np.array(all_ys_DLIP_dis_up)
#all_xs_CHL_dis_up = np.array(all_xs_CHL_dis_up)
#all_ys_CHL_dis_up = np.array(all_ys_CHL_dis_up)
#all_xs_DSPC_dis_up = np.array(all_xs_DSPC_dis_up)
#all_ys_DSPC_dis_up = np.array(all_ys_DSPC_dis_up) 


all_xs_DLIP_dis_down = np.array(all_xs_DLIP_dis_down)
all_ys_DLIP_dis_down = np.array(all_ys_DLIP_dis_down)
#all_xs_CHL_dis_down = np.array(all_xs_CHL_dis_down)
#all_ys_CHL_dis_down = np.array(all_ys_CHL_dis_down)
#all_xs_DSPC_dis_down = np.array(all_xs_DSPC_dis_down)
#all_ys_DSPC_dis_down = np.array(all_ys_DSPC_dis_down)

# =============================================================================
# np.save (output_dir + 'all_xs_dis.npy', all_xs_dis)
# np.save (output_dir + 'all_ys_dis.npy', all_ys_dis)
# np.save (output_dir + 'all_xs_DLIP_dis.npy', all_xs_DLIP_dis)
# np.save (output_dir + 'all_ys_DLIP_dis.npy', all_ys_DLIP_dis)
# np.save (output_dir + 'all_xs_CHL_dis.npy', all_xs_CHL_dis)
# np.save (output_dir + 'all_ys_CHL_dis.npy', all_ys_CHL_dis)
# np.save (output_dir + 'all_xs_DSPC_dis.npy', all_xs_DSPC_dis)
# np.save (output_dir + 'all_ys_DSPC_dis.npy', all_ys_DSPC_dis)
# 
# =============================================================================
#lipids ord   
#all_xs_ord = np.array(all_xs_ord)
#all_ys_ord = np.array(all_ys_ord)
#all_xs_DLIP_ord = np.array(all_xs_DLIP_ord)
#all_ys_DLIP_ord = np.array(all_ys_DLIP_ord)
all_xs_CHL_ord_up = np.array(all_xs_CHL_ord_up)
all_ys_CHL_ord_up = np.array(all_ys_CHL_ord_up)
all_xs_DSPC_ord_up = np.array(all_xs_DSPC_ord_up)
all_ys_DSPC_ord_up = np.array(all_ys_DSPC_ord_up)

all_xs_CHL_ord_down = np.array(all_xs_CHL_ord_down)
all_ys_CHL_ord_down = np.array(all_ys_CHL_ord_down)
all_xs_DSPC_ord_down = np.array(all_xs_DSPC_ord_down)
all_ys_DSPC_ord_down = np.array(all_ys_DSPC_ord_down)

# =============================================================================
# np.save (output_dir + 'all_xs_ord.npy', all_xs_ord)
# np.save (output_dir + 'all_ys_ord.npy', all_ys_ord)
# #np.save (output_dir + 'all_xs_DLIP_ord.npy', all_xs_DLIP_ord)
# #np.save (output_dir + 'all_ys_DLIP_ord.npy', all_ys_DLIP_ord)
# np.save (output_dir + 'all_xs_CHL_ord.npy', all_xs_CHL_ord)
# np.save (output_dir + 'all_ys_CHL_ord.npy', all_ys_CHL_ord)
# np.save (output_dir + 'all_xs_DSPC_ord.npy', all_xs_DSPC_ord)
# np.save (output_dir + 'all_ys_DSPC_ord.npy', all_ys_DSPC_ord)
# 
# =============================================================================


# =============================================================================
# for i in range(all_xs_DSPC_ord.shape[1]):
#     plt.plot(all_xs_DSPC_ord[:, i]- CM_upper_x[:], all_ys_DSPC_ord[:, i]- CM_upper_y[:])
# #plt.plot(all_xs_DSPC_ord[:, 10]- CM_upper_x[:], all_ys_DSPC_ord[:, 10]- CM_upper_y[:])
#     plt.axis("equal")
#     
#     
# for i in range(all_xs_CHL_ord.shape[1]):
#     plt.plot(all_xs_CHL_ord[:, i]- CM_upper_x[:], all_ys_CHL_ord[:, i]- CM_upper_y[:])
# #plt.plot(all_xs_DSPC_ord[:, 10]- CM_upper_x[:], all_ys_DSPC_ord[:, 10]- CM_upper_y[:])
#     plt.axis("equal")    
# 
# for i in range(all_xs_DLIP_dis.shape[1]):
#     plt.plot(all_xs_DLIP_dis[:, i]- CM_upper_x[:], all_ys_DLIP_dis[:, i]- CM_upper_y[:])
# #plt.plot(all_xs_DSPC_ord[:, 10]- CM_upper_x[:], all_ys_DSPC_ord[:, 10]- CM_upper_y[:])
#     plt.axis("equal") 
# =============================================================================

#sliceof the array every delay  
def calcMSD_Cstyle(rx, ry, rx_cm, ry_cm):
    N_FR = rx.shape[0]
    N_PA = rx.shape[1]

    sdx = np.zeros(N_FR)
    sdy = np.zeros(N_FR)
    cnt = np.zeros(N_FR)
    for t in range(N_FR):
        for dt in range(1, (N_FR - t)):
            cnt[dt] = cnt[dt] + 1
            for n in range(N_PA):
                sdx[dt] = sdx[dt] + ((rx[t+dt, n] - rx_cm[t+dt]) - (rx[t, n] - rx_cm[t]))**2
                sdy[dt] = sdy[dt] + ((ry[t+dt, n] - ry_cm[t+dt]) - (ry[t, n] - ry_cm[t]))**2
    for t in range(N_FR):
        sdx[t] = sdx[t] / ((N_PA * cnt[t]) if int(cnt[t]) else 1)
        sdy[t] = sdy[t] / ((N_PA * cnt[t]) if int(cnt[t]) else 1)
    return sdx, sdy 




def calcMSD_C_single_molecule(rx, ry, rx_cm, ry_cm):
    N_FR = rx.shape[0]
    N_PA = rx.shape[1]
    a = np.zeros((N_FR, N_PA))
    b = np.zeros((N_FR, N_PA))
    cnt = np.zeros(N_FR)
    for t in range(N_FR):
        #print ("T:", t)
        for dt in range(1, (N_FR - t)):
            cnt[dt] = cnt[dt] + 1
            for n in range(N_PA):
                a[dt,n] = a[dt,n] + ((rx[t+dt, n] - rx_cm[t+dt]) - (rx[t, n] - rx_cm[t]))**2
                b[dt,n] = b[dt,n] + ((ry[t+dt, n] - ry_cm[t+dt]) - (ry[t, n] - ry_cm[t]))**2
    for t in range(N_FR):
        a[t] = a[t] / (cnt[t] if int(cnt[t]) else 1)
        b[t] = b[t] / (cnt[t] if int(cnt[t]) else 1)
    return a+b





def calcCorr_C_single_molecule(rx, ry, rx_cm, ry_cm, dt):    
    N_FR =rx.shape[0]  #rx.shape[0]
    N_PA = rx.shape[1]
    a = np.zeros((N_FR, N_PA))
    b = np.zeros((N_FR, N_PA))
    for n in range(N_PA):
        for  t in range(N_FR-dt):
            a[t,n] = a[t,n]+ ((rx[t + dt,n] - rx_cm[t+ dt]) - (rx[t,n] - rx_cm[t]))* ((rx[dt,n] - rx_cm[dt]) - (rx[0,n] - rx_cm[0]))#x(0) t+dt
            b[t,n] = a[t,n]+ ((ry[t + dt,n] - ry_cm[t+ dt]) - (ry[t,n] - ry_cm[t]))* ((ry[dt,n] - ry_cm[dt]) - (ry[0,n] - ry_cm[0]))#x(0) t+dt

    return a+b 


import scipy.optimize as opt
 
def func(x, a, b):
    return 4* a * np.power(x, b)


def FLE_theory(t, dt, alpha):
    y = ((t+dt)** alpha - 2*t**alpha + np.abs(t - dt)**alpha )/2*dt**alpha
    return y


def CTRW_theory(t, dt, alpha):
    y = np.zeros((t.shape[0]))
    for i in range (len(t)):
        if t[i] <= dt:
            y[i] =  1 - (t[i]/dt)**alpha
        else:
            y[i] = 0
    return y

#do this for a trajectory of 2 ns every 10 ps and a delta t of 200
dt = 100
DSPC_ord_up = calcCorr_C_single_molecule(all_xs_DSPC_ord_up, all_ys_DSPC_ord_up, CM_upper_x, CM_upper_y, dt)
DSPC_ord_down = calcCorr_C_single_molecule(all_xs_DSPC_ord_down, all_ys_DSPC_ord_down, CM_lower_x, CM_lower_y, dt)

DSPC_ord = np.column_stack((DSPC_ord_up, DSPC_ord_down))
av_DSPC_ord = np.mean(DSPC_ord, axis=1)/(dt**2)


plt.plot(av_DSPC_ord/av_DSPC_ord[0], '.')

MSD_DSPC_ord_up =  calcMSD_C_single_molecule(all_xs_DSPC_ord_up, all_ys_DSPC_ord_up, CM_upper_x, CM_upper_y)
MSD_DSPC_ord_down =  calcMSD_C_single_molecule(all_xs_DSPC_ord_down, all_ys_DSPC_ord_down, CM_lower_x, CM_lower_y)
MSD_DSPC_ord =  np.column_stack((MSD_DSPC_ord_up, MSD_DSPC_ord_down))

av_MSD_DSPCs_ord = np.mean(MSD_DSPC_ord, axis=1)

xdata = np.linspace(0, u.trajectory.n_frames-1, u.trajectory.n_frames)

plt.loglog(xdata[1:u.trajectory.n_frames] ,av_MSD_DSPCs_ord[1:u.trajectory.n_frames], c = 'blue', lw =2)
fitted_DSPC_ord_10ns, pcov_DsPC_ord_10ns = opt.curve_fit(func, xdata[2:10] , av_MSD_DSPCs_ord[2:10])
plt.loglog(xdata[2:10], func(xdata[2:10], *fitted_DSPC_ord_10ns), label="fit", lw=4)

DSPC_ord_CTRW_theory = CTRW_theory(xdata, dt, fitted_DSPC_ord_10ns[1])
DSPC_ord_FLE_theory = FLE_theory(xdata, dt, fitted_DSPC_ord_10ns[1])
plt.plot(DSPC_ord_FLE_theory/ np.max(DSPC_ord_FLE_theory))
plt.plot(DSPC_ord_CTRW_theory/ np.max(DSPC_ord_CTRW_theory), )
plt.plot(av_DSPC_ord/av_DSPC_ord[0])
plt.xlim(-10,800)



dt = 200
CHL_ord_up = calcCorr_C_single_molecule(all_xs_CHL_ord_up, all_ys_CHL_ord_up, CM_upper_x, CM_upper_y, dt)
CHL_ord_down = calcCorr_C_single_molecule(all_xs_CHL_ord_down, all_ys_CHL_ord_down, CM_lower_x, CM_lower_y, dt)

CHL_ord = np.column_stack((CHL_ord_up, CHL_ord_down))
av_CHL_ord = np.mean(CHL_ord, axis=1)/(dt**2)

plt.plot(av_CHL_ord/av_CHL_ord[0], '.')

MSD_CHL_ord_up =  calcMSD_C_single_molecule(all_xs_CHL_ord_up, all_ys_CHL_ord_up, CM_upper_x, CM_upper_y)
MSD_CHL_ord_down =  calcMSD_C_single_molecule(all_xs_CHL_ord_down, all_ys_CHL_ord_down, CM_lower_x, CM_lower_y)
MSD_CHL_ord =  np.column_stack((MSD_CHL_ord_up, MSD_CHL_ord_down))
av_MSD_CHLs_ord = np.mean(MSD_CHL_ord, axis=1)

xdata = np.linspace(0, u.trajectory.n_frames-1, u.trajectory.n_frames)

plt.loglog(xdata[1:u.trajectory.n_frames] ,av_MSD_CHLs_ord[1:u.trajectory.n_frames], c = 'blue', lw =2)
fitted_CHL_ord_10ns, pcov_CHL_ord_10ns = opt.curve_fit(func, xdata[2:10] , av_MSD_CHLs_ord[2:10])
plt.loglog(xdata[2:10], func(xdata[2:10], *fitted_CHL_ord_10ns), label="fit", lw=4)

CHL_ord_CTRW_theory = CTRW_theory(xdata, dt, fitted_CHL_ord_10ns[1])
CHL_ord_FLE_theory = FLE_theory(xdata, dt, fitted_CHL_ord_10ns[1])
plt.plot(CHL_ord_FLE_theory/ np.max(CHL_ord_FLE_theory), c='black')
plt.plot(CHL_ord_CTRW_theory/ np.max(CHL_ord_CTRW_theory), c='black')
plt.plot(xdata[:], av_CHL_ord[:]/av_CHL_ord[0], '.', markersize=2, c= 'green')
plt.xlim(-10,800)



dt = 100
DLIP_dis_up = calcCorr_C_single_molecule(all_xs_DLIP_dis_up, all_ys_DLIP_dis_up, CM_upper_x, CM_upper_y, dt)
DLIP_dis_down = calcCorr_C_single_molecule(all_xs_DLIP_dis_down, all_ys_DLIP_dis_down, CM_lower_x, CM_lower_y, dt)

DLIP_dis = np.column_stack((DLIP_dis_up, DLIP_dis_down))
av_DLIP_dis = np.mean(DLIP_dis, axis=1)/(dt**2)

plt.plot(av_DLIP_dis/av_DLIP_dis[0], '.')

MSD_DLIP_dis_up =  calcMSD_C_single_molecule(all_xs_DLIP_dis_up, all_ys_DLIP_dis_up, CM_upper_x, CM_upper_y)
MSD_DLIP_dis_down =  calcMSD_C_single_molecule(all_xs_DLIP_dis_down, all_ys_DLIP_dis_down, CM_lower_x, CM_lower_y)
MSD_DLIP_dis =  np.column_stack((MSD_DLIP_dis_up, MSD_DLIP_dis_down))
av_MSD_DLIPs_dis = np.mean(MSD_DLIP_dis, axis=1)

xdata = np.linspace(0, u.trajectory.n_frames -1, u.trajectory.n_frames)

plt.loglog(xdata[1:u.trajectory.n_frames] ,av_MSD_DLIPs_dis[1:u.trajectory.n_frames], c = 'blue', lw =2)
fitted_DLIP_dis_10ns, pcov_DLIP_dis_10ns = opt.curve_fit(func, xdata[2:40] , av_MSD_DLIPs_dis[2:40])
plt.loglog(xdata[2:40], func(xdata[2:40], *fitted_DLIP_dis_10ns), label="fit", lw=4)

DLIP_dis_FLE_theory = FLE_theory(xdata, dt, fitted_DLIP_dis_10ns[1])
DLIP_dis_CTRW_theory = CTRW_theory(xdata , dt, fitted_DLIP_dis_10ns[1])

plt.plot(DLIP_dis_FLE_theory/ np.max(DLIP_dis_FLE_theory), c='black')
plt.plot(DLIP_dis_CTRW_theory/ np.max(DLIP_dis_CTRW_theory), c='black')
plt.plot(xdata[:], av_DLIP_dis[:]/av_DLIP_dis[0], '.', markersize=2)

plt.xlim(-10,800)




# =============================================================================
# MSD_CHL_ord_up =  calcMSD_C_single_molecule(all_xs_CHL_ord_up, all_ys_CHL_ord_up, CM_upper_x, CM_upper_y)
# av_MSD_CHLs_ord_up = np.mean(MSD_CHL_ord_up, axis=1)
# 
# MSD_CHL_ord_down =  calcMSD_C_single_molecule(all_xs_CHL_ord_down, all_ys_CHL_ord_down, CM_lower_x, CM_lower_y)
# av_MSD_CHLs_ord_down = np.mean(MSD_CHL_ord_down, axis=1)
# 
# 
# MSD_DLIP_dis_up =  calcMSD_C_single_molecule(all_xs_DLIP_dis_up, all_ys_DLIP_dis_up, CM_upper_x, CM_upper_y)
# MSD_DLIP_dis_down =  calcMSD_C_single_molecule(all_xs_DLIP_dis_down, all_ys_DLIP_dis_down, CM_lower_x, CM_lower_y)
# 
# av_MSD_DLIPs_dis_up= np.mean(MSD_DLIP_dis_up, axis=1)
# av_MSD_DLIPs_dis_down= np.mean(MSD_DLIP_dis_down, axis=1)
# 
# 
# # =============================================================================
# plt.loglog(MSD_DSPC_ord_up[:])
# plt.loglog(MSD_DSPC_ord_down[:])
# plt.loglog(av_MSD_DSPCs_ord_up, c = 'black', lw =4)
# plt.loglog(av_MSD_DSPCs_ord_down, c = 'red', lw =4)
# 
# MSD_DSPCs_ord = np.column_stack((MSD_DSPC_ord_up, MSD_DSPC_ord_down))
# av_MSD_DSPCs_ord = np.mean(MSD_DSPCs_ord, axis=1)
# # plt.loglog(MSD_CHL_ord[:])
# #
# #==============================================================================
# plt.loglog(MSD_CHL_ord_up[:])
# plt.loglog(MSD_CHL_ord_down[:])
# plt.plot(av_MSD_CHLs_ord_up, c = 'black', lw =4)
# plt.plot(av_MSD_CHLs_ord_down, c = 'red', lw =4)
# 
# MSD_CHLs_ord = np.column_stack((MSD_CHL_ord_up, MSD_CHL_ord_down))
# av_MSD_CHLs_ord = np.mean(MSD_CHLs_ord, axis=1)
# 
# #==============================================================================
# plt.loglog(MSD_DLIP_dis_up[:3200])
# plt.loglog(MSD_DLIP_dis_down[:3200])
# 
# MSD_DLIPs_dis = np.column_stack((MSD_DLIP_dis_up, MSD_DLIP_dis_down))
# av_MSD_DLIPs_dis = np.mean(MSD_DLIPs_dis, axis=1)
# plt.plot(av_MSD_DLIPs_dis[:3200], c = 'blue', lw =4)
# 
# plt.plot(av_MSD_DLIPs_dis_up[:3200], c = 'black', lw =4)
# plt.plot(av_MSD_DLIPs_dis_down[:3200], c = 'red', lw =4)
# 
# #==============================================================================
# #  Fitting 
# #==============================================================================
# import scipy.optimize as opt
#  
# def func(x, a, b):
#     return 4* a * np.power(x, b)
# 
# xdata = np.linspace(0, 4000, 4001)
# 
# plt.loglog(xdata[1:2000] ,av_MSD_DLIPs_dis[1:2000], c = 'blue', lw =2)
# fitted_DLIP_dis_10ns, pcov_DLIP_dis_10ns = opt.curve_fit(func, xdata[2:10] , av_MSD_DLIPs_dis[2:10])
# plt.loglog(xdata[2:10], func(xdata[2:10], *fitted_DLIP_dis_10ns), label="fit", lw=4)
# 
# fitted_DLIP_dis_1000ns, pcov_DLIP_dis_1000ns = opt.curve_fit(func, xdata[100:500] , av_MSD_DLIPs_dis[100:500])
# plt.loglog(xdata[100:500], func(xdata[100:500], *fitted_DLIP_dis_1000ns), label="fit", lw=4)
# 
# 
# plt.plot(xdata[1:2000], av_MSD_DSPCs_ord[1:2000], c = 'blue', lw =1)
# fitted_DSPC_ord_10ns, pcov_DSPC_ord_10ns = opt.curve_fit(func, xdata[2:10] , av_MSD_DSPCs_ord[2:10])
# plt.loglog(xdata[2:10], func(xdata[2:10], *fitted_DSPC_ord_10ns), label="fit",lw=4)
# 
# fitted_DSPC_ord_1000ns, pcov_DSPC_ord_1000ns = opt.curve_fit(func, xdata[100:500] , av_MSD_DSPCs_ord[100:500])
# plt.loglog(xdata[100:500], func(xdata[100:500], *fitted_DSPC_ord_1000ns), label="fit", lw=4)
# 
# 
# plt.plot(xdata[1:2000], av_MSD_CHLs_ord[1:2000], c = 'blue', lw =1)
# fitted_CHL_ord_10ns, pcov_CHL_ord_10ns = opt.curve_fit(func, xdata[2:10] , av_MSD_CHLs_ord[2:10])
# plt.loglog(xdata[2:10], func(xdata[2:10], *fitted_CHL_ord_10ns), label="fit",lw=4)
# 
# fitted_CHL_ord_1000ns, pcov_CHL_ord_1000ns = opt.curve_fit(func, xdata[50:500] , av_MSD_CHLs_ord[50:500])
# plt.loglog(xdata[100:500], func(xdata[100:500], *fitted_CHL_ord_1000ns), label="fit", lw=4)
# 
# plt.xlim(1,2000)
# plt.grid(True, which='minor')
# =============================================================================


####This i9n case you want to do a error analysis
# now plot the best fit curve and also +- 3 sigma curves
# the square root of the diagonal covariance matrix element 
# is the uncertianty on the corresponding fit parameter.


#sigma = [fitCovariances[0,0], fitCovariances[1,1], fitCovariances[2,2] ]
#plt.plot(t, fitFunc(t, fitParams[0], fitParams[1], fitParams[2]),\
#         t, fitFunc(t, fitParams[0] + sigma[0], fitParams[1] - sigma[1], fitParams[2] + sigma[2]),\
#         t, fitFunc(t, fitParams[0] - sigma[0], fitParams[1] + sigma[1], fitParams[2] - sigma[2])\
#         )


#plt.plot(4* fitted_DLIP_dis[0] * np.power(xdata[20:70], fitted_DLIP_dis[1]) +100)
# 
# 
# import scipy.optimize as opt
# 
# def func(x, a, b):
#      return 4* a * np.power(x, b)
# 
# # The actual curve fitting happens here
# xdata = np.linspace(0, 400, 401)
# optimizedParameters, pcov = opt.curve_fit(func, xdata[2:5] , av_MSD_DSPCs_ord[2:5])
# 
# plt.loglog(xdata, av_MSD_DSPCs_ord)
# # Use the optimized parameters to plot the best fit
# plt.loglog(xdata[2:10], func(xdata[2:10], *optimizedParameters), label="fit");
# 
# # Show the graph
# plt.legend()
# #plt.plot(MSD_ord)
# plt.loglog(av_MSD_DSPCs_ord, c='red')
# #plt.xlim(0.9,500)
# 
# 
# optimizedParameters_later, pcov_later = opt.curve_fit(func, xdata[20:70] , av_MSD_DSPCs_ord[20:70])
# plt.loglog(xdata[20:70], func(xdata[20:70], *optimizedParameters_later), label="fit");
# 
# optimizedParameters_later2, pcov_later = opt.curve_fit(func, xdata[200:300] , av_MSD_DSPCs_ord[200:300])
# plt.loglog(xdata[200:300], func(xdata[200:300], *optimizedParameters_later2), label="fit");
# plt.plot(MSD_DSPC_ord)
# 
# 
# #retrieve the MSD for the different lipids 
# MSD_x_dis, MSD_y_dis = calcMSD_Cstyle(all_xs_dis, all_ys_dis, disordered_CM_x, disordered_CM_y)
# MSD_DLIP_x_dis, MSD_DLIP_y_dis = calcMSD_Cstyle(all_xs_DLIP_dis, all_ys_DLIP_dis, disordered_CM_x, disordered_CM_y)
# MSD_CHL_x_dis, MSD_CHL_y_dis = calcMSD_Cstyle(all_xs_CHL_dis, all_ys_CHL_dis, disordered_CM_x, disordered_CM_y)
# MSD_DSPC_x_dis, MSD_DSPC_y_dis = calcMSD_Cstyle(all_xs_DSPC_dis, all_ys_DSPC_dis, disordered_CM_x, disordered_CM_y)
# 
# MSD_x_ord, MSD_y_ord = calcMSD_Cstyle(all_xs_ord, all_ys_ord, ordered_CM_x, ordered_CM_y)
# #MSD_DLIP_x_ord, MSD_DLIP_y_ord = calcMSD_Cstyle(all_xs_DLIP_ord, all_ys_DLIP_ord, ordered_CM_x, ordered_CM_y)
# MSD_CHL_x_ord, MSD_CHL_y_ord = calcMSD_Cstyle(all_xs_CHL_ord, all_ys_CHL_ord, ordered_CM_x, ordered_CM_y)
# MSD_DSPC_x_ord, MSD_DSPC_y_ord = calcMSD_Cstyle(all_xs_DSPC_ord, all_ys_DSPC_ord, ordered_CM_x, ordered_CM_y)
# 
# 
# 
# 
# 
# 
# 
# MSD_dis = MSD_x_dis + MSD_y_dis
# MSD_DLIP_dis = MSD_DLIP_x_dis + MSD_DLIP_y_dis
# MSD_DSPC_dis = MSD_DSPC_x_dis + MSD_DSPC_y_dis
# MSD_CHL_dis = MSD_CHL_x_dis + MSD_CHL_y_dis
# 
# np.savetxt(output_dir + 'MSD_dis.dat', MSD_dis)
# np.savetxt(output_dir + 'MSD_DLIP_dis.dat', MSD_DLIP_dis)
# np.savetxt(output_dir + 'MSD_DSPC_dis.dat', MSD_DSPC_dis)
# np.savetxt(output_dir + 'MSD_CHL_dis.dat', MSD_CHL_dis)
# 
# MSD_ord =  MSD_x_ord + MSD_y_ord
# MSD_CHL_ord = MSD_CHL_x_ord + MSD_CHL_y_ord
# MSD_DSPC_ord = MSD_DSPC_x_ord + MSD_DSPC_y_ord
# #MSD_DLIP_ord = MSD_DLIP_x_ord + MSD_DLIP_y_ord
# 
# np.savetxt(output_dir + 'MSD_ord.dat', MSD_ord)
# #np.savetxt(output_dir + 'MSD_DLIP_ord.dat', MSD_DLIP_ord)
# np.savetxt(output_dir + 'MSD_DSPC_ord.dat', MSD_DSPC_ord)
# np.savetxt(output_dir + 'MSD_CHL_ord.dat', MSD_CHL_ord)
# 
# #TODO check if it is OK
# #D_test_tuple  = fit_linear_diffusion_data(MSD_DSPC_ord)
# 
# 
# #plt.plot(MSD_dis)
# #plt.plot(MSD_DLIP_dis)
# #plt.plot(MSD_CHL_dis)
# #plt.plot(MSD_DSPC_dis)
# 
# 
# 
# 
# plt.plot(all_xs_DSPC_ord[:,0]- ordered_CM_x[101:], all_ys_DSPC_ord[:,0]- ordered_CM_y[101:], c='green')
# plt.plot(all_xs_DSPC_ord[:,10]- ordered_CM_x[101:], all_ys_DSPC_ord[:,10]- ordered_CM_y[101:], c='green')
# 
# plt.plot(all_xs_DSPC_ord[:,50]- ordered_CM_x[101:], all_ys_DSPC_ord[:,50]- ordered_CM_y[101:], c='green')
# plt.plot(all_xs_CHL_ord[:,60]-ordered_CM_x[101:], all_ys_CHL_ord[:,60]-ordered_CM_y[101:], c ='red' )
# plt.plot(all_xs_CHL_ord[:,50]-ordered_CM_x[101:], all_ys_CHL_ord[:,50]-ordered_CM_y[101:], c ='red' )
# plt.plot(all_xs_CHL_ord[:,0]-ordered_CM_x[101:], all_ys_CHL_ord[:,0]-ordered_CM_y[101:], c='red' )
# plt.plot(all_xs_CHL_ord[:,1]-ordered_CM_x[101:], all_ys_CHL_ord[:,1]-ordered_CM_y[101:], c='red' )
# plt.plot(all_xs_CHL_ord[:,2]-ordered_CM_x[101:], all_ys_CHL_ord[:,2]-ordered_CM_y[101:], c ='red' )
# plt.axis("equal")
# 
# plt.plot(all_xs_CHL_dis[:,2]-disordered_CM_x[101:], all_ys_CHL_dis[:,2]-disordered_CM_y[101:] )
# plt.plot(all_xs_DLIP_dis[:,2]-disordered_CM_x[101:], all_ys_DLIP_dis[:,2]-disordered_CM_y[101:] )
# plt.axis("equal")
# 
# =============================================================================
#np.save(output_dir + 'test'+ str(ts.frame) + '.npy', coord_o)