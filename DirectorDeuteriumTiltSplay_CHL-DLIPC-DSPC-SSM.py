#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:59:33 2019

@author: anna
This script computes the Director order parameter, the Deuterium Order parameters,
lipids tilt and splay angles.



"""

import MDAnalysis
import matplotlib.pyplot as plt
import MDAnalysis.lib.NeighborSearch as NS
import numpy as np
from numpy.linalg import norm
import os
import sys

import pandas as pd 

top  = 'ANALYSIS/recentered_x.gro'
traj = 'ANALYSIS/recentered_x.xtc'



side = sys.argv[1] #'up'  #sys.argv[1] # "up" for upper leaflet "down" for lower leaflet



u = MDAnalysis.Universe(top,traj) 
resnames  = np.unique(u.select_atoms('all and name P or name O2').resnames)

def identify_leaflets(u, time_ts):
    """Assign lipids to respective leaflets"""
    z =  u.select_atoms("all").center_of_geometry()[2]
    COM_z= np.array([0,0,z]) #defines the global midplane position along z
    x, y, z = u.trajectory.ts.triclinic_dimensions[0][0], u.trajectory.ts.triclinic_dimensions[1][1], u.trajectory.ts.triclinic_dimensions[2][2]
    box = np.array([x, y, z, 90, 90, 90])    
        ### Determining side of the bilayer CHOL belongs to in this frame
    lipid1 = 'CHL'
    lipid2 = 'DLIP'
    lipid3 = 'SSM'
    lipid4 = 'DSPC'
        
    lpd1_atoms = u.select_atoms('resname %s and name O2'%lipid1)  
    lpd2_atoms = u.select_atoms('resname %s and name P '%lipid2) 
    lpd3_atoms = u.select_atoms('resname %s and name P '%lipid3) 
    lpd4_atoms = u.select_atoms('resname %s and name P '%lipid4)
    
    num_lpd2 = lpd2_atoms.n_atoms
    num_lpd3 = lpd3_atoms.n_atoms
    num_lpd4 = lpd4_atoms.n_atoms   
        # atoms in the upper leaflet as defined by insane.py or the CHARMM-GUI membrane builders
        # select cholesterol headgroups within 1.5 nm of lipid headgroups in the selected leaflet
        # this must be done because CHOL rapidly flip-flops between leaflets
        # so we must assign CHOL to each leaflet at every time step, and in large systems
        # with substantial membrane undulations, a simple cut-off in the z-axis just will not cut it
    if side == 'up':
        lpd2i = lpd2_atoms[:int((num_lpd2)/2)]
        lpd3i = lpd3_atoms[:int((num_lpd3)/2)]
        lpd4i = lpd4_atoms[:int((num_lpd4)/2)]
        

        lipids = lpd2i + lpd3i + lpd4i 

        ns_lipids = NS.AtomNeighborSearch(lpd1_atoms, box=box) 
        lpd1i = ns_lipids.search(lipids,15.0) #1.5 nm
        leaflet = lpd1i + lpd2i + lpd3i + lpd4i 

    elif side == 'down':
        lpd2i = lpd2_atoms[int((num_lpd2)/2):]
        lpd3i = lpd3_atoms[int((num_lpd3)/2):]
        lpd4i = lpd4_atoms[int((num_lpd4)/2):]

        lipids = lpd2i + lpd3i + lpd4i #+ lpd3i
        
        ns_lipids = NS.AtomNeighborSearch(lpd1_atoms, box=box)
        lpd1i = ns_lipids.search(lipids,15.0) # 1.5nm
        leaflet = lpd1i + lpd2i + lpd3i+  lpd4i 
    return lpd1i, lpd2i, lpd3i, lpd4i, COM_z, box, leaflet  



# =============================================================================
# Identify local normals to the lipids:
#   1- Find the local best fitting plane to each lipid and its neighbours within a cutoff
#      (taking into account the first neighbour to each lipid) using SVD algorithm
#   2- Find the normal to each plane and assign it to the each lipid   
    
 
def standard_fit(X): 
    #algorithm 3.1 from https://www.ltu.se/cms_fs/1.51590!/svd-fitting.pdf
    #easy explanation at: https://towardsdatascience.com/svd-8c2f72e264f
    # Find the average of points (centroid) along the columns
    C = np.average(X, axis=0)
    # Create CX vector (centroid to point) matrix
    CX = X - C
    # Singular value decomposition
    U, S, V = np.linalg.svd(CX)
    # The last row of V matrix indicate the eigenvectors of
    # smallest eigenvalues (singular values).
    N = V[-1] 
    return C, N



def get_normals_CM_headgroups(head_coord, box, cutoff): 
 normals = np.zeros((len(head_coord),3))
 indices = np.zeros((len(head_coord),1)) 
 
 distarr = MDAnalysis.lib.distances.distance_array(head_coord, head_coord, box=box)
 sigma_ij =  np.zeros ((len(distarr), len(distarr)))
 for row, i in enumerate(distarr):
     for col, j in enumerate(distarr):
         if distarr[row, col] < cutoff : #first neighbour 
             sigma_ij[row, col] = 1
         else : 
             sigma_ij[row, col] = 0
             
 for i in range(len(distarr)):            
     coords_for_plane = head_coord[sigma_ij[:,i] ==1]
     C, N = standard_fit(coords_for_plane)                
     normals[i] = N
     indices[i] = i
     
 return np.column_stack((normals[:,0], normals[:,1], np.abs(normals[:,2]))) , indices , sigma_ij
 
# =============================================================================

def order_parameter(angle):
    u = ((3/2)*(pow(np.cos(angle), 2))) - 0.5
    return u

def head_tail_angle(head, tail, normal):
    vect = head - tail        
    theta = np.arccos(np.sum(vect*normal, axis=1) / (norm(vect, axis =1) * norm(normal, axis=1)))      
    return theta
        
def compute_order_parameters(head, tail, normal):
    theta = head_tail_angle(head, tail, normal)
    u_lp = order_parameter(theta)
    return u_lp    


""" functions for DLIP """
def compute_directors_order_parameter_DLIP(DLIP_resid, leaflet):
    
    head_DLIP_1 =  u.select_atoms('resnum ' + str(DLIP_resid) + ' and name C24').positions
    tail_DLIP_1 =  u.select_atoms('resnum ' + str(DLIP_resid) + ' and (name 6C21 or name C216)').positions
    head_DLIP_2 =  u.select_atoms('resnum ' + str(DLIP_resid) + ' and name C34').positions
    tail_DLIP_2 =  u.select_atoms('resnum ' + str(DLIP_resid) + ' and (name 6C21 or name C216)').positions    
     
    u_DLIP_1 = compute_order_parameters(head_DLIP_1, tail_DLIP_1,  normal_leaflet[leaflet.resnames == 'DLIP'])
    u_DLIP_2 = compute_order_parameters(head_DLIP_2, tail_DLIP_2,  normal_leaflet[leaflet.resnames == 'DLIP'])
    
    return u_DLIP_1, u_DLIP_2


def compute_SCD_DLIP(DLIP_resid, leaflet, tails_DLIP_carbon):
    DLIP_order_param_sn1 = []    
    DLIP_order_param_sn2 = []
    
    for t, carbons in enumerate(tails_DLIP_carbon):  
        DLIP_Ci = u.select_atoms('resnum '+ str(DLIP_resid) + ' ' + 'and name C2%i'%carbons).positions 
        DLIP_HiR = u.select_atoms('resnum '+ str(DLIP_resid) + ' ' + 'and name H%iR'%carbons).positions    
        DLIP_Ci_sn1 = u.select_atoms('resnum '+ str(DLIP_resid) + ' ' + 'and name C3%i'%carbons).positions    
        DLIP_HiX = u.select_atoms('resnum '+ str(DLIP_resid) + ' ' + 'and name H%iX'%carbons).positions    
        
        DLIP_Scd_iR = compute_order_parameters(DLIP_Ci, DLIP_HiR, normal_leaflet[leaflet.resnames == 'DLIP']) 
        DLIP_Scd_iX = compute_order_parameters(DLIP_Ci_sn1, DLIP_HiX, normal_leaflet[leaflet.resnames == 'DLIP'])     
        DLIP_order_param_sn2.append(DLIP_Scd_iR)
        DLIP_order_param_sn1.append(DLIP_Scd_iX)
    return DLIP_order_param_sn2, DLIP_order_param_sn1
        
        
def compute_tilt_angle_DLIP(DLIP_resid, head_CM_DLIP, leaflet):
    
    chain_A_1 =  u.select_atoms('resnum ' + str(DLIP_resid) + ' and name C216').positions
    chain_A_2 =  u.select_atoms('resnum ' + str(DLIP_resid) + ' and name C217').positions
    chain_A_3 =  u.select_atoms('resnum ' + str(DLIP_resid) + ' and name C218').positions

    chain_B_1 = u.select_atoms('resnum ' + str(DLIP_resid) + ' and name C316').positions
    chain_B_2 = u.select_atoms('resnum ' + str(DLIP_resid) + ' and name C317').positions
    chain_B_3 = u.select_atoms('resnum ' + str(DLIP_resid) + ' and name C318').positions 
    
    last_three_carbons_CM_DLIP = np.average([chain_A_1,  chain_A_2,  chain_A_3, chain_B_1, chain_B_2, chain_B_3], axis = 0)
    tilt_angle_DLIP = head_tail_angle(head_CM_DLIP, last_three_carbons_CM_DLIP, normal_leaflet[leaflet.resnames == 'DLIP'])
    return  head_CM_DLIP, last_three_carbons_CM_DLIP, tilt_angle_DLIP, head_CM_DLIP - last_three_carbons_CM_DLIP 


def return_coordinates_chains_DLIP(DLIP_resid):
    
    chain_DLIP_A_1 = u.select_atoms('resnum ' + str(DLIP_resid) + ' and (name C26)').positions
    chain_DLIP_A_2 = u.select_atoms('resnum ' + str(DLIP_resid) + ' and (name C27)').positions
    chain_DLIP_A_3 = u.select_atoms('resnum ' + str(DLIP_resid) + ' and (name C28)').positions
    chain_DLIP_A_4 = u.select_atoms('resnum ' + str(DLIP_resid) + ' and (name C29)').positions
 
    chain_DLIP_B_1 = u.select_atoms('resnum ' + str(DLIP_resid) + ' and (name C36)').positions
    chain_DLIP_B_2 = u.select_atoms('resnum ' + str(DLIP_resid) + ' and (name C37)').positions
    chain_DLIP_B_3 = u.select_atoms('resnum ' + str(DLIP_resid) + ' and (name C38)').positions
    chain_DLIP_B_4 = u.select_atoms('resnum ' + str(DLIP_resid) + ' and (name C39)').positions   
     
    #coordinates of the lipids
    DLIP_C_o_G_chainA = np.average([chain_DLIP_A_1, chain_DLIP_A_2, chain_DLIP_A_3, chain_DLIP_A_4], axis = 0)
    DLIP_C_o_G_chainB = np.average([chain_DLIP_B_1, chain_DLIP_B_2, chain_DLIP_B_3, chain_DLIP_B_4], axis = 0) 
    return DLIP_C_o_G_chainA, DLIP_C_o_G_chainB


""" functions for DSPC """
def compute_directors_order_parameter_DSPC(DSPC_resid, leaflet):
    
    head_DSPC_1 =  u.select_atoms('resnum ' + str(DSPC_resid) + ' and name C24').positions
    tail_DSPC_1 =  u.select_atoms('resnum ' + str(DSPC_resid) + ' and (name 6C21 or name C216)').positions
    head_DSPC_2 =  u.select_atoms('resnum ' + str(DSPC_resid) + ' and name C34').positions
    tail_DSPC_2 =  u.select_atoms('resnum ' + str(DSPC_resid) + ' and (name 6C31 or name C316)').positions     
    u_DSPC_1 = compute_order_parameters(head_DSPC_1,tail_DSPC_1,  normal_leaflet[leaflet.resnames == 'DSPC'])
    u_DSPC_2 = compute_order_parameters(head_DSPC_2,tail_DSPC_2,  normal_leaflet[leaflet.resnames == 'DSPC'])
    
    return u_DSPC_1, u_DSPC_2

def return_coordinates_chains_DSPC(DSPC_resid):
    
    chain_DSPC_A_1 = u.select_atoms('resnum ' + str(DSPC_resid) + ' and (name C26)').positions
    chain_DSPC_A_2 = u.select_atoms('resnum ' + str(DSPC_resid) + ' and (name C27)').positions
    chain_DSPC_A_3 = u.select_atoms('resnum ' + str(DSPC_resid) + ' and (name C28)').positions
    chain_DSPC_A_4 = u.select_atoms('resnum ' + str(DSPC_resid) + ' and (name C29)').positions
 
    chain_DSPC_B_1 = u.select_atoms('resnum ' + str(DSPC_resid) + ' and (name C36)').positions
    chain_DSPC_B_2 = u.select_atoms('resnum ' + str(DSPC_resid) + ' and (name C37)').positions
    chain_DSPC_B_3 = u.select_atoms('resnum ' + str(DSPC_resid) + ' and (name C38)').positions
    chain_DSPC_B_4 = u.select_atoms('resnum ' + str(DSPC_resid) + ' and (name C39)').positions        
    #coordinates of the lipids
    DSPC_C_o_G_chainA = np.average([chain_DSPC_A_1, chain_DSPC_A_2, chain_DSPC_A_3, chain_DSPC_A_4], axis = 0)
    DSPC_C_o_G_chainB = np.average([chain_DSPC_B_1, chain_DSPC_B_2, chain_DSPC_B_3, chain_DSPC_B_4], axis = 0)
    
    return DSPC_C_o_G_chainA, DSPC_C_o_G_chainB

def compute_SCD_DSPC(DSPC_resid, leaflet, tails_DSPC_carbon):
    
    DSPC_order_param_sn1 = []    
    DSPC_order_param_sn2 = []
    
    for i, carbons in enumerate(tails_DSPC_carbon):  
        DSPC_Ci = u.select_atoms('resnum '+ str(DSPC_resid) + ' ' + 'and name C2%i'%carbons).positions 
        DSPC_HiR = u.select_atoms('resnum '+ str(DSPC_resid) + ' ' + 'and name H%iR'%carbons).positions    
        DSPC_Ci_sn1 = u.select_atoms('resnum '+ str(DSPC_resid) + ' ' + 'and name C3%i'%carbons).positions    
        DSPC_HiX = u.select_atoms('resnum '+ str(DSPC_resid) + ' ' + 'and name H%iX'%carbons).positions    
        
        DSPC_Scd_iR = compute_order_parameters(DSPC_Ci, DSPC_HiR, normal_leaflet[leaflet.resnames == 'DSPC']) 
        DSPC_Scd_iX = compute_order_parameters(DSPC_Ci_sn1, DSPC_HiX, normal_leaflet[leaflet.resnames == 'DSPC'])     
        DSPC_order_param_sn2.append(DSPC_Scd_iR)
        DSPC_order_param_sn1.append(DSPC_Scd_iX) 
    return DSPC_order_param_sn1, DSPC_order_param_sn2


def compute_tilt_angle_DSPC(DSPC_resid, head_CM_DSPC , leaflet):
        #compute the TILT angle for DSPC
    chain_A_1 =  u.select_atoms('resnum ' + str(DSPC_resid) + ' and name C216').positions
    chain_A_2 =  u.select_atoms('resnum ' + str(DSPC_resid) + ' and name C217').positions
    chain_A_3 =  u.select_atoms('resnum ' + str(DSPC_resid) + ' and name C218').positions

    chain_B_1 = u.select_atoms('resnum ' + str(DSPC_resid) + ' and name C316').positions
    chain_B_2 = u.select_atoms('resnum ' + str(DSPC_resid) + ' and name C317').positions
    chain_B_3 = u.select_atoms('resnum ' + str(DSPC_resid) + ' and name C318').positions 
    
    last_three_carbons_CM_DSPC = np.average([chain_A_1,  chain_A_2,  chain_A_3, chain_B_1, chain_B_2, chain_B_3], axis = 0)
    tilt_angle_DSPC = head_tail_angle(head_CM_DSPC, last_three_carbons_CM_DSPC, normal_leaflet[leaflet.resnames == 'DSPC'])
    
    return  head_CM_DSPC, last_three_carbons_CM_DSPC, tilt_angle_DSPC , head_CM_DSPC - last_three_carbons_CM_DSPC 
    
    


""" functions for SSM """
def compute_directors_order_parameter_SSM(SSM_resid, leaflet):
    
    head_SSM_1 =  u.select_atoms('resnum ' + str(SSM_resid) + ' and name C6S').positions
    tail_SSM_1 =  u.select_atoms('resnum ' + str(SSM_resid) + ' and (name C16S)').positions
    head_SSM_2 =  u.select_atoms('resnum ' + str(SSM_resid) + ' and name C6F').positions
    tail_SSM_2 =  u.select_atoms('resnum ' + str(SSM_resid) + ' and (name C16F )').positions     
    u_SSM_1 = compute_order_parameters(head_SSM_1,tail_SSM_1,  normal_leaflet[leaflet.resnames == 'SSM'])
    u_SSM_2 = compute_order_parameters(head_SSM_2,tail_SSM_2,  normal_leaflet[leaflet.resnames == 'SSM'])
    
    return u_SSM_1, u_SSM_2

def return_coordinates_chains_SSM(SSM_resid):
    
    chain_SSM_A_1 = u.select_atoms('resnum ' + str(SSM_resid) + ' and (name C8S)').positions
    chain_SSM_A_2 = u.select_atoms('resnum ' + str(SSM_resid) + ' and (name C9S)').positions
    chain_SSM_A_3 = u.select_atoms('resnum ' + str(SSM_resid) + ' and (name C10S)').positions
    chain_SSM_A_4 = u.select_atoms('resnum ' + str(SSM_resid) + ' and (name C11S)').positions
 
    chain_SSM_B_1 = u.select_atoms('resnum ' + str(SSM_resid) + ' and (name C8F)').positions
    chain_SSM_B_2 = u.select_atoms('resnum ' + str(SSM_resid) + ' and (name C9F)').positions
    chain_SSM_B_3 = u.select_atoms('resnum ' + str(SSM_resid) + ' and (name C10F)').positions
    chain_SSM_B_4 = u.select_atoms('resnum ' + str(SSM_resid) + ' and (name C11F)').positions        
    #coordinates of the lipids
    SSM_C_o_G_chainA = np.average([chain_SSM_A_1, chain_SSM_A_2, chain_SSM_A_3, chain_SSM_A_4], axis = 0)
    SSM_C_o_G_chainB = np.average([chain_SSM_B_1, chain_SSM_B_2, chain_SSM_B_3, chain_SSM_B_4], axis = 0)
    
    return SSM_C_o_G_chainA, SSM_C_o_G_chainB

def compute_SCD_SSM(SSM_resid, leaflet, tails_SSM_carbon):
    
    SSM_order_param_sn1 = []    
    SSM_order_param_sn2 = []
    
    for i, carbons in enumerate(tails_SSM_carbon):  
        SSM_Ci = u.select_atoms('resnum '+ str(SSM_resid) + ' ' + 'and name C%iS'%carbons).positions 
        SSM_HiR = u.select_atoms('resnum '+ str(SSM_resid) + ' ' + 'and name H%iS'%carbons).positions    
        SSM_Ci_sn1 = u.select_atoms('resnum '+ str(SSM_resid) + ' ' + 'and name C%iF'%carbons).positions    
        SSM_HiX = u.select_atoms('resnum '+ str(SSM_resid) + ' ' + 'and name H%iS'%carbons).positions    
        
        SSM_Scd_iR = compute_order_parameters(SSM_Ci, SSM_HiR, normal_leaflet[leaflet.resnames == 'SSM']) 
        SSM_Scd_iX = compute_order_parameters(SSM_Ci_sn1, SSM_HiX, normal_leaflet[leaflet.resnames == 'SSM'])     
        SSM_order_param_sn2.append(SSM_Scd_iR)
        SSM_order_param_sn1.append(SSM_Scd_iX) 
    return SSM_order_param_sn1, SSM_order_param_sn2


def compute_tilt_angle_SSM(SSM_resid, head_CM_SSM, leaflet):
    # take atoms to define the CM of teh headgroups. These will be used to find the local normals
    #compute the TILT angle for SSM
    chain_A_1 =  u.select_atoms('resnum ' + str(SSM_resid) + ' and name C16S').positions
    chain_A_2 =  u.select_atoms('resnum ' + str(SSM_resid) + ' and name C17S').positions
    chain_A_3 =  u.select_atoms('resnum ' + str(SSM_resid) + ' and name C18S').positions

    chain_B_1 = u.select_atoms('resnum ' + str(SSM_resid) + ' and name C16F').positions
    chain_B_2 = u.select_atoms('resnum ' + str(SSM_resid) + ' and name C17F').positions
    chain_B_3 = u.select_atoms('resnum ' + str(SSM_resid) + ' and name C18F').positions 
    
    last_three_carbons_CM_SSM = np.average([chain_A_1,  chain_A_2,  chain_A_3, chain_B_1, chain_B_2, chain_B_3], axis = 0)
    tilt_angle_SSM = head_tail_angle(head_CM_SSM, last_three_carbons_CM_SSM, normal_leaflet[leaflet.resnames == 'SSM'])
    
    return  head_CM_SSM, last_three_carbons_CM_SSM, tilt_angle_SSM , head_CM_SSM - last_three_carbons_CM_SSM 
    


""" Compute splay angle   """
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

import numpy.linalg as la
def compute_angle(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
        The sign of the angle is dependent on the order of v1 and v2
        so acos(norm(dot(v1, v2))) does not work and atan2 has to be used, see:
        https://stackoverflow.com/questions/21483999/using-atan2-to-find-angle-between-two-vectors
    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    angle = np.arctan2(sinang, cosang)
    return angle 



def compute_splays(first_neighbors_splay, t, all_tilts_vect_upper):
    angles_splay = np.zeros(( len(first_neighbors_splay[0]), 4))
    t = np.full(len(first_neighbors_splay[0]), t)
    for i in range(len(first_neighbors_splay[0])):        
        angles_splay[i, :] = compute_angle(all_tilts_vect_upper[first_neighbors_splay[0][i]], all_tilts_vect_upper[first_neighbors_splay[1][i]]), first_neighbors_splay[0][i], first_neighbors_splay[1][i], t[i]
    return angles_splay 



def main(leaflet, COM_z, box, lipid1, lipid2,lipid3, lipid4, 
         DLIP_resid, DSPC_resid, SSM_resid, CHL_resid, 
         head_CHL_1, head_CM_lipid2, head_CM_lipid3, head_CM_lipid4, 
         tails_DLIP_carbon, tails_DSPC_carbon, tails_SSM_carbon):         
    """ DLIP """     
    #DLIP_resid =' '.join(str(x) for x in DLIP_per_leaflet.resnums[:] )
    # compute the director order parameter
    u_DLIP_1, u_DLIP_2 = compute_directors_order_parameter_DLIP(DLIP_resid, leaflet)
    DLIP_C_o_G_chainA, DLIP_C_o_G_chainB = return_coordinates_chains_DLIP(DLIP_resid)
    
    #compute the tilt angles (according to the definition of https://doi.org/10.1021/ct400492e)
    head_CM_DLIP, last_three_carbons_CM_DLIP, tilt_angle_DLIP, tilt_vect_DLIP = compute_tilt_angle_DLIP(DLIP_resid, head_CM_lipid2, leaflet)

    #compute the SCD order parameter      
    DLIP_order_param_sn2, DLIP_order_param_sn1 = compute_SCD_DLIP(DLIP_resid, leaflet, tails_DLIP_carbon)
        
    """ SSM """ 
    #compute the director order parameter for DSPC
    u_SSM_1, u_SSM_2 = compute_directors_order_parameter_SSM(SSM_resid, leaflet) 
    SSM_C_o_G_chainA, SSM_C_o_G_chainB = return_coordinates_chains_SSM(SSM_resid)
    
    #compute the SCD order 
    SSM_order_param_sn1, SSM_order_param_sn2 = compute_SCD_SSM(SSM_resid, leaflet, tails_SSM_carbon)
    
    #compute the tilt angles (according to the definition of https://doi.org/10.1021/ct400492e)
    head_CM_SSM, last_three_carbons_CM_SSM, tilt_angle_SSM, tilt_vect_SSM = compute_tilt_angle_SSM(SSM_resid,head_CM_lipid3, leaflet)
    
    
    """ DSPC """ 
    #compute the director order parameter for DSPC
    u_DSPC_1, u_DSPC_2 = compute_directors_order_parameter_DSPC(DSPC_resid, leaflet) 
    DSPC_C_o_G_chainA, DSPC_C_o_G_chainB = return_coordinates_chains_DSPC(DSPC_resid)
    
    #compute the SCD order 
    DSPC_order_param_sn1, DSPC_order_param_sn2 = compute_SCD_DSPC(DSPC_resid, leaflet, tails_DSPC_carbon)
    
    #compute the tilt angles (according to the definition of https://doi.org/10.1021/ct400492e)
    head_CM_DSCP, last_three_carbons_CM_DSCP, tilt_angle_DSPC, tilt_vect_DSPC = compute_tilt_angle_DSPC(DSPC_resid,head_CM_lipid4, leaflet)

             
    """ CHL """
    tail_CHL_1 = u.select_atoms('resnum ' + str(CHL_resid) + ' and name C65').positions
    chain_CHL_A_1 = u.select_atoms('resnum ' + str(CHL_resid) + ' and name C24').positions
 
    #Todo check how to calculate theta
    u_CHL_1 = compute_order_parameters(head_CHL_1, tail_CHL_1,  normal_leaflet[leaflet.resnames == 'CHL'])
      
    tilt_angle_CHL = head_tail_angle(head_CHL_1, tail_CHL_1,  normal_leaflet[leaflet.resnames == 'CHL'])
    
    tilt_vect_CHL= head_CHL_1 - tail_CHL_1
    
    CHL_C_o_G_chainA = np.average([chain_CHL_A_1, head_CHL_1, tail_CHL_1], axis = 0)
    
    O2_z = u.select_atoms('resnum ' + str(CHL_resid) + ' and name O2').positions 
    C24_z = chain_CHL_A_1
    
    
    O2_CM = O2_z - COM_z
    C1_CM = head_CHL_1 - COM_z
    C24_CM =C24_z - COM_z
    
    dist_O2_CM = O2_CM [:,2]  
    dist_C24_CM = C24_CM[:,2]
    dist_C1_CM = C1_CM [:,2]
   
    """ END """
    u_CHL_1 = u_CHL_1.reshape(len(u_CHL_1), 1)
    u_DLIP_1 = u_DLIP_1.reshape(len(u_DLIP_1), 1) 
    u_DLIP_2 = u_DLIP_2.reshape(len(u_DLIP_2), 1)
    u_DSPC_1 = u_DSPC_1.reshape(len(u_DSPC_1), 1) 
    u_DSPC_2 = u_DSPC_2.reshape(len(u_DSPC_2), 1) 
    u_SSM_1 = u_SSM_1.reshape(len(u_SSM_1), 1) 
    u_SSM_2 = u_SSM_2.reshape(len(u_SSM_2), 1)

    CHL_res =  lipid1.resnums
    DLIP_res = lipid2.resnums
    SSM_res = lipid3.resnums    
    DSPC_res = lipid4.resnums

        
    res1 = CHL_res.reshape(len(CHL_res), 1)
    res2 = DLIP_res.reshape(len(DLIP_res), 1)
    res3 = DSPC_res.reshape(len(DSPC_res), 1)
    res4 = SSM_res.reshape(len(SSM_res), 1)
 
    u_all = np.vstack((u_CHL_1, u_DLIP_1, u_DLIP_2, u_SSM_1, u_SSM_2, u_DSPC_1, u_DSPC_2))  
    lpd_coords = np.vstack((CHL_C_o_G_chainA, DLIP_C_o_G_chainA, DLIP_C_o_G_chainB, SSM_C_o_G_chainA, SSM_C_o_G_chainB, DSPC_C_o_G_chainA, DSPC_C_o_G_chainB )) 
    lpd_res_all =np.vstack((res1, res2, res2, res3, res3, res4, res4))
    u_1 = np.vstack((u_CHL_1, u_DLIP_1, u_SSM_1, u_DSPC_1))
    u_2 = np.vstack((u_CHL_1, u_DLIP_2, u_SSM_2, u_DSPC_2))
     
        
    return lpd_coords, lpd_res_all, u_all, res1, CHL_C_o_G_chainA, dist_O2_CM, dist_C24_CM, dist_C1_CM, res2, res3, res4, box,\
            DLIP_order_param_sn2, DLIP_order_param_sn1, DLIP_C_o_G_chainA, DLIP_C_o_G_chainB,\
            DSPC_order_param_sn2, DSPC_order_param_sn1, DSPC_C_o_G_chainA, DSPC_C_o_G_chainB,\
            SSM_order_param_sn2, SSM_order_param_sn1, SSM_C_o_G_chainA, SSM_C_o_G_chainB,\
            tilt_angle_DSPC,  tilt_angle_DLIP, tilt_angle_CHL, tilt_angle_SSM, tilt_vect_DSPC, tilt_vect_DLIP, tilt_vect_CHL, tilt_vect_SSM, u_1, u_2
    
directory_directors = "ANALYSIS/directors/" 
directory_deuterium = "ANALYSIS/deuterium/"
directory_tilts = "ANALYSIS/tilts_local_normals/"
 
#create the directory in the file path if it does not exist
if not os.path.exists(directory_directors):
    os.makedirs(directory_directors)
    
if not os.path.exists(directory_deuterium):
    os.makedirs(directory_deuterium)

if not os.path.exists(directory_tilts):
    os.makedirs(directory_tilts)

tails_DLIP_carbon = np.arange(2, 19)
tails_DSPC_carbon = np.arange(2, 19)
tails_SSM_carbon = np.arange(2, 19)

av_C_DLIP_sn1 = []
av_C_DLIP_sn2 = []
av_C_SSM_sn1 = []
av_C_SSM_sn2 = []
av_C_DSPC_sn1 = []
av_C_DSPC_sn2 = [] 
#results_splay = [] #should go befor the ts loop
for ts in u.trajectory[0:u.trajectory.n_frames:1]:  #every 1 ns
    #lipid1 == CHOL, lipid2 == DLIPC, lipid4 == DSPC
    lipid1, lipid2, lipid3, lipid4, COM_z, box,leaflet = identify_leaflets(u, ts)
    lpd_resid =' '.join(str(x) for x in leaflet.resnums[:])   
    
    #headgroups= u.select_atoms('resnum ' + str(lpd_resid) + ' and ( (name P or name O2)) ')
    
    lpd1_resid =' '.join(str(x) for x in lipid1.resnums[:]) 
    lpd2_resid =' '.join(str(x) for x in lipid2.resnums[:]) 
    lpd3_resid =' '.join(str(x) for x in lipid3.resnums[:]) 
    lpd4_resid =' '.join(str(x) for x in lipid4.resnums[:])  
    
    #headgroups_for_normals CHL   
    head_CM_lipid1 = u.select_atoms('resnum ' + str(lpd1_resid) + ' and name C1').positions 
    
    #headgroups_for_normals DLIPC
    head_1_lpd2 = u.select_atoms('resnum ' + str(lpd2_resid) + ' and name P').positions
    head_2_lpd2 = u.select_atoms('resnum ' + str(lpd2_resid) + ' and name C3').positions
    head_3_lpd2 = u.select_atoms('resnum ' + str(lpd2_resid) + ' and name C21').positions    
    head_4_lpd2 = u.select_atoms('resnum ' + str(lpd2_resid) + ' and name C2').positions 
    head_CM_lipid2 = np.average([head_1_lpd2,  head_2_lpd2,  head_3_lpd2, head_4_lpd2], axis = 0)
    
    
    #headgroups_for_normals SSM   
    head_1_lpd3 = u.select_atoms('resnum ' + str(lpd3_resid) + ' and name P').positions
    head_2_lpd3 = u.select_atoms('resnum ' + str(lpd3_resid) + ' and name C3S').positions
    head_3_lpd3 = u.select_atoms('resnum ' + str(lpd3_resid) + ' and name C2S').positions    
    head_4_lpd3 = u.select_atoms('resnum ' + str(lpd3_resid) + ' and name C1F').positions 
    head_CM_lipid3 = np.average([head_1_lpd3,  head_2_lpd3,  head_3_lpd3, head_4_lpd3], axis = 0)
    
    
    #headgroups_for_normals  DSPC
    head_1_lpd4 = u.select_atoms('resnum ' + str(lpd4_resid) + ' and name P').positions
    head_2_lpd4 = u.select_atoms('resnum ' + str(lpd4_resid) + ' and name C3').positions
    head_3_lpd4 = u.select_atoms('resnum ' + str(lpd4_resid) + ' and name C21').positions    
    head_4_lpd4 = u.select_atoms('resnum ' + str(lpd4_resid) + ' and name C2').positions 
    head_CM_lipid4 = np.average([head_1_lpd4,  head_2_lpd4,  head_3_lpd4, head_4_lpd4], axis = 0) 
    
    headgroups_coord = np.vstack((head_CM_lipid1, head_CM_lipid2, head_CM_lipid3, head_CM_lipid4))
  
    normal_leaflet, indices,  sigma = get_normals_CM_headgroups(headgroups_coord, box, cutoff=15) 
    
    
    coord, residues, directors, chl_res, chl_coord, distance_head_chl_to_center, distance_tail_chl_to_center,distance_C1_chl_to_center, DLIP_res, ssm_res, dspc_res, box,DLIP_order_param_sn2, DLIP_order_param_sn1,DLIP_C_o_G_chainA, DLIP_C_o_G_chainB,DSPC_order_param_sn2, DSPC_order_param_sn1, DSPC_C_o_G_chainA,DSPC_C_o_G_chainB, SSM_order_param_sn2, SSM_order_param_sn1, SSM_C_o_G_chainA, SSM_C_o_G_chainB, tilt_angle_DSPC,tilt_angle_DLIP, tilt_angle_CHL, tilt_angle_SSM, tilt_vect_DSPC, tilt_vect_DLIP, tilt_vect_CHL, tilt_vect_SSM, u_1, u_2  = main(leaflet, COM_z, box, lipid1, lipid2, lipid3, lipid4, lpd2_resid, lpd4_resid, lpd3_resid, lpd1_resid, head_CM_lipid1, head_CM_lipid2,head_CM_lipid3, head_CM_lipid4, tails_DLIP_carbon,tails_DSPC_carbon, tails_SSM_carbon)
    
    
    DLIP_sn1= np.mean(np.vstack((DLIP_order_param_sn1)) , axis=0)
    DLIP_sn2= np.mean(np.vstack((DLIP_order_param_sn2)), axis=0)
    SSM_sn1= np.mean(np.vstack((SSM_order_param_sn1)) , axis=0)
    SSM_sn2= np.mean(np.vstack((SSM_order_param_sn2)), axis=0)     
    DSPC_sn1= np.mean(np.vstack((DSPC_order_param_sn1)) , axis=0)
    DSPC_sn2= np.mean(np.vstack((DSPC_order_param_sn2)), axis=0) 
    
    av_C_DLIP_sn1.append(DLIP_order_param_sn1)
    av_C_DLIP_sn2.append(DLIP_order_param_sn2)
    av_C_SSM_sn1.append(SSM_order_param_sn1)
    av_C_SSM_sn2.append(SSM_order_param_sn2)
    av_C_DSPC_sn1.append(DSPC_order_param_sn1)
    av_C_DSPC_sn2.append(DSPC_order_param_sn2)    
    
    resids = np.vstack((chl_res, DLIP_res, ssm_res, dspc_res))
    resnames = np.vstack(( np.array(['CHL']*len(chl_res)).reshape(len(chl_res),1), np.array(['DLIP']*len(DLIP_res)).reshape(len(DLIP_res),1), np.array(['SSM']*len(ssm_res)).reshape(len(ssm_res),1), np.array(['DSPC']*len(dspc_res)).reshape(len(dspc_res),1) ))
    Scd_sn1 = np.vstack(( np.array([np.nan]*len(chl_res)).reshape(len(chl_res),1), DLIP_sn1.reshape(len(DLIP_sn1), 1), SSM_sn1.reshape(len(SSM_sn1), 1), DSPC_sn1.reshape(len(DSPC_sn1), 1) ))
    Scd_sn2 = np.vstack(( np.array([np.nan]*len(chl_res)).reshape(len(chl_res),1), DLIP_sn2.reshape(len(DLIP_sn2), 1), SSM_sn2.reshape(len(SSM_sn2), 1), DSPC_sn2.reshape(len(DSPC_sn2), 1) ))
    tilt_angles = np.vstack(( tilt_angle_CHL.reshape(len(tilt_angle_CHL),1), tilt_angle_DLIP.reshape(len(tilt_angle_DLIP),1), tilt_angle_SSM.reshape(len(tilt_angle_SSM),1), tilt_angle_DSPC.reshape(len(tilt_angle_DSPC),1) ))
    tilt_vects = np.vstack(( tilt_vect_CHL, tilt_vect_DLIP, tilt_vect_SSM, tilt_vect_DSPC))
    lpd_coords1 = np.vstack((chl_coord, DLIP_C_o_G_chainA, SSM_C_o_G_chainA,  DSPC_C_o_G_chainA)) 
    lpd_coords2 = np.vstack((chl_coord, DLIP_C_o_G_chainB, SSM_C_o_G_chainB,  DSPC_C_o_G_chainB))       
    times = [ts.frame]*len(resids) 
    
    
    
    
    """ SPLAY calcultions """
    """ read in the headgroups coordinates and find who is the neighbours to every single lipid """
    dist_vect_arr = MDAnalysis.lib.distances.distance_array(headgroups_coord, headgroups_coord, box=box)      
    """ compute only the firs neighbour""" 
    first_neighbors= np.argsort(dist_vect_arr, axis =0)[0:2,:]
    """ compute the splay angle for every lipid and its firts neighbour and store the information into an array """
    angles_splay = (compute_splays(first_neighbors, ts.frame, tilt_vects)) #this is an array

    df = pd.DataFrame({'Time': times, 'Resid': resids.flatten(),'Resnames': resnames.flatten(), 'Scd_sn1': Scd_sn1.flatten(),'Scd_sn2': Scd_sn2.flatten(),
                      'u_1': u_1.flatten() , 'u_2': u_2.flatten(), 'CM coord1 X': lpd_coords1[:,0], 'CM coord1 Y': lpd_coords1[:,1],'CM coord1 Z': lpd_coords1[:,2],
                      'CM coord2 X': lpd_coords2[:,0], 'CM coord2 Y': lpd_coords2[:,1],'CM coord2 Z': lpd_coords2[:,2],                     
                      'Head Coord X': headgroups_coord[:,0], 'Head Coord Y': headgroups_coord[:,1], 'Head Coord Z': headgroups_coord[:,2] ,'Tilt_angles': tilt_angles.flatten(), #,
                      'Tilt_vects X': tilt_vects[:,0], 'Tilt_vects Y': tilt_vects[:,1], 'Tilt_vects Z':  tilt_vects[:,2], 'Splay': angles_splay[:,0] })
    df.to_pickle(directory_directors + 'Dataframe'+ str(side) + str(ts.frame))
    box_df = pd.DataFrame({'Time':times[0] , 'box_x': [box[0]], 'box_y': [box[1]], 'box_z': [box[2]], 'alpha' : [box[3]], 'beta' : [box[4]], 'gamma' : [box[5]]})
    box_df.to_csv( directory_directors + '/box' +str(ts.frame) +str(side)+ '.csv', index = None, header=True)
  #  print("--- %s seconds ---" % (time.time() - start_time)) 

    if side == 'up':
        
        np.save(directory_directors + 'directors_upper_tail_' + str(ts.frame) + '.npy', directors)
        np.save(directory_directors + 'coordinates_upper_tail_' + str(ts.frame) + '.npy', coord)
        np.save(directory_directors + 'residues_upper_tail_' + str(ts.frame) + '.npy'  , residues)        
        np.save(directory_directors + 'cholesterol_upper_tail_' + str(ts.frame) + '.npy', chl_res)
        np.save(directory_directors+  'cholesterol_angle_upper_tail_' + str(ts.frame) + '.npy', tilt_angle_CHL)
        np.save(directory_directors + 'cholesterol_coord_upper_tail_' + str(ts.frame) + '.npy', chl_coord)
        np.save(directory_directors + 'cholesterol_H_distance_upper_tail_' + str(ts.frame) + '.npy', distance_head_chl_to_center)
        np.save(directory_directors + 'cholesterol_T_distance_upper_tail_' + str(ts.frame) + '.npy', distance_tail_chl_to_center)
        np.save(directory_directors + 'cholesterol_C1_distance_upper_tail_' + str(ts.frame) + '.npy', distance_C1_chl_to_center)
        np.save(directory_directors + 'dlipc_upper_tail_' + str(ts.frame) + '.npy', DLIP_res)
        np.save(directory_directors + 'dspc_upper_tail_' + str(ts.frame) + '.npy', dspc_res)
        np.save(directory_directors + 'ssm_upper_tail_' + str(ts.frame) + '.npy', ssm_res)
        np.save(directory_directors + 'box' + str(ts.frame) + '.npy', box)
        
        np.save(directory_tilts  + 'box' + str(ts.frame) + '.npy', box)
        np.save(directory_tilts + 'tilt_angles_CHL_upper_tail_' + str(ts.frame) + '.npy', tilt_angle_CHL)
        np.save(directory_tilts + 'tilt_angles_DLIP_upper_tail_' + str(ts.frame) + '.npy', tilt_angle_DLIP)  
        np.save(directory_tilts + 'tilt_angles_DSPC_upper_tail_' + str(ts.frame) + '.npy', tilt_angle_DSPC)  
        np.save(directory_tilts + 'tilt_vect_CHL_upper_tail_' + str(ts.frame) + '.npy', tilt_vect_CHL)
        np.save(directory_tilts + 'tilt_vect_DLIP_upper_tail_' + str(ts.frame) + '.npy', tilt_vect_DLIP)  
        np.save(directory_tilts + 'tilt_vect_DSPC_upper_tail_' + str(ts.frame) + '.npy', tilt_vect_DSPC)  
               
        
        np.save(directory_tilts + 'normalsleaflet_indices_sigma_upper_tail_' + str(ts.frame) + '.npy', normal_leaflet, indices,  sigma) 
        np.save(directory_tilts + 'headgroups_coords_upper_tail_' + str(ts.frame) + '.npy', headgroups_coord) 
        
        np.save(directory_deuterium  + 'box' + str(ts.frame) + '.npy', box)  
        np.save(directory_deuterium  + 'DLIP_SCD_coordinate_sn1_upper_tail_' + str(ts.frame) + '.npy', DLIP_sn2, DLIP_C_o_G_chainA)
        np.save(directory_deuterium  + 'DLIP_SCD_coordinate_sn2_upper_tail_' + str(ts.frame) + '.npy', DLIP_sn1, DLIP_C_o_G_chainB)  
        np.save(directory_deuterium  + 'DSPC_SCD_coordinate_sn1_upper_tail_' + str(ts.frame) + '.npy', DSPC_sn2, DSPC_C_o_G_chainA)
        np.save(directory_deuterium  + 'DSPC_SCD_coordinate_sn2_upper_tail_' + str(ts.frame) + '.npy', DSPC_sn1, DSPC_C_o_G_chainB)
        
        
    elif side == 'down':
        
        np.save(directory_directors+ 'directors_lower_tail_' + str(ts.frame) + '.npy', directors)
        np.save(directory_directors + 'coordinates_lower_tail_' + str(ts.frame) + '.npy', coord)
        np.save(directory_directors + 'residues_lower_tail_' + str(ts.frame) + '.npy'  , residues)        
        np.save(directory_directors + 'cholesterol_lower_tail_' + str(ts.frame) + '.npy', chl_res)
        np.save(directory_directors + 'cholesterol_angle_lower_tail_' + str(ts.frame) + '.npy', tilt_angle_CHL)
        np.save(directory_directors + 'cholesterol_coord_lower_tail_' + str(ts.frame) + '.npy', chl_coord)
        np.save(directory_directors + 'cholesterol_H_distance_lower_tail_' + str(ts.frame) + '.npy', distance_head_chl_to_center)
        np.save(directory_directors + 'cholesterol_T_distance_lower_tail_' + str(ts.frame) + '.npy', distance_tail_chl_to_center)
        np.save(directory_directors + 'cholesterol_C1_distance_lower_tail_' + str(ts.frame) + '.npy', distance_C1_chl_to_center)
        np.save(directory_directors + 'dlipc_lower_tail_' + str(ts.frame) + '.npy', DLIP_res)
        np.save(directory_directors + 'dspc_lower_tail_' + str(ts.frame) + '.npy', dspc_res)
        np.save(directory_directors + 'ssm_lower_tail_' + str(ts.frame) + '.npy', ssm_res)
        np.save(directory_directors + 'box' + str(ts.frame) + '.npy', box)

        np.save(directory_tilts  + 'box' + str(ts.frame) + '.npy', box)
        np.save(directory_tilts + 'tilt_angles_CHL_lower_tail_' + str(ts.frame) + '.npy', tilt_angle_CHL, tilt_vect_CHL)
        np.save(directory_tilts + 'tilt_angles_DLIP_lower_tail_' + str(ts.frame) + '.npy', tilt_angle_DLIP, tilt_vect_DLIP )  
        np.save(directory_tilts + 'tilt_angles_DSPC_lower_tail_' + str(ts.frame) + '.npy', tilt_angle_DSPC, tilt_vect_DSPC)  
        np.save(directory_tilts + 'tilt_vect_CHL_lower_tail_' + str(ts.frame) + '.npy', tilt_vect_CHL)
        np.save(directory_tilts + 'tilt_vect_DLIP_lower_tail_' + str(ts.frame) + '.npy', tilt_vect_DLIP)  
        np.save(directory_tilts + 'tilt_vect_DSPC_lower_tail_' + str(ts.frame) + '.npy', tilt_vect_DSPC)  
        
        np.save(directory_tilts + 'normalsleaflet_indices_sigma_lower_tail_' + str(ts.frame) + '.npy', normal_leaflet, indices,  sigma) 
        np.save(directory_tilts + 'headgroups_coords_lower_tail_' + str(ts.frame) + '.npy', headgroups_coord) 
        

        np.save(directory_deuterium  + 'box' + str(ts.frame) + '.npy', box)  
        np.save(directory_deuterium  + 'DLIP_SCD_coordinate_sn1_lower_tail_' + str(ts.frame) + '.npy', DLIP_sn2, DLIP_C_o_G_chainA)
        np.save(directory_deuterium  + 'DLIP_SCD_coordinate_sn2_lower_tail_' + str(ts.frame) + '.npy', DLIP_sn1, DLIP_C_o_G_chainB)  
        np.save(directory_deuterium  + 'DSPC_SCD_coordinate_sn1_lower_tail_' + str(ts.frame) + '.npy', DSPC_sn2, DSPC_C_o_G_chainA)
        np.save(directory_deuterium  + 'DSPC_SCD_coordinate_sn2_lower_tail_' + str(ts.frame) + '.npy', DSPC_sn1, DSPC_C_o_G_chainB)

deuterium_C_DLIP_sn1_t = []
deuterium_C_DLIP_sn2_t = []  
deuterium_C_DSPC_sn1_t = []
deuterium_C_DSPC_sn2_t = []
deuterium_C_SSM_sn1_t = []
deuterium_C_SSM_sn2_t = []
for i in range(u.trajectory.n_frames):
    deuterium_C_DLIP_sn1_t.append(np.mean(av_C_DLIP_sn1[i], axis=1))
    deuterium_C_DLIP_sn2_t.append(np.mean(av_C_DLIP_sn2[i], axis=1))    
    deuterium_C_DSPC_sn1_t.append(np.mean(av_C_DSPC_sn1[i], axis=1))    
    deuterium_C_DSPC_sn2_t.append(np.mean(av_C_DSPC_sn2[i], axis=1))
    deuterium_C_SSM_sn1_t.append(np.mean(av_C_SSM_sn1[i], axis=1))    
    deuterium_C_SSM_sn2_t.append(np.mean(av_C_SSM_sn2[i], axis=1))   
    
deuterium_C_DLIP_sn1 =  np.mean(deuterium_C_DLIP_sn1_t,  axis=0) 
deuterium_C_DLIP_sn2 =  np.mean(deuterium_C_DLIP_sn2_t,  axis=0)    
deuterium_C_DSPC_sn1 =  np.mean(deuterium_C_DSPC_sn1_t,  axis=0) 
deuterium_C_DSPC_sn2 =  np.mean(deuterium_C_DSPC_sn2_t,  axis=0) 
deuterium_C_SSM_sn1 =  np.mean(deuterium_C_SSM_sn1_t,  axis=0) 
deuterium_C_SSM_sn2 =  np.mean(deuterium_C_SSM_sn2_t,  axis=0)    

np.savetxt(directory_deuterium + 'Deuterium_order_DLIP_Sn1'+ str(side)+ '.dat', np.column_stack((tails_DLIP_carbon, -1*deuterium_C_DLIP_sn1)))
np.savetxt(directory_deuterium + 'Deuterium_order_DLIP_Sn2'+ str(side)+ '.dat', np.column_stack((tails_DLIP_carbon, -1*deuterium_C_DLIP_sn2)))
np.savetxt(directory_deuterium + 'Deuterium_order_DSPC_Sn1'+ str(side)+ '.dat', np.column_stack((tails_DSPC_carbon, -1*deuterium_C_DSPC_sn1)))
np.savetxt(directory_deuterium + 'Deuterium_order_DSPC_Sn2'+ str(side)+ '.dat', np.column_stack((tails_DSPC_carbon, -1*deuterium_C_DSPC_sn2)))
np.savetxt(directory_deuterium + 'Deuterium_order_SSM_Sn1'+ str(side)+ '.dat', np.column_stack((tails_SSM_carbon, -1*deuterium_C_SSM_sn1)))
np.savetxt(directory_deuterium + 'Deuterium_order_SSM_Sn2'+ str(side)+ '.dat', np.column_stack((tails_SSM_carbon, -1*deuterium_C_SSM_sn2)))


#===============================================================================================================
# =============================================================================
# plt.figure()
# plt.scatter(df['CM coord1 X'].values, df['CM coord1 Y'].values, c=df['u_1'].values)
# plt.scatter(df['CM coord2 X'].values, df['CM coord2 Y'].values, c=df['u_2'].values)       
# plt.colorbar()
#         
# plt.scatter(df['CM coord1 X'].values, df['CM coord1 Y'].values, c=(df['Scd_sn2'].values), cmap='Set1', vmin=-0.5  )
# plt.scatter(df['CM coord2 X'].values, df['CM coord2 Y'].values, c=(df['Scd_sn1'].values), cmap='Set1',  vmin=-0.5)
# plt.colorbar()
# 
# 
# def first_quadrant(x):
#     if (x >= 90) :
#         x= 180 - x
#     else:
#         x= x
#     return x
# 
# angles_in_degree = np.degrees(df['Tilt_angles'].values)
# angles_first_quadrant = np.array([first_quadrant(x) for x in angles_in_degree])
# 
# 
# plt.scatter(df['CM coord1 X'].values, df['CM coord1 Y'].values, c = angles_first_quadrant,cmap='coolwarm' )
# plt.scatter(df['CM coord2 X'].values, df['CM coord2 Y'].values , c = angles_first_quadrant, cmap='coolwarm')
# plt.colorbar()
# plt.axis('equal')
# 
# plt.figure()
# plt.scatter(df['CM coord1 X'].values, df['CM coord1 Y'].values, c = np.degrees(df['Splay'].values))
# #plt.figure()
# plt.scatter(df['CM coord2 X'].values, df['CM coord2 Y'].values , c = np.degrees(df['Splay'].values ))
# plt.colorbar()
# plt.axis('equal')
# 
# plt.figure()
# plt.scatter(df['CM coord1 X'].values, df['CM coord1 Y'].values, c = order_parameter(df['Splay'].values))
# #plt.figure()
# plt.scatter(df['CM coord2 X'].values, df['CM coord2 Y'].values , c = order_parameter(df['Splay'].values ))
# plt.colorbar()
# plt.axis('equal')
# 
# 
# plt.scatter(df['CM coord1 X'].values, df['CM coord1 Y'].values, c = order_parameter(np.radians(angles_first_quadrant)) )
# plt.scatter(df['CM coord2 X'].values, df['CM coord2 Y'].values , c =order_parameter( np.radians(angles_first_quadrant) ))
# plt.colorbar()
# plt.axis('equal')
# 
# 
# 
# 
# 
# 
# 
# """Plotting  maps """
# 
# def pbc_coordinates(x, y,box_x, box_y, value):
#     #main
#     x1 = x
#     y1 = y
#     #under main
#     x2 = x
#     y2 = y - box_y
#     #upper main
#     x3 = x
#     y3 = y + box_y
#     #right main 4
#     x4 = x + box_x
#     y4 = y
#     #left main 5
#     x5 = x - box_x
#     y5 = y
#     #left up 6
#     x6 = x - box_x
#     y6 = y + box_y
#     #left down 7
#     x7 = x - box_x
#     y7 = y - box_y
#     #right up
#     x8 = x + box_x
#     y8 = y + box_y
#     #right down
#     x9 = x + box_x
#     y9 = y - box_y
#     
#     x_all = np.concatenate((x1, x2 , x3, x4, x5, x6, x7 ,x8, x9))
#     y_all = np.concatenate((y1, y2 , y3, y4, y5, y6, y7 ,y8, y9))
#     values_all = np.concatenate((value, value , value, value, value, value, value, value, value))
#     return x_all, y_all, values_all
# 
# from scipy.interpolate import griddata
# from scipy.ndimage import gaussian_filter 
# 
# def contours_plots(time_frame, df, order_type, box, leaflet):
#     
#     box_t = box[box['Time'] == time_frame]
#     df_t = df[df['Time'] == time_frame]
#     #df_A_pos_t = df[df_A_positions['Time'] == time_frame]
#     #df_B_pos_t = df_B_positions[df_B_positions['Time'] == time_frame] 
#     #df_coords_t = pd.concat([df_A_pos_t, df_B_pos_t], axis=0)
#         
#     df_A_pos_t_CHL = df_t.loc[(df_t['Resnames'] == 'CHL' )]
#     df_A_pos_t_DLIP = df_t.loc[(df_t['Resnames'] == 'DLIP' )]
#     df_A_pos_t_DSPC = df_t.loc[(df_t['Resnames'] == 'DSPC' )]
#     df_A_pos_t_SSM = df_t.loc[(df_t['Resnames'] == 'SSM' )]
#     
#     df_B_pos_t_CHL = df_t.loc[(df_t['Resnames'] == 'CHL' )]
#     df_B_pos_t_DLIP = df_t.loc[(df_t['Resnames'] == 'DLIP' )]
#     df_B_pos_t_DSPC = df_t.loc[(df_t['Resnames'] == 'DSPC' )]
#     df_B_pos_t_SSM = df_t.loc[(df_t['Resnames'] == 'SSM' )]    
# 
#     
#     if order_type == 'Scd_sn':
#         valuesA = df_t['Scd_sn1']
#         valuesB = df_t['Scd_sn2']
#         #values = pd.concat([valuesA, valuesB], axis=0).dropna().values
#         values = pd.concat([valuesA, valuesB], axis=0).fillna(2).values
#     elif order_type == 'Tilt angles':       
#         angles_in_degree = np.degrees(df_t['Tilt_angles'].values)
#         angles_first_quadrant = np.array([first_quadrant(x) for x in angles_in_degree])
#         values = np.concatenate([ angles_first_quadrant, angles_first_quadrant])
#     elif order_type == 'Splay': 
#         angles_in_degree =  np.degrees (df_t['Splay'].values)
#         values = np.concatenate([angles_in_degree, angles_in_degree])
# 
# 
#     if order_type == 'Scd_sn':  
#         #coords_X = np.concatenate( [ df_A_pos_t_DLIP['CM coord1 X'].values, df_A_pos_t_DSPC['CM coord1 X'].values, df_A_pos_t_SSM['CM coord1 X'].values, df_B_pos_t_DLIP['CM coord2 X'].values, df_B_pos_t_DSPC['CM coord2 X'].values, df_B_pos_t_SSM['CM coord2 X'].values ] )
#         #coords_Y = np.concatenate( [ df_A_pos_t_DLIP['CM coord1 Y'].values, df_A_pos_t_DSPC['CM coord1 Y'].values, df_A_pos_t_SSM['CM coord1 Y'].values, df_B_pos_t_DLIP['CM coord2 Y'].values, df_B_pos_t_DSPC['CM coord2 Y'].values, df_B_pos_t_SSM['CM coord2 Y'].values] )
#         coords_X = np.concatenate( [ df_t['CM coord1 X'].values, df_t['CM coord1 X'].values] )
#         coords_Y = np.concatenate( [ df_t['CM coord1 Y'].values, df_t['CM coord2 Y'].values] )
#         x_pbc, y_pbc, value_pbc = pbc_coordinates(coords_X, coords_Y, (box_t['box_x'].values), (box_t['box_y'].values), values )# angstrom 
#         plt.scatter(x_pbc, y_pbc, c= value_pbc)
#     else:
#         coords_X = np.concatenate( [ df_t['CM coord1 X'].values, df_t['CM coord1 X'].values] )
#         coords_Y = np.concatenate( [ df_t['CM coord1 Y'].values, df_t['CM coord2 Y'].values] )
#         x_pbc, y_pbc, value_pbc = pbc_coordinates(coords_X, coords_Y, (box_t['box_x'].values), (box_t['box_y'].values), values )
#         
#         
#     x_A_CHL_pbc , y_A_CHL_pbc, _  = pbc_coordinates( df_A_pos_t_CHL['CM coord1 X'].values, df_A_pos_t_CHL['CM coord1 Y'].values, (box_t['box_x'].values), (box_t['box_y'].values), [None] )
#     x_A_DLIP_pbc , y_A_DLIP_pbc, _  = pbc_coordinates( df_A_pos_t_DLIP['CM coord1 X'].values, df_A_pos_t_DLIP['CM coord1 Y'].values, (box_t['box_x'].values), (box_t['box_y'].values), [None] )
#     x_A_DSPC_pbc , y_A_DSPC_pbc, _  = pbc_coordinates( df_A_pos_t_DSPC['CM coord1 X'].values, df_A_pos_t_DSPC['CM coord1 Y'].values, (box_t['box_x'].values), (box_t['box_y'].values), [None] )
#     x_A_SSM_pbc , y_A_SSM_pbc, _  = pbc_coordinates( df_A_pos_t_SSM['CM coord1 X'].values, df_A_pos_t_SSM['CM coord1 Y'].values, (box_t['box_x'].values), (box_t['box_y'].values), [None] )
# 
#         
#     x_B_CHL_pbc , y_B_CHL_pbc, _  = pbc_coordinates( df_B_pos_t_CHL['CM coord2 X'].values, df_B_pos_t_CHL['CM coord2 Y'].values, (box_t['box_x'].values), (box_t['box_y'].values), [None] )
#     x_B_DLIP_pbc , y_B_DLIP_pbc, _  = pbc_coordinates( df_B_pos_t_DLIP['CM coord2 X'].values, df_B_pos_t_DLIP['CM coord2 Y'].values, (box_t['box_x'].values), (box_t['box_y'].values), [None] )
#     x_B_DSPC_pbc , y_B_DSPC_pbc, _  = pbc_coordinates( df_B_pos_t_DSPC['CM coord2 X'].values, df_B_pos_t_DSPC['CM coord2 Y'].values, (box_t['box_x'].values), (box_t['box_y'].values), [None] )
#     x_B_SSM_pbc , y_B_SSM_pbc, _  = pbc_coordinates( df_B_pos_t_SSM['CM coord2 X'].values, df_B_pos_t_SSM['CM coord2 Y'].values, (box_t['box_x'].values), (box_t['box_y'].values), [None] )
#     
#     
#     
#     nbox_x = 3000
#     nbox_y = 1000
#     xi = np.linspace(np.min(x_pbc), np.max(x_pbc), nbox_x)
#     yi = np.linspace(np.min(y_pbc), np.max(y_pbc), nbox_y)
#     zi = griddata((x_pbc, y_pbc),value_pbc, (xi[None,:], yi[:,None]), method='linear')
#     zii = gaussian_filter(zi,(6,3))
#     
#     fig, ax = plt.subplots(1, 1, figsize=(5, 3))  
#     #ax.scatter(x_pbc, y_pbc, alpha=0.8)
#     
#     ax.scatter(x_A_CHL_pbc , y_A_CHL_pbc , facecolors='black', edgecolors='black', lw=1, label = 'CHL')
#     ax.scatter(x_A_DSPC_pbc , y_A_DSPC_pbc , facecolors='#696969', edgecolors='#696969', lw=1, label = 'DSPC' )
#     ax.scatter(x_A_SSM_pbc , y_A_SSM_pbc , facecolors='#C0C0C0', edgecolors='#C0C0C0', lw=1, label = 'SSM')
#     ax.scatter(x_B_CHL_pbc , y_B_CHL_pbc , facecolors='black', edgecolors='black', lw=1, label = 'CHL')
#     ax.scatter(x_B_DSPC_pbc , y_B_DSPC_pbc , facecolors='#696969', edgecolors='#696969', lw=1, label = 'DSPC' )
#     ax.scatter(x_B_SSM_pbc , y_B_SSM_pbc , facecolors='#C0C0C0', edgecolors='#C0C0C0', lw=1, label = 'SSM')    
#                
#                #ax.scatter(x_B_POPC_pbc , y_B_POPC_pbc , facecolors='black', edgecolors='black', lw=1)
#     #ax.scatter(x_B_PSM_pbc , y_B_PSM_pbc, facecolors='#696969', edgecolors='#696969', lw=1 )
#     #ax.scatter(x_B_CHL_pbc , y_B_CHL_pbc, facecolors='#C0C0C0', edgecolors='#C0C0C0', lw=1) 
# 
# 
#  
#     #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=3, loc ='upper center') 
# 
#     #hmm_type = 'Hmm_Order'   
# #    if  order_type == 'H' or hmm_type == 'Apl_Hmm' or  hmm_type == 'Thick_Hmm': 
# #        img = ax.imshow(zii,extent=[min(xi), max(xi), min(yi), max(yi) ], origin= 'lower',vmin=0, vmax=2)
# #        cbar = plt.colorbar(img, ticks=[0,1,2], label = str(hmm_type), fraction=0.046, pad=0.04) 
# #        cbar.ax.set_yticklabels(['D', 'I', 'O'])
# #    else:
#     img = ax.imshow(zii, interpolation= 'gaussian', extent=[min(xi), max(xi),min(yi), max(yi)], origin='lower')
#     #img = ax.contourf(xi, yi, zi)
#     plt.colorbar(img, label = str(order_type), fraction=0.046, pad=0.04) 
# 
#     ax.set_xlim(0 , (box_t['box_x'].values) ) 
#     ax.set_ylim(0, (box_t['box_y'].values))  
#     #plt.axis('equal')     
#     ax.axes.get_xaxis().set_ticks([])
#     ax.axes.get_yaxis().set_ticks([]) 
#     #plt.savefig(directory_deuterium + '/'+ str(leaflet) + str(order_type)+'_frame'+str(time_frame)+'_density.png', dpi=300)
#     #plt.close()
# 
# =============================================================================


#contour = np.load('/Volumes/LaCie/BACKUP/anna/PHASES/PORES/DLIPC-DSPC-SSM-CHL/Bilayer_flat/directors/contours/contours_lower.1.npy', allow_pickle=True)
#contours_plots(1, df, 'Tilt angles', box_df, side)


#zii = contours_plots(1, df, 'Splay', box_df, side)




#plt.plot(contour[0][:,0], contour[0][:,1])
#plt.plot(contour[1][:,0], contour[1][:,1])
#contours_plots(1, df, 'Scd_sn', box_df, side)



# =============================================================================
# #TEST duplicate coordinates
# #main
# x = df['CM coord1 X'].values
# y = df['CM coord1 Y'].values
# 
# 
# 
# 
# 
# nbox_x = np.int(3*box[0] +1)
# nbox_y = np.int(3*box[1] +1)
# del_x = 3*box[0]/nbox_x
# del_y = 3*box[1]/nbox_y
# 
# XX, YY = np.meshgrid( del_x*(np.arange(nbox_x) +0.5), del_y*(np.arange(nbox_y) +0.5) )
# 
# xi = np.linspace(min(x_all),max(x_all),nbox_x)
# yi = np.linspace(min(y_all),max(y_all),nbox_y)
# 
# 
# 
# #zi = griddata( (df['CM coord1 X'].values, df['CM coord1 Y'].values), angles_first_quadrant, (XX,YY), method='linear')
# 
# zi = griddata( (x_all, y_all), angles, (xi[None,:], yi[:,None]), method='linear')
# #plt.imshow(zi)
# 
# plt.figure()
# #plt.scatter(x_all + box[0], y_all +box[1], c= angles, edgecolors='red', alpha=0.9)
# zii = gaussian_filter(zi, (5,5))
# #plt.contourf(xi, yi, zi, levels=20)
# plt.scatter(x_all , y_all , c= angles, edgecolors='red', alpha=0.9)
# # This is the fix for the white lines between contour levels
# #for c in ax.collections:
# #    c.set_edgecolor("black")
# plt.imshow(zi, extent= [np.min(xi), np.max(xi), np.min(yi), np.max(yi)], origin='lower', interpolation='none')
# plt.xlim(0, box[0])
# plt.ylim(0, box[1])
# plt.savefig("test-contour.pdf")
# =============================================================================


""" TILT MODULUS 
Tilt_modulus_analysis = False
if Tilt_modulus_analysis == True:

    Kb = 0.0083144621
    T = 298
    #fit the curve for small theta
    from scipy.optimize import curve_fit
    def parabole_fit(x, a, b, x0):
        return  a + (b) * (x-x0)**2.0
    
    def first_quadrant(x):
        if (x >= 90) :
            x= 180 - x
        else:
            x= x
        return x
    
    from sklearn.neighbors.kde import KernelDensity
    
    def _gauss(x, *p):
        A, mu, sigma = p
        return A * npy.exp(-(x - mu)**2 / (2. * sigma**2))
    
    def _FitGaussian(bincenters, pa):
        mu0 = np.sum(bincenters * pa) / np.sum(pa)
        A0 = max(pa)
        sigma0 = np.sqrt(np.sum(((bincenters - mu0)**2.0) * pa) / np.sum(pa))
        (A, mu, sigma), v = curve_fit(_gauss, bincenters, pa, [A0, mu0, sigma0])
        return A, mu, abs(sigma)
    
    
    
    

    w = np.std(np.radians(angles_first_quadrant))
    m = npy.average(splay_list)
    x_range = [m - 3 * w, m + 3 * w]
    pa = npy.histogram(splay_list, nbins, range=x_range, density=True)
    bincenters = 0.5 * (pa[1][1:] + pa[1][:-1])
    fa = -npy.log(pa[0])
    
    
    
    def tilt_modulus(tilt_angle):    
        angles_in_degree = np.degrees(tilt_angle)
        angles_first_quadrant = np.array([first_quadrant(x) for x in angles_in_degree])   
        
        X_plot = np.linspace(0, np.pi/2, len(angles_first_quadrant))[:, np.newaxis]
    
        # instantiate and fit the KDE model
        kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(np.radians(angles_first_quadrant).reshape(-1,1))
        # score_samples returns the log of the probability density
        log_dens = kde.score_samples(X_plot)
        
        x = np.linspace(0, np.pi/2, len(angles_first_quadrant))[:, np.newaxis]
        y=np.sin(x)
        Area=trapz(y[:,0], x[:,0])
        y_normalized=y/Area
    
        Probability = (np.exp(log_dens[1:]))/(y_normalized[1:,0])
        PMF= (-1)*(np.log(Probability))
        
        #TODO better define the range for the fit !!!! This is important!!!
        xdata = x[11:59]
    
        popt, pcov = curve_fit(parabole_fit, xdata.ravel(), PMF[10:58])
        ####################################
        #plot  the probability[ax0] and the PMF[ax 1] and the fit
        fig, ax = plt.subplots(3, 1, sharex=True, sharey=False)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        ax[0].fill_between(X_plot[:, 0], np.exp(log_dens), alpha=0.5)
        ax[0].plot(X_plot[:, 0], np.exp(log_dens),'-')
        ax[1].plot(X_plot[1:, 0], Probability,'-')
        ax[0].plot(X_plot[:, 0], y_normalized,'-', c ='purple')
        ax[2].plot(X_plot[1:, 0], PMF,'-')
        ax[2].plot(x[11:59], parabole_fit(x[11:59], *popt), 'g--', label =r'$\chi_{DSPC}$ = %3.1f' %(popt[1]*2))
        ax[2].grid('True')
        plt.xlim(0,np.pi/2)
        plt.legend()
        plt.show()
        chi_t = popt[1]*2
        return chi_t
    
        
    
    chi_DSPC = tilt_modulus(tilt_angle_DSPC)
    chi_DLIP = tilt_modulus(tilt_angle_DLIP)
    chi_CHL = tilt_modulus(tilt_angle_CHL)

"""


    





# =============================================================================
# from scipy.ndimage import gaussian_filter
# data = np.histogram2d(DLIP_C_o_G_chainB[:,0], DLIP_C_o_G_chainB[:,1], bins=150)[0]
# data2 = np.histogram2d(DSPC_C_o_G_chainA[:,0], DSPC_C_o_G_chainA[:,1], bins=150)[0]
# data = gaussian_filter(data, sigma=5)
# data2 = gaussian_filter(data2, sigma=5)
# plt.pcolormesh(data.T, cmap='inferno', shading='gouraud')
# plt.pcolormesh(data2.T, cmap='inferno', shading='gouraud')
# plt.axis('equal') 
# plt.colorbar() 
# =============================================================================

# =============================================================================
# plt.figure()
# plt.scatter(coord[:,0], coord[:,1], c = directors[:,0])
# plt.axis('equal') 
# plt.colorbar()
# 
# =============================================================================


    
    
    
  
    
    
    

# =============================================================================

# =============================================================================
# 
#     """ test plot """
#     plots = False
#     if  plots == 'True':
#         plt.scatter(DLIP_C_o_G_chainA[:,0], DLIP_C_o_G_chainA[:,1], c = u_DLIP_1[:])
#         plt.scatter(DLIP_C_o_G_chainB[:,0], DLIP_C_o_G_chainB[:,1], c = u_DLIP_2[:])
# 
#         plt.scatter(DSPC_C_o_G_chainA[:,0], DSPC_C_o_G_chainA[:,1], c = u_DSPC_1[:])
#         plt.scatter(DSPC_C_o_G_chainB[:,0], DSPC_C_o_G_chainB[:,1], c = u_DSPC_2[:])
# 
#         plt.scatter(CHL_C_o_G_chainA[:,0], CHL_C_o_G_chainA[:,1], c= u_CHL_1[:])
#         plt.axis('equal')   
#         plt.colorbar()
# 
#         headgroups_coords = headgroups.positions        
#         from mpl_toolkits.mplot3d import Axes3D  
# 
#         plt.scatter(headgroups_coords [:,0],headgroups_coords [:,1], c ='grey')
#         plt.scatter(headgroups_coords [0,0],headgroups_coords [0,1], c = 'black', s= 130)
#  
# 
# 
#         for i in range(0,10):
#             #plt.scatter(headgroups_coords [ sigma[:,i] ==1 ][:,0],headgroups_coords [ sigma[:,i] ==1 ][:,2])
#             plt.scatter(headgroups_coords [ sigma[:,i] ==1 ][:,1],headgroups_coords [ sigma[:,i] ==1 ][:,2])            
#             
#         plt.axis('equal')
#         
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         for i in range(0,10):
#             ax.quiver( headgroups_coords [ sigma[:,i] ==1 ][:,0],headgroups_coords [ sigma[:,i] ==1 ][:,1], headgroups_coords [ sigma[:,i] ==1 ][:,2] , 
#                       normal_leaflet [ sigma[:,i] ==1 ][:,0], normal_leaflet [ sigma[:,i] ==1 ][:,1], 
#                       normal_leaflet [ sigma[:,i] ==1 ][:,2], length=5)
#             ax.scatter(headgroups_coords [:,0],headgroups_coords [:,1], headgroups_coords [:,2], c= 'grey')
#             ax.scatter(headgroups_coords [ sigma[:,i] ==1 ][:,0],headgroups_coords [ sigma[:,i] ==1 ][:,1], 
#                        headgroups_coords [ sigma[:,i] ==1 ][:,2], s= 50)
# 
# =============================================================================




# =============================================================================
# 
# """ Test for normals --> passed"""
# from mpl_toolkits.mplot3d import Axes3D    
# fig = plt.figure()
# 
# u = np.linspace(0,  2*np.pi, 10)
# v = np.linspace(0, -np.pi /2, 10)
# x = 10 * np.outer(np.cos(u), np.sin(v))
# y = 10 * np.outer(np.sin(u), np.sin(v))
# z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z)
# 
# headgroups_coords_t = np.concatenate((x.flatten().reshape(-1,1), y.flatten().reshape(-1,1), z.flatten().reshape(-1,1)), axis= 1 )
# 
# #headgroups_coords_t= np.array([[1,1,5], [2,2,4], [3,5,3], [2,6,2]])    
# cutoff = 10 
# 
# normals_test = np.zeros((len(headgroups_coords_t),3))
# #indices = np.zeros((len(leaflet.resnums),1)) 
# 
# distarr_t = MDAnalysis.lib.distances.distance_array(headgroups_coords_t, headgroups_coords_t) #box=box
# sigma_ij_t =  np.zeros ((len(distarr_t), len(distarr_t)))
# for row, i in enumerate(distarr_t):
#     for col, j in enumerate(distarr_t):
#         if distarr_t[row, col] < cutoff : #first neighbour (check)
#             sigma_ij_t[row, col] = 1
#         else : 
#             sigma_ij_t[row, col] = 0
#             
# for i in range(len(distarr_t)):            
#     coords_for_plane_t = headgroups_coords_t[sigma_ij_t[:,i] ==1]
#     C_t, N_t = standard_fit(coords_for_plane_t)                
#     normals_test[i] = N_t
# #    indices[i] = i
#     
# from mpl_toolkits.mplot3d import Axes3D    
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.quiver(headgroups_coords_t[:,0], headgroups_coords_t[:,1], headgroups_coords_t[:,2],
#           normals_test[:,0], normals_test[:,1], normals_test[:,2]  )
# ax.scatter(headgroups_coords_t[:,0], headgroups_coords_t[:,1], headgroups_coords_t[:,2])
# #=======================================
# =============================================================================


  
#
#def get_angles_CM_lipids(lpd1i, lpd2i,  lpd4i, COM_z):
#    """ retrieve lipids coordinates, resIDs and the director order parameter u=3/2cosˆ2(theta)-1/2 """
#    #COM_z =[0, 0, 57.7] this is wrong
#    
#    lpd4_coords = np.zeros((len(lpd4i.resnums),3))
#    lpd4_res = np.zeros((len(lpd4i.resnums),1))
#    lpd4_u =np.zeros((len(lpd4i.resnums),1))
#    lpd4_angle =np.zeros((len(lpd4i.resnums),1))
#    dist_O2_CM = np.zeros((len(lpd4i.resnums),1))
#    dist_C24_CM = np.zeros((len(lpd4i.resnums),1))
#    dist_C1_CM = np.zeros((len(lpd4i.resnums),1))
#    
#    for i in np.arange(len(lpd4i.resnums)):
#
#        resnum = lpd4i.resnums[i]
#        head_CHL = u.select_atoms('resnum %i and (name C1)'%resnum).positions #center_of_geometry() 
#        tail_CHL = u.select_atoms('resnum %i and (name C65)'%resnum).positions #center_of_geometry() #C24 
#        O2_z = (u.select_atoms('resnum %i and (name O2)'%resnum).positions) 
#        C24_z = (u.select_atoms('resnum %i and (name C24)'%resnum).positions) 
#        
#        
#        
#        
#        theta_CHL = head_tail_angle(head_CHL,tail_CHL)
#
#        u_CHL = order_parameter(theta_CHL)
#        
#        O2_CM = O2_z - COM_z
#        C1_CM = head_CHL - COM_z
#        C24_CM =C24_z - COM_z
#
#        group = u.select_atoms('resnum %i'%resnum)
#        group_cog = group.center_of_geometry()
#        lpd4_coords[i] = group_cog
#        lpd4_res[i] = resnum
#        lpd4_u[i] = u_CHL
#        lpd4_angle[i] = theta_CHL
#        dist_O2_CM[i] = O2_CM [:,2]  
#        dist_C24_CM[i] = C24_CM[:,2]
#        dist_C1_CM[i] = C1_CM [:,2]
#            
## ID coordinates for lipids on indicated bilayer side, renaming variables
## lipid 1= DLIP
## =============================================================================
#    
#    lpd1_start = min(lpd1i.resnums)
#    lpd1_end = max(lpd1i.resnums)
#    
#    #head_lpd1_A = np.zeros((len(lpd1i.resnums),3))
#    #tail_lpd1_A = np.zeros((len(lpd1i.resnums),3))
#    
#    head_lpd1_A = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name C24)').positions
#    tail_lpd1_A = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name 6C21)').positions
#    theta_lp1_A = head_tail_angle(head_lpd1_A , tail_lpd1_A)
#    
#    head_lpd1_B = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name C34)').positions
#    tail_lpd1_B = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name 6C31)').positions
#    theta_lp1_B = head_tail_angle(head_lpd1_B , tail_lpd1_B)
#    
#    #theta and order parameters for lipids DLIPC
#    theta_lp1 = (theta_lp1_A + theta_lp1_B)/2 #this is correct
#    u_lp1 = order_parameter(theta_lp1)
#    
#    chainA_1 = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name C26)').positions
#    chainA_2 = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name C27)').positions
#    chainA_3 = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name C28)').positions
#    chainA_4 = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name C29)').positions
#
#    chainB_1 = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name C36)').positions
#    chainB_2 = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name C37)').positions
#    chainB_3 = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name C38)').positions
#    chainB_4 = u.select_atoms('resnum ' + str(lpd1_start) + '-'+ str(lpd1_end) + ' and (name C39)').positions   
#    
#    #coordinates of the lipids
#    lpd1_C_o_G_chainA = np.average([chainA_1, chainA_2,chainA_3, chainA_4], axis = 0)
#    lpd1_C_o_G_chainB = np.average([chainB_1, chainB_2,chainB_3, chainB_4], axis = 0)  
#    lpd1_C_o_G = np.average([lpd1_C_o_G_chainA, lpd1_C_o_G_chainB], axis = 0)
#    lpd1_res = lpd1i.resnums
#
#    
## lipid 2
## =============================================================================   
#    lpd2_start = min(lpd2i.resnums)
#    lpd2_end = max(lpd2i.resnums)
#    
#    #head_lpd1_A = np.zeros((len(lpd1i.resnums),3))
#    #tail_lpd1_A = np.zeros((len(lpd1i.resnums),3))
#    
#    head_lpd2_A = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name C24)').positions
#    tail_lpd2_A = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name 6C21)').positions
#    theta_lp2_A = head_tail_angle(head_lpd2_A , tail_lpd2_A)
#    
#    head_lpd2_B = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name C34)').positions
#    tail_lpd2_B = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name 6C31)').positions
#    theta_lp2_B = head_tail_angle(head_lpd2_B , tail_lpd2_B)
#    
#    #theta and order parameters for lipids DLIPC
#    theta_lp2 = (theta_lp2_A + theta_lp2_B)/2
#    u_lp2 = order_parameter(theta_lp2)
#    
#    chainA_1 = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name C26)').positions
#    chainA_2 = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name C27)').positions
#    chainA_3 = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name C28)').positions
#    chainA_4 = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name C29)').positions
#
#    chainB_1 = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name C36)').positions
#    chainB_2 = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name C37)').positions
#    chainB_3 = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name C38)').positions
#    chainB_4 = u.select_atoms('resnum ' + str(lpd2_start) + '-'+ str(lpd2_end) + ' and (name C39)').positions   
#    
#    #coordinates of the lipids
#    lpd2_C_o_G_chainA = np.average([chainA_1, chainA_2,chainA_3, chainA_4], axis = 0)
#    lpd2_C_o_G_chainB = np.average([chainB_1, chainB_2,chainB_3, chainB_4], axis = 0)  
#    lpd2_C_o_G = np.average([lpd2_C_o_G_chainA, lpd2_C_o_G_chainB], axis = 0)
#    lpd2_res = lpd2i.resnums
#    
#
#    
#
#    u_1 = u_lp1.reshape(len(u_lp1), 1)
#    u_2 = u_lp2.reshape(len(u_lp2), 1)
#    #u_3 = u_lp3.reshape(len(u_lp3), 1)
#    res1 = lpd1_res.reshape(len(lpd1_res), 1)
#    res2 = lpd2_res.reshape(len(lpd2_res), 1)
# #   res3 = lpd3_res.reshape(len(lpd3_res), 1)
# 
#    lpd_coords = np.vstack((lpd1_C_o_G_chainA, lpd1_C_o_G_chainB, lpd2_C_o_G_chainA, lpd2_C_o_G_chainB, lpd4_coords)) #lpd3_C_o_G_chainA, lpd3_C_o_G_chainB
#    u_all = np.vstack((u_1, u_1, u_2, u_2,lpd4_u)) #u_3, u_3,
#    lpd_res_all =np.vstack((res1,res1, res2,res2, lpd4_res)) #res3, res3,
#    
#    lpd_coords_av = np.vstack((lpd1_C_o_G, lpd2_C_o_G,lpd4_coords))#lpd3_C_o_G,
#    u_all_av = np.vstack((u_1, u_2, lpd4_u)) #u_3,
#    lpd_res_av =np.vstack((res1, res2, lpd4_res)) #res3, 
#    
#
#    
##=========================================================================================
##put stuff together and return
#    
#    return lpd_coords, lpd_res_all, u_all, lpd_coords_av, lpd_res_av, u_all_av , lpd4_res, lpd4_angle, lpd4_coords, dist_O2_CM, dist_C24_CM, dist_C1_CM, res1, res2 #, res3



