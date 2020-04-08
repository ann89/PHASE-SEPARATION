import MDAnalysis

import numpy as np
import sys

import matplotlib.pyplot as plt

import os
import pandas as pd
import math

input_dir = 'ANALYSIS/directors/'
input_dir_contours = 'ANALYSIS/directors/contours/'
output_dir = 'ANALYSIS/chol_flip_flop/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
    
    
def diff(A,B):
    """subtract elements in lists"""
    #A-B
    x = list(set(A) - set(B))
    #B-A
    y = list(set(B) - set(A))
    return x, y



def mean_sorter(a1, a2):
    mean1 = np.mean(a1)
    mean2 = np.mean(a2)
    if mean1 <= mean2:
        return a1, a2
    else:
        return a2, a1
    
    
    
def transitions(df, state1, state2, tr, ele):
    time = []
    trans =[]
    for i in range(len(df)-1):
        if((df.iloc[i,ele]==state1) & (df.iloc[i+1,ele]==state2)):
            time.append(i)
            trans.append(tr)
    res=np.column_stack((time, trans))
    return(res)  
    
def find_flips(trans, tr1, tr2):
    flipstart=[]
    flipend=[]
    for i in range(len(trans[:,0])-1):
        if ((trans[i,1]==tr1) & (trans[i+1,1]==tr2)):
            flipstart.append(trans[i,0])
            flipend.append(trans[i+1,0])  
    res=np.column_stack((flipstart, flipend))
    return(res)

def find_all_flip_flops (trans, tr1=1, tr2=2, tr3=3, tr4=4):
    flip=find_flips(trans, tr1, tr2)
    flop=find_flips(trans, tr3, tr4)
    stack=np.r_[flip, flop]
    sortvec=stack[:,0].argsort()
    sortstart=stack[:,0][sortvec]
    sortend=stack[:,1][sortvec]
    res=np.column_stack((sortstart, sortend))
    return(res)
    
    
  
""" take the data from input files """    
CHL_resid = []
UP = []
DOWN = []
stack_up_down = []
all_assigned = []
UP_all_assigned = []
DOWN_all_assigned = []

UP_CONTOURS_all = []
DOWN_CONTOURS_all = []

dist_UP = []
dist_Down = []

top  = 'ANALYSIS/recentered_x.gro'
traj = 'ANALYSIS/recentered_x.xtc'
u = MDAnalysis.Universe(top,traj) 
for i in range(0, u.trajectory.n_frames, 1):
    up = np.load(input_dir + 'cholesterol_upper_tail_'+ str(i) + '.npy',  allow_pickle=True)
    up_angle = np.load(input_dir + 'cholesterol_angle_upper_tail_'+ str(i) + '.npy', allow_pickle=True)
    up_H_distance = np.load(input_dir + 'cholesterol_H_distance_upper_tail_'+ str(i) + '.npy', allow_pickle=True)
    up_T_distance = np.load(input_dir + 'cholesterol_T_distance_upper_tail_'+ str(i) + '.npy', allow_pickle=True)
    up_C1_distance = np.load(input_dir + 'cholesterol_C1_distance_upper_tail_' + str(i) + '.npy', allow_pickle=True) 
    up_pos = np.load(input_dir + 'cholesterol_coord_upper_tail_'+ str(i) + '.npy', allow_pickle=True)
    up_numbers = [1]*len(up)
    
    #import contours and order them
    contours_UP =np.load(input_dir_contours +  'contours_upper.'+ str(i)+'.npy', allow_pickle=True)    
    sorted_contours_UP = sorted(contours_UP, key=len, reverse=True)
    contour_sorted_UP = sorted_contours_UP[0:2]
    sorted_again_UP  = mean_sorter(contour_sorted_UP[0], contour_sorted_UP[1])
    
    
    down = np.load(input_dir + 'cholesterol_lower_tail_'+ str(i) + '.npy', allow_pickle=True)
    down_angle = np.load(input_dir + 'cholesterol_angle_lower_tail_'+ str(i) + '.npy', allow_pickle=True)
    down_H_distance = np.load(input_dir + 'cholesterol_H_distance_lower_tail_'+ str(i) + '.npy', allow_pickle=True)
    down_T_distance = np.load(input_dir + 'cholesterol_T_distance_lower_tail_'+ str(i) + '.npy', allow_pickle=True)
    down_C1_distance = np.load(input_dir + 'cholesterol_C1_distance_lower_tail_' + str(i) + '.npy', allow_pickle=True)
    down_pos = np.load(input_dir + 'cholesterol_coord_lower_tail_'+ str(i) + '.npy', allow_pickle=True)    
    down_numbers = [0]*len(down)
    
    #import contours and order them
    contours_DOWN =  np.load(input_dir_contours +  'contours_lower.'+ str(i)+'.npy', allow_pickle=True) 
    sorted_contours_DOWN = sorted(contours_DOWN, key=len, reverse=True)
    contour_sorted_DOWN = sorted_contours_DOWN[0:2]
    sorted_again_DOWN  = mean_sorter(contour_sorted_DOWN[0], contour_sorted_DOWN[1])
    
    #CHL_resid.append((up, down))
    UP_all = np.column_stack((up,up_angle,up_H_distance, up_T_distance, up_C1_distance, up_pos, up_numbers))
    DOWN_all = np.column_stack((down, down_angle, down_H_distance, down_T_distance, down_C1_distance, down_pos, down_numbers))



    stack_up_down = np.vstack((UP_all, DOWN_all))
    all_assigned.append((stack_up_down))
    UP_all_assigned.append((UP_all))
    DOWN_all_assigned.append((DOWN_all))
    
    UP_CONTOURS_all.append((sorted_again_UP))    
    DOWN_CONTOURS_all.append((sorted_again_UP))




flips_distances_UP =[] 
for k in range(len(all_assigned)-1):
  d = list(set(UP_all_assigned[k][:,0]).symmetric_difference(set(UP_all_assigned[k+1][:,0])))
  flips_distances_UP.append(d)

  
flips_distances_DOWN =[] 
for k in range(len(all_assigned)-1):
  d = list(set(DOWN_all_assigned[k][:,0]).symmetric_difference(set(DOWN_all_assigned[k+1][:,0])))
  flips_distances_DOWN.append(d) 
  
flat_list_UP = [item for sublist in flips_distances_UP for item in sublist]
flat_list_unique_UP = list(set(flat_list_UP))


flat_list_DOWN = [item for sublist in flips_distances_DOWN for item in sublist]
flat_list_unique_DOWN = list(set(flat_list_DOWN))

flat_list_unique = np.unique(flat_list_unique_UP + flat_list_unique_DOWN)

df = pd.DataFrame()
for ele in flat_list_unique:
#    print(ele)
    angles = []
    distances_O2 = []
#    distances_T = []
#    distances_H = []
    up_down_up =  []
    dist_contour_left = []
    dist_contour_right = []
##    dist_contourx_left = []
##    dist_contourx_right = []
#    
    for j  in range(0, len(all_assigned)):
        selection  = (all_assigned[j][:,0] == [ele])
        if selection.any() == True:
#            
            angle_selection = (all_assigned[j][selection,1]) [0]
            dist_O2_selection = (all_assigned[j][selection,2]) [0]
#            dist_T_selection = (all_assigned[j][selection,3]) [0]
#            dist_H_selection = (all_assigned[j][selection,4]) [0]
#            
            pos_selection_x = (all_assigned[j][selection,5])[0]
            pos_selection_y = ((all_assigned[j][selection,6])[0]) 
#            
            up_down_selection = ((all_assigned[j][selection,8])[0]) 
#            
            up_cnt_x = UP_CONTOURS_all[j][1][:,0]
            up_cnt_y = UP_CONTOURS_all[j][1][:,1]
##            
            if up_down_selection == 1 :
                dist_1 =  np.sqrt( (pos_selection_x - up_cnt_x)**2 + (pos_selection_y - up_cnt_y)**2 )#  + (pos_selection_y - UP_CONTOURS_all[j][1][:,1])**2 )
                dist_2 =  np.sqrt( (pos_selection_x - UP_CONTOURS_all[j][0][:,0])**2 + (pos_selection_y - UP_CONTOURS_all[j][0][:,1])**2  )
##                
##                dist_xl =  np.sqrt( (pos_selection_x - UP_CONTOURS_all[j][1][:,0])**2)
##                dist_xr =  np.sqrt( (pos_selection_x - UP_CONTOURS_all[j][0][:,0])**2)
            else:
                dist_1 =  np.sqrt( (pos_selection_x - DOWN_CONTOURS_all[j][1][:,0])**2  + (pos_selection_y - DOWN_CONTOURS_all[j][1][:,1])**2 )
                dist_2 =  np.sqrt( (pos_selection_x - DOWN_CONTOURS_all[j][0][:,0])**2  + (pos_selection_y - DOWN_CONTOURS_all[j][0][:,1])**2 )
##                
##                dist_xl =  np.sqrt( (pos_selection_x - DOWN_CONTOURS_all[j][1][:,0])**2  )
##                dist_xr =  np.sqrt( (pos_selection_x - DOWN_CONTOURS_all[j][0][:,0])**2  )
##                
        angles.append(angle_selection)
        distances_O2.append(dist_O2_selection)
#        distances_T.append(dist_T_selection)
#        distances_H.append(dist_H_selection)
        dist_contour_left.append(min(dist_1))
        dist_contour_right.append(min(dist_2))
##        dist_contourx_left.append(min(dist_xl))
##        dist_contourx_right.append(min(dist_xr))
#        
        up_down_up.append(up_down_selection)
#        
    df["angles " + str(int(ele))] = angles
    df["distances_O2 " + str(int(ele))] = distances_O2
    window=1
    df["rolling_distances_O2 "+ str(int(ele))] = df["distances_O2 " + str(int(ele))].rolling(window, center=True).mean() #to smooth eventually
    lim = 12
    lim_angle_up = 60
    lim_angle_down = 120
    foo = df.loc[: , "rolling_distances_O2 "+ str(int(ele))].copy()
    #foo=df["rolling_distances_O2 "+ str(int(ele))].copy()
    foo_angle=np.degrees(df.loc[: , "angles "+ str(int(ele))]).copy()
    result = pd.concat([foo, foo_angle], axis=1)   
 # binning Stefan #& ((foo_angle>=50)&(foo_angle<=150))
    result.loc[((foo>lim) & (foo_angle <lim_angle_up))] = 1
    result.loc[((foo<-lim) & (foo_angle >lim_angle_down))] = 0
    result [(result.loc[:, result.columns[0]] != 0 ) & (result.loc[:, result.columns[0]] != 1)] = 0.5
    

    df["rolling_up_down " + str(int(ele))] = result["rolling_distances_O2 "+ str(int(ele))]

    df["d_Contour_L " + str(int(ele))] = dist_contour_left
    df["d_Contour_R " + str(int(ele))] = dist_contour_right

    df["d_Contour_min " + str(int(ele))] = np.min(np.transpose(np.array([dist_contour_left, dist_contour_right])), axis=1)
    df["up_down_up " + str(int(ele))] = up_down_up
##---------------------------------------------------------------------------
#
#

df_binned = df.loc[:, df.columns.str.contains('rolling_up_down', regex=True)]   
df_dist_contour = df.loc[:, df.columns.str.contains('d_Contour_min', regex=True)]      

df_flips = pd.DataFrame()
att = []
for ele_idx in range(df_binned.shape[1]):
 
    trans1 = transitions(df=df_binned, state1 =0, state2=0.5, tr=1, ele= ele_idx)
    trans2 = transitions(df=df_binned, state1 =0.5, state2=1, tr=2, ele= ele_idx)
    trans3 = transitions(df=df_binned, state1 =1, state2=0.5, tr=3, ele= ele_idx)
    trans4 = transitions(df=df_binned, state1 =0.5, state2=0, tr=4, ele= ele_idx)

    stack=np.r_[trans1, trans2, trans3, trans4]
    sortvec=stack[:,0].argsort()
    sorttime=stack[:,0][sortvec]
    sorttrans=stack[:,1][sortvec]
    sorted_transitions=np.column_stack((sorttime, sorttrans))

    all_flip_flops =  find_all_flip_flops(sorted_transitions)
    number= [(len(all_flip_flops))]
    attributes = []
    if number[0] !=0 :
        for i in range(len(all_flip_flops)):
            start = all_flip_flops[i][0]
            stop = all_flip_flops[i][1]
            time_i = stop - start            
            type_tr = (sorted_transitions[sorted_transitions[:,0]==start, 1])[0]
            idx = np.unique(ele_idx)
            dist_contour_start = (df_dist_contour.iloc[int(start), idx])[0]
            dist_contour_end = (df_dist_contour.iloc[int(stop), idx])[0]
            attributes.extend((start, stop, time_i, type_tr, dist_contour_start,dist_contour_end))        
            res_list = [y for x in [number, attributes] for y in x]
        att.append(res_list)
    else:
        att.append([0])

df_flips = pd.DataFrame(att).set_index(flat_list_unique.astype(int))
Total_flips = df_flips[0].sum()
np.arange(0, np.max(df_flips[0]), 1)
string = []
for i in range((np.max(df_flips[0]))):
        a = ["Start " + str(i), "Stop " +str(i), "Duration " + str(i), "Type " + str(i), "Dist_Contour start " + str(i), "Dist_Contour stop " + str(i)]
        string.append(a)

#==================================================================================================
# to check distance from contour
     
flattened_string = [val for  sublist in string for val in sublist]     
df_flips.columns =["Number of flips",] +   flattened_string

df_dists_at_start = (df_flips.loc[:, df_flips.columns.str.contains('Dist_Contour start ', regex=True)])
L2 = df_dists_at_start.loc[:, df_dists_at_start.columns.str.contains('Dist_Contour start', regex=True)].astype(float).values.tolist()
a = np.vstack(L2)
x = a[np.logical_not(np.isnan(a))]
plt.hist(x)
plt.xlabel('Distance from contours [A]')
plt.ylabel('Counts')
plt.savefig(output_dir + 'histogram-distances-to-contour.png', dpi=300) 
##
df_flips.to_excel(output_dir + "flips.xlsx")
df.to_excel(output_dir + "chol_check.xlsx")
#==================================================================================================

# plotting
plots = True
if plots == True :
    for ele in flat_list_unique:
    
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Time (ns)', fontsize=14)
        ax1.set_ylabel('angle', color=color, fontsize=14)
        ax1.plot((df["angles " + str(int(ele))]), color=color)
        ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
        ax1.tick_params(axis='x',  labelsize=14)
        ax1.set_ylim([0, math.pi])
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('distance O2', color=color, fontsize=14)  # we already handled the x-label with ax1
        ax2.plot(df["distances_O2 " + str(int(ele))], color=color)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=14)
        ax2.set_ylim([-30,30])
        plt.title('Cholesterol ' + str(int(ele)), size= 14)
        
        number_of_flips= df_flips.loc[int(ele), "Number of flips"]
        for i in range(number_of_flips):
            plt.axvline(df_flips.loc[int(ele), "Start " + str(int(i))], c ='#A9A9A9', linestyle=':')
            plt.axvline(df_flips.loc[int(ele), "Stop " + str(int(i))], c ='#696969', linestyle=':')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig( output_dir + 'Flips_cholesterol_angle_O2dist' + str(int(ele)) + '.png', dpi=300)
        plt.close()
    
