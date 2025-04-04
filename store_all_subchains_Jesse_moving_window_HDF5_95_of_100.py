import pandas as pd
import mdtraj as md
import numpy as np
from numpy.random import seed
from numpy.random import shuffle
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
#from __future__ import print_function
import seaborn as sns
from matplotlib.ticker import NullFormatter, MaxNLocator
from pandas.plotting import scatter_matrix
import matplotlib.ticker as ticker
#import plotly.graph_objects as go
import scipy as sp
from itertools import chain, combinations
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy import spatial
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
import scipy.stats as stats
#import statsmodels.stats.weightstats
from matplotlib import path
import matplotlib
from scipy.stats import probplot,shapiro, sem
#import statsmodels.api as sm
from scipy.interpolate import make_interp_spline
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score, mean_squared_error
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.linear_model import RidgeCV
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay,classification_report
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler

from matplotlib import cm
from numpy import linspace
#import umap.umap_ as umap
#import pylab
import os
#import PIL
#from scipy.ndimage import gaussian_filter, uniform_filter1d

# import soursop
import afrc
import soursop
import MDAnalysis
import pandas as pd
import numpy as np
from soursop.sstrajectory import SSTrajectory
from Bio.PDB import *
from itertools import chain
import h5py
import multiprocessing as mp
#initialization
seq_name_AFRC = pd.read_csv('../holehouse_project/IDRome_shape_mean_size_mean_added.csv')
seq_name_dir_df = pd.read_csv('../larsen_paper_2023_compaction/seq_name_dir_df.csv')
seq_name_fluctations = pd.read_csv('../larsen_paper_2023_compaction/HPC_computed_fC_values_all.csv').set_index('seq_name_list')
seq_stdev = pd.read_csv('../holehouse_project/IDRome_Rg_Rs_RSA_stdev.csv').set_index('seq_name')
seq_ALBATROSS = pd.read_csv('../holehouse_project/IDRome_with_ALBATROSS_calculations.csv').set_index('seq_name')
seq_ranges = pd.read_csv('../holehouse_project/IDRome_Rg_Rs_RSA_range.csv').set_index('seq_name')
# the bounded_frac_size_shape is only for size-shape (through pyconformap_modified)


#add fP to the property df
seq_name_AFRC['fP'] = [seq.count('P')/len(seq) for seq in seq_name_AFRC.fasta.values]

#recalculate FCR
seq_name_AFRC['net_charge'] = [(seq.count('K')+seq.count('R')-seq.count('D')-seq.count('E'))/len(seq) for seq in seq_name_AFRC.fasta.values]

idrome_prop_flucs = pd.concat([seq_name_AFRC.set_index('seq_name'),
           seq_name_fluctations[['fC_shape_shape',
                                 'fA_shape_shape',
                                 'fC_size_shape',
                                 'fA_size_shape',
                                 'bounded_frac_size_shape']],
                              seq_stdev[['Rg_std','Rs_std','RSA_std']],
                              seq_ALBATROSS[['albatross_Rg','albatross_Rg_scaled','albatross_Ree','albatross_Ree_scaled',
                                             'albatross_scaling_exponent']],
                              seq_ranges[['Rg_range','Rs_range','RSA_range']]],
          axis=1).reset_index().rename(columns={'index':'seq_name'}).copy()

del seq_name_fluctations, seq_name_AFRC, seq_stdev, seq_ALBATROSS

#drop these FOUR IDR simulations because of low snapshots #
seq_name_dir_df = seq_name_dir_df[~seq_name_dir_df.seq_name.isin(['Q53SF7_218_1128',
                                             'Q7Z2Y5_341_1224',
                                             'Q9Y2W1_1_611',
                                             'Q9BXT5_1_968'])].copy()

idrome_prop_flucs = idrome_prop_flucs[~idrome_prop_flucs.seq_name.isin(['Q53SF7_218_1128',
                                             'Q7Z2Y5_341_1224',
                                             'Q9Y2W1_1_611',
                                             'Q9BXT5_1_968'])].copy()
#helper functions
#Rij is from lindorff-larsen
def Rij(traj):
    pairs = traj.top.select_pairs('all','all')
    d = md.compute_distances(traj,pairs)
    nres = traj.n_atoms
    ij = np.arange(2,nres,1)
    diff = [x[1]-x[0] for x in pairs]
    dij = np.empty(0)
    for i in ij:
        dij = np.append(dij,np.sqrt((d[:,diff==i]**2).mean().mean()))
    f = lambda x,R0,v : R0*np.power(x,v)
    popt, pcov = curve_fit(f,ij[ij>5],dij[ij>5],p0=[.4,.5])
    nu = popt[1]
    nu_err = pcov[1,1]**0.5
    R0 = popt[0]
    R0_err = pcov[0,0]**0.5
    return ij,dij,nu,nu_err,R0,R0_err

def calculate_nu_KLL_from_seq_name(seq_name, start_residue, end_residue):
    example_protein_dir = seq_name_dir_df[seq_name_dir_df.seq_name == seq_name].seq_dir.values[0]
    t_md = md.load(f'../larsen_paper_2023_compaction/{example_protein_dir}/traj.xtc', 
                   top=f'../larsen_paper_2023_compaction/{example_protein_dir}/top.pdb')
    # Skipping the first 10 frames
    t_md = t_md[10:]
    subsequence_indices = t_md.topology.select(f'residue {start_residue-1} to {end_residue-1}')
    
    subsequence_traj = t_md.atom_slice(subsequence_indices)

    ij, dij, nu, nu_err, R0, R0_err = Rij(subsequence_traj)
    return nu, nu_err

#main function

def compute_30mer_data_from_seq_name(seq_name,k_frac, h5_file_path):
    print('using 30 as moving window size, constant, based on protein of smallest length')
    print('for proteins of length more than 60, moving window size = 30')
    print('for proteins of length less than or equal to 60, moving window size = 1/k_frac of length')
    example_protein_dir = seq_name_dir_df[seq_name_dir_df.seq_name==seq_name].seq_dir.values[0]
    
    traj = md.load(f'../larsen_paper_2023_compaction/{example_protein_dir}/traj.xtc',
                   top=f'../larsen_paper_2023_compaction/{example_protein_dir}/top.pdb')
    
    fasta_sequence = idrome_prop_flucs[idrome_prop_flucs.seq_name==seq_name].fasta.values[0]
    
    # Skipping the first 10 frames
    traj = traj[10:]    
    
    # Set the subsequence length k (replace 4 with your desired value)
    n_residues = traj.topology.n_residues

    if len(fasta_sequence) <= 60:
        k = round(traj.topology.n_residues/k_frac)
    elif len(fasta_sequence) > 60:
        k = 30

    with h5py.File(h5_file_path, 'a') as h5file:
        protein_group = h5file.create_group(seq_name)

        # Compute various properties for the whole protein
        complete_protein_rgyr = np.mean(md.compute_rg(traj))
        protein_group.create_dataset('full_protein_rgyr', data=complete_protein_rgyr)

        # store fasta sequence
        protein_group.create_dataset('fasta_sequence_whole_protein', data=fasta_sequence)

        # store seq_name
        protein_group.create_dataset('seq_name', data=seq_name)
        
        first_bead_index = traj.topology.select(f"residue {0}")[0]
        last_bead_index = traj.topology.select(f"residue {n_residues - 1}")[0]    
        end_to_end_distances = md.compute_distances(traj, [[first_bead_index, last_bead_index]])
        end_to_end_distances = end_to_end_distances.flatten()
        complete_protein_ete = end_to_end_distances
        complete_protein_inst_ratio = np.mean((complete_protein_ete**2)/(complete_protein_rgyr**2))
        protein_group.create_dataset('full_protein_ratio', data=complete_protein_inst_ratio)
    
        complete_protein_moments = pd.DataFrame(md.principal_moments(traj),columns=['R3','R2','R1']).copy()
        complete_protein_moments['asphericity']=complete_protein_moments.R1.values-(0.5*(complete_protein_moments.R2.values+complete_protein_moments.R3.values))
        complete_protein_moments['acylindricity']=complete_protein_moments.R2.values-complete_protein_moments.R3.values
        complete_protein_moments['RSA']=(complete_protein_moments.asphericity.values**2+(0.75*complete_protein_moments.acylindricity.values**2))/(complete_protein_moments.R1.values+complete_protein_moments.R2.values+complete_protein_moments.R3.values)**2
        complete_protein_RSA = np.mean(complete_protein_moments['RSA'].values)    
        protein_group.create_dataset('full_protein_RSA', data=complete_protein_RSA)

        #AFRC
        complete_protein_afrc_init = afrc.AnalyticalFRC(fasta_sequence)
        complete_protein_rg_theta_mean = complete_protein_afrc_init.get_mean_radius_of_gyration()
        complete_protein_rg_rg_theta_mean = np.mean((10*md.compute_rg(traj))/complete_protein_rg_theta_mean)
        protein_group.create_dataset('full_protein_rg_rg_theta_mean', data=complete_protein_rg_rg_theta_mean)

        #nu
        complete_protein_nu = calculate_nu_KLL_from_seq_name(seq_name, 1, n_residues)[0]
        complete_protein_nu_err = calculate_nu_KLL_from_seq_name(seq_name, 1, n_residues)[1]
        protein_group.create_dataset('full_protein_nu_recompute', data=complete_protein_nu)
        protein_group.create_dataset('full_protein_nu_recompute_err', data=complete_protein_nu_err)
    
        j = 0
        seq_name_list = []
        k_list = []
        mid_residue = []
        subchain_mean_Rg = []
        subchain_mean_Rs = []
        subchain_mean_RSA = []
        subchain_mean_rg_rg_theta = []
        subchain_nu = []
        
        # Iterate through each subsequence of k residues
        for start_res in range(1, n_residues - k + 2):  # Ensures we don't go out of bounds
            seq_name_list.append(seq_name)
            k_list.append(k)
            # Select k consecutive residues
            subsequence_group = protein_group.create_group(f'mer_{start_res}_{start_res + k - 1}')
            selection_string = f"residue {start_res-1} to {start_res + k - 2}"  # MDTraj is zero-indexed

            mid_residue.append(((start_res-1) + (start_res + k - 2) ) //2 )

            
            subsequence_indices = traj.topology.select(selection_string)
            
            fasta_slice = fasta_sequence[(start_res-1):(start_res + k - 1)]
            subsequence_group.create_dataset('fasta_subsequence', data=fasta_slice)
            subsequence_group.create_dataset('selection_string', data=selection_string)
            
            
            # Create a trajectory slice for the selected subsequence
            subsequence_traj = traj.atom_slice(subsequence_indices)
            
            # Calculate the radius of gyration for the subsequence over all remaining frames
            rgyr = md.compute_rg(subsequence_traj)
            subsequence_group.create_dataset('Rg/nm', data=rgyr)
            subchain_mean_Rg.append(np.mean(rgyr))
            
            #calculate nu
            subsequence_nu = calculate_nu_KLL_from_seq_name(seq_name, start_res, start_res + k - 1)[0]
            subsequence_nu_err = calculate_nu_KLL_from_seq_name(seq_name, start_res, start_res + k - 1)[1]
            subsequence_group.create_dataset('nu_recompute', data=subsequence_nu)
            subsequence_group.create_dataset('nu_recompute_err', data=subsequence_nu_err)
            subchain_nu.append(subsequence_nu)
            
            # Select indices for the first and last bead in the subsequence for end-to-end distance calculation
            first_bead_index = traj.topology.select(f"residue {start_res-1}")[0]
            last_bead_index = traj.topology.select(f"residue {start_res + k - 2}")[0]
        
            # Calculate end-to-end distances for the subsequence over all remaining frames
            end_to_end_distances = md.compute_distances(traj, [[first_bead_index, last_bead_index]])
            end_to_end_distances = end_to_end_distances.flatten()
            subsequence_group.create_dataset('ete', data=end_to_end_distances)
            
            running_inst_ratio = (end_to_end_distances**2)/(rgyr**2)
            subsequence_group.create_dataset('inst_ratio', data=running_inst_ratio)
            subchain_mean_Rs.append(np.mean(running_inst_ratio))
            
            t_df_moments = pd.DataFrame(md.principal_moments(subsequence_traj),columns=['R3','R2','R1']).copy()
            t_df_moments['asphericity']=t_df_moments.R1.values-(0.5*(t_df_moments.R2.values+t_df_moments.R3.values))
            t_df_moments['acylindricity']=t_df_moments.R2.values-t_df_moments.R3.values
            t_df_moments['RSA']=(t_df_moments.asphericity.values**2+(0.75*t_df_moments.acylindricity.values**2))/(t_df_moments.R1.values+t_df_moments.R2.values+t_df_moments.R3.values)**2
            subsequence_group.create_dataset('RSA', data=t_df_moments['RSA'].values)
            subchain_mean_RSA.append(np.mean(t_df_moments['RSA'].values))
    
            #AFRC
            afrc_init = afrc.AnalyticalFRC(fasta_slice)
            rg_theta_mean = afrc_init.get_mean_radius_of_gyration()
            rg_rg_theta_mean = (10*rgyr) / rg_theta_mean
            subsequence_group.create_dataset('rg_rg_theta_mean', data=rg_rg_theta_mean)
            subchain_mean_rg_rg_theta.append(np.mean(rg_rg_theta_mean))
        global combined_subchain_metrics
        combined_subchain_metrics = pd.DataFrame(zip(seq_name_list,k_list,mid_residue,subchain_mean_Rg,subchain_mean_Rs,subchain_mean_RSA,
                                                subchain_mean_rg_rg_theta,subchain_nu),
                                             columns = ['seq_name','window_size','mid_residue','mean_Rg','mean_Rs','mean_RSA',
                                                'mean_rg_rg_theta','nu'])
        #Rg metrics
        min_mean_Rg = [combined_subchain_metrics.mean_Rg.min()]
        mid_res_min_mean_Rg = [combined_subchain_metrics[combined_subchain_metrics.mean_Rg == combined_subchain_metrics.mean_Rg.min()].mid_residue.values[0]]
        max_mean_Rg = [combined_subchain_metrics.mean_Rg.max()]
        mid_res_max_mean_Rg = [combined_subchain_metrics[combined_subchain_metrics.mean_Rg == combined_subchain_metrics.mean_Rg.max()].mid_residue.values[0]]
        stdev_mean_Rg = [combined_subchain_metrics.mean_Rg.std()]

        #Rs metrics
        min_mean_Rs = [combined_subchain_metrics.mean_Rs.min()]
        mid_res_min_mean_Rs = [combined_subchain_metrics[combined_subchain_metrics.mean_Rs == combined_subchain_metrics.mean_Rs.min()].mid_residue.values[0]]
        max_mean_Rs = [combined_subchain_metrics.mean_Rs.max()]
        mid_res_max_mean_Rs = [combined_subchain_metrics[combined_subchain_metrics.mean_Rs == combined_subchain_metrics.mean_Rs.max()].mid_residue.values[0]]
        stdev_mean_Rs = [combined_subchain_metrics.mean_Rs.std()]
        
        #RSA metrics
        min_mean_RSA = [combined_subchain_metrics.mean_RSA.min()]
        mid_res_min_mean_RSA = [combined_subchain_metrics[combined_subchain_metrics.mean_RSA == combined_subchain_metrics.mean_RSA.min()].mid_residue.values[0]]
        max_mean_RSA = [combined_subchain_metrics.mean_RSA.max()]
        mid_res_max_mean_RSA = [combined_subchain_metrics[combined_subchain_metrics.mean_RSA == combined_subchain_metrics.mean_RSA.max()].mid_residue.values[0]]
        stdev_mean_RSA = [combined_subchain_metrics.mean_RSA.std()]
        
        #rg_rg_theta metrics
        min_mean_rg_rg_theta = [combined_subchain_metrics.mean_rg_rg_theta.min()]
        mid_res_min_mean_rg_rg_theta = [combined_subchain_metrics[combined_subchain_metrics.mean_rg_rg_theta == combined_subchain_metrics.mean_rg_rg_theta.min()].mid_residue.values[0]]
        max_mean_rg_rg_theta = [combined_subchain_metrics.mean_rg_rg_theta.max()]
        mid_res_max_mean_rg_rg_theta = [combined_subchain_metrics[combined_subchain_metrics.mean_rg_rg_theta == combined_subchain_metrics.mean_rg_rg_theta.max()].mid_residue.values[0]]
        stdev_mean_rg_rg_theta = [combined_subchain_metrics.mean_rg_rg_theta.std()]
        
        #nu metrics
        min_mean_nu = [combined_subchain_metrics.nu.min()]
        mid_res_min_mean_nu = [combined_subchain_metrics[combined_subchain_metrics.nu == combined_subchain_metrics.nu.min()].mid_residue.values[0]]
        max_mean_nu = [combined_subchain_metrics.nu.max()]
        mid_res_max_mean_nu = [combined_subchain_metrics[combined_subchain_metrics.nu == combined_subchain_metrics.nu.max()].mid_residue.values[0]]
        stdev_mean_nu = [combined_subchain_metrics.nu.std()]
        
        #generate dataframe of chosen subchain metrics
        selected_subchain_metrics = pd.DataFrame(zip([combined_subchain_metrics.seq_name.unique()[0]],[combined_subchain_metrics.window_size.unique()[0]],
            min_mean_Rg,mid_res_min_mean_Rg,max_mean_Rg,mid_res_max_mean_Rg,stdev_mean_Rg,
                                                    min_mean_Rs,mid_res_min_mean_Rs,max_mean_Rs,mid_res_max_mean_Rs,stdev_mean_Rs,
                                                    min_mean_RSA,mid_res_min_mean_RSA,max_mean_RSA,mid_res_max_mean_RSA,stdev_mean_RSA,
                        min_mean_rg_rg_theta,mid_res_min_mean_rg_rg_theta,max_mean_rg_rg_theta,mid_res_max_mean_rg_rg_theta,stdev_mean_rg_rg_theta,
                                            min_mean_nu,mid_res_min_mean_nu,max_mean_nu,mid_res_max_mean_nu,stdev_mean_nu),
                                                 columns=['seq_name','window_size',
                                                 'min_mean_Rg','mid_res_min_mean_Rg','max_mean_Rg','mid_res_max_mean_Rg','stdev_mean_Rg',
                                                         'min_mean_Rs','mid_res_min_mean_Rs','max_mean_Rs','mid_res_max_mean_Rs','stdev_mean_Rs',
                                                'min_mean_RSA','mid_res_min_mean_RSA','max_mean_RSA','mid_res_max_mean_RSA','stdev_mean_RSA',
            'min_mean_rg_rg_theta','mid_res_min_mean_rg_rg_theta','max_mean_rg_rg_theta','mid_res_max_mean_rg_rg_theta','stdev_mean_rg_rg_theta',
                                            'min_mean_nu','mid_res_min_mean_nu','max_mean_nu','mid_res_max_mean_nu','stdev_mean_nu'])
        print('ALL DONE')    
    return selected_subchain_metrics

#divide the seq_name list to 100 chunks containing ~280 sequences for PARALLELIZATION
seq_name_chunks = np.array_split(idrome_prop_flucs.seq_name.values, 100)
chunk_no = 94
#THIS script is for chunk # 95
count = 0
for seq_n in seq_name_chunks[chunk_no]:
    if count == 0:
       master_df = compute_30mer_data_from_seq_name(seq_n,3, f'./subchain_repository_hdf5/{seq_n}.h5').copy()
    elif count > 0:
       running_df = compute_30mer_data_from_seq_name(seq_n,3, f'./subchain_repository_hdf5/{seq_n}.h5')
       master_df = pd.concat([master_df,running_df]).copy()  
    count+=1
    
master_df.to_csv(f'./subchain_repository_hdf5/chosen_subchain_metrics_chunk_no_{chunk_no}.csv')
print('done with chunk')
print('make sure to do some testing to verify order and accuracy of metrics')