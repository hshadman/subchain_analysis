exec(open("SM_functions_part_only.py").read())

#divide the seq_name list to 100 chunks containing ~280 sequences for PARALLELIZATION
seq_name_chunks = np.array_split(idrome_prop_flucs.seq_name.values, 100)
chunk_no = 0
#THIS script is for first chunk
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