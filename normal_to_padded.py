import numpy as np

old_arr = np.load('kmap_all_input_data_new_x_terms.npy')

new_arr = np.zeros((old_arr.shape[0],5,5), dtype = 'uint8')

new_arr[:,0:4, 0:4] = old_arr

new_arr[:,0:4,4] = old_arr[:,0:4,0]

new_arr[:,4,0:4] = new_arr[:,0,0:4]

np.save('padded_kmap_all_input_data_new_x_terms.npy', new_arr)