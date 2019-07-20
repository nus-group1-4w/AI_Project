# Settings
crop_size = 112
cell_size = 8
bin_size = 8
block_size = 16
feature_length = int(((crop_size/cell_size-1)**2)*((block_size/cell_size)**2)*bin_size)
threshold = 1.2