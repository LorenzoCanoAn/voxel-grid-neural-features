import numpy
import os 
import json 



class VoxelDatasetsManager:
    def __init__(self):
        base_folder = os.path.join(os.environ["HOME"],"datasets",".voxel_datasets")
        if not os.path.isdir(base_folder):
            os.makedirs(base_folder)
        self.base_folder = base_folder 
        self.index_file_path = os.path.join(self.base_folder, "index.json")
        if not os.path.isfile(self.index_file_path):
            pass
    def new_dataset(self):
        pass
    
class VoxelDatasetFilesManager:
    def __init__(self):
        pass