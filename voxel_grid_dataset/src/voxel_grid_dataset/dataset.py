from voxel_grid_dataset.dataset_io import DatasetInputManager, DatasetOutputManager
import numpy as np
from torch.utils.data import Dataset
import torch


class TwoVoxelGridsTwoTransforms(Dataset):
    def __init__(
        self,
        name,
        mode="read",
        identifiers=dict(),
        unwanted_characteristics=dict(),
        load_by_name=True,
    ):
        self.name = name
        self.mode = mode
        if self.mode == "read":
            wanted_characteristics = {
                "dataset_type": self.dataset_type,
                "identifiers": identifiers,
            }
            if load_by_name:
                wanted_characteristics["name"] = self.name
            self.input_manager = DatasetInputManager(
                wanted_characteristics, unwanted_characteristics
            )
        elif self.mode == "write":
            assert "voxelgrid_width" in identifiers.keys()
            assert "voxelgrid_length" in identifiers.keys()
            assert "voxelgrid_height" in identifiers.keys()
            assert "max_x" in identifiers.keys()
            assert "max_y" in identifiers.keys()
            assert "max_z" in identifiers.keys()
            self.output_manager = DatasetOutputManager(
                self.name, self.dataset_type, identifiers
            )

    def write_datapoint(self, vg1, vg2, label1, label2):
        path = self.output_manager.get_path_to_new_train_sample()
        with open(path, "wb+") as f:
            np.savez_compressed(f, vg1=vg1, vg2=vg2, label1=label1, label2=label2)

    def read_datapoint(self, path):
        data = np.load(path)
        vg1 = data["vg1"]
        vg2 = data["vg2"]
        label1 = data["labe1"]
        label2 = data["labe2"]
        return vg1, vg2, label1, label2

    def load_dataset(self):
        self.vg1_general = torch.zeros(size=(self.input_manager.n_datapoints,))
        self.vg2_general = torch.zeros(
            self.input_manager.n_datapoints,
        )
        self.label1 = torch.zeros(self.input_manager.n_datapoints)
        self.label2 = torch.zeros(self.input_manager.n_datapoints)

    def new_env(self, path_to_env):
        if self.mode != "write":
            raise Exception("This method should only be called in write mode")
        self.output_manager.new_datafolder(path_to_env)

    def __len__(self):
        return self.input_manager.n_datapoints * 2

    def __getitem__(self, idx):
        if idx >= self.input_manager.n_datapoints:
            idm = idx - self.input_manager.n_datapoints
            invert_label = True
        else:
            idm = idx
            invert_label = False
        vg1 = self.vg1_general[idm]
        vg2 = self.vg2_general[idm]
        label = self.label_general[idm]
        if invert_label:
            return vg2, vg1, self.invert_label(label)
        else:
            return vg1, vg2, label
