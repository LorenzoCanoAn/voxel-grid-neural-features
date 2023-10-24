from dataset_management.dataset import DatasetFileManagerToPytorchDataset
import numpy as np
import torch
import math
from tqdm import tqdm

def T_to_xyzrpy(T):
    translation = T[:3, 3]
    x, y, z = translation
    rotation_matrix = T[:3, :3]
    pitch = -math.asin(rotation_matrix[2, 0])
    if math.cos(pitch) != 0:
        yaw = math.atan2(
            rotation_matrix[1, 0] / math.cos(pitch),
            rotation_matrix[0, 0] / math.cos(pitch),
        )
    else:
        yaw = 0
    roll = math.atan2(
        rotation_matrix[2, 1] / math.cos(pitch), rotation_matrix[2, 2] / math.cos(pitch)
    )
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    roll = math.degrees(roll)
    return x, y, z, roll, pitch, yaw


def xyzrpyw_to_T(x, y, z, roll, pitch, yaw):
    translation_matrix = np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
    )
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation_yaw = np.array(
        [
            [cos_yaw, -sin_yaw, 0, 0],
            [sin_yaw, cos_yaw, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    rotation_pitch = np.array(
        [
            [1, 0, 0, 0],
            [0, cos_pitch, -sin_pitch, 0],
            [0, sin_pitch, cos_pitch, 0],
            [0, 0, 0, 1],
        ]
    )
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    rotation_roll = np.array(
        [
            [cos_roll, 0, sin_roll, 0],
            [0, 1, 0, 0],
            [-sin_roll, 0, cos_roll, 0],
            [0, 0, 0, 1],
        ]
    )
    rotation_matrix = np.dot(np.dot(rotation_yaw, rotation_pitch), rotation_roll)
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)
    return transformation_matrix


class VoxelGridDataset(DatasetFileManagerToPytorchDataset):
    required_identifiers = []

    def write_datapoint(self, vg1, vg2, label1, label2):
        path = self.output_manager.get_path_to_new_train_sample()
        with open(path, "wb+") as f:
            np.savez_compressed(f, vg1=vg1, vg2=vg2, label1=label1, label2=label2)

    def read_datapoint(self, path):
        data = np.load(path, allow_pickle=True)
        vg1 = data["vg1"]
        vg2 = data["vg2"]
        label1 = data["label1"]
        label2 = data["label2"]
        return vg1, vg2, label1, label2

    def load_dataset(self):
        print("Loading Dataset")
        voxel_size = self.input_manager.selected_datafolders[0].identifiers[
            "voxel_size"
        ]
        max_x = self.input_manager.selected_datafolders[0].identifiers["max_x"]
        max_y = self.input_manager.selected_datafolders[0].identifiers["max_y"]
        max_z = self.input_manager.selected_datafolders[0].identifiers["max_z"]
        length = int(np.floor(max_x / voxel_size * 2))
        width = int(np.floor(max_y / voxel_size * 2))
        height = int(np.floor(max_z / voxel_size * 2))
        self.vg1_general = torch.zeros(
            size=(self.input_manager.n_datapoints, 1, length, width, height),
            dtype=torch.float32,
        )
        self.vg2_general = torch.zeros(
            size=(self.input_manager.n_datapoints, 1, length, width, height),
            dtype=torch.float32,
        )
        self.poses1 = torch.zeros(size=(self.input_manager.n_datapoints, 6))
        self.poses2 = torch.zeros(size=(self.input_manager.n_datapoints, 6))
        for i in tqdm(range(self.input_manager.n_datapoints)):
            path_to_sample = self.input_manager.file_paths[i]
            vg1, vg2, label1, label2 = self.read_datapoint(path_to_sample)
            self.vg1_general[i, 0, ...] = torch.Tensor(vg1)
            self.vg2_general[i, 0, ...] = torch.Tensor(vg2)
            self.poses1[i] = torch.Tensor(label1)
            self.poses2[i] = torch.Tensor(label2)
        self.gen_all_labels()

    def gen_all_labels(self):
        self.labels = torch.zeros(size=(self.input_manager.n_datapoints * 2, 1))
        for i in range(self.input_manager.n_datapoints):
            self.labels[i, :] = torch.Tensor(
                self.gen_label(self.poses1[i], self.poses2[i])
            )
            self.labels[i + self.input_manager.n_datapoints, :] = torch.Tensor(
                self.gen_label(self.poses2[i], self.poses1[i])
            )

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
        if invert_label:
            return vg2, vg1, self.labels[idx]
        else:
            return vg1, vg2, self.labels[idx]

    def gen_label(self, transform1, transform2):
        x1, y1, z1, r1, p1, yw1 = transform1
        x2, y2, z2, r2, p2, yw2 = transform2
        T1 = xyzrpyw_to_T(x1, y1, z1, r1, p1, yw1)
        T2 = xyzrpyw_to_T(x2, y2, z2, r2, p2, yw2)
        T12 = np.dot(np.linalg.inv(T1), T2)
        x, y, z, roll, pitch, yaw = T_to_xyzrpy(T12)
        dist = np.linalg.norm(np.array([x, y, z]))
        return (dist,)  # roll, pitch, yaw
