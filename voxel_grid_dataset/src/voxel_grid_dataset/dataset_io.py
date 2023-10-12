import numpy
import os
import json


class VoxelDataFoldersManager:
    thereis_instance = False

    def __init__(self):
        self.index = []
        self.data_folders: list(VoxelDataFolder) = None
        if self.__class__.thereis_instance:
            raise Exception(
                f"There can only be one instance of the class {self.__class__}."
            )
        self.__class__.thereis_instance = True
        self.base_folder = os.path.join(
            os.environ["HOME"], ".datasets", ".voxel_datasets"
        )
        if not os.path.isdir(self.base_folder):
            os.makedirs(self.base_folder)
        self.index_file_path = os.path.join(self.base_folder, "index.json")
        self.__read_index()

    def __read_index(self):
        if os.path.isfile(self.index_file_path):
            with open(self.index_file_path, "r") as f:
                self.index = json.loads(f)
        else:
            self.index = []
        self.data_folders = []
        for manager_dict in self.index["datasets"]:
            self.data_folders.append(VoxelDataFolder.from_dict(manager_dict))

    def __write_index(self):
        with open(self.index_file_path, "w") as f:
            json.dumps(self.index, f)

    def __generate_new_id(self):
        return max([dataset["dataset_id"] for dataset in self.index[["datasets"]]])

    def new_datafolder(self):
        self.index
        self.__write_index()

    def delele_datafolder(self):
        self.__write_index()

    def load_dataset(self):
        pass

    def __del__(self):
        self.__class__.thereis_instance = False


class VoxelDataFolder:
    def __init__(
        self,
        manager: VoxelDataFoldersManager,
        dataset_id: int,
        name: str,
        dataset_type: str,
        path_to_env: str,
        n_datapoints: int,
        identifiers: list,
    ):
        self.manager = manager
        self.dataset_id = dataset_id
        self.name = name
        self.dataset_type = dataset_type
        self.path_to_env = path_to_env
        self.n_datapoints = n_datapoints
        self.identifiers = identifiers

    def to_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "dataset_type": self.dataset_type,
            "path_to_env": self.path_to_env,
            "n_datapoints": self.n_datapoints,
            "identifiers": self.identifiers,
        }

    @property
    def folder(self):
        return str(self.dataset_id)

    @property
    def path(self):
        return os.path.join(self.manager.base_folder, self.folder)

    @classmethod
    def from_dict(cls, manager, dict):
        return cls(
            manager,
            dict["dataset_id"],
            dict["name"],
            dict["dataset_type"],
            dict["path_to_env"],
            dict["n_datapoints"],
            dict["identifiers"],
        )


a = VoxelDataFoldersManager()
