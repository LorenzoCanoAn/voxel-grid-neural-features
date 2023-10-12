import os
import json
import shutil

class DataFoldersManager:
    instances = []

    def __init__(self, datafolder_group = "voxel_datasets", is_writer = False):
        self.is_writer = is_writer 
        self.index_dict = []
        self.index: list(DataFolder) = []
        if datafolder_group in self.__class__.instances:
            raise Exception(
                f"There can only be one instance of the class {self.__class__}."
            )
        self.__class__.instances.append(datafolder_group)
        self.base_folder = os.path.join(
            os.environ["HOME"], ".datasets", datafolder_group
        )
        if not os.path.isdir(self.base_folder):
            os.makedirs(self.base_folder)
        self.index_file_path = os.path.join(self.base_folder, "index.json")
        self.__read_index()

    def __read_index(self):
        if os.path.isfile(self.index_file_path):
            with open(self.index_file_path, "r") as f:
                self.index_dict = json.load(f)
        else:
            self.index_dict = []
        self.index = []
        for manager_dict in self.index_dict:
            self.index.append(DataFolder.from_dict(self,manager_dict))

    def __write_index(self):
        self.index_dict = []
        for data_folder in self.index:
            self.index_dict.append(data_folder.to_dict())
        with open(self.index_file_path, "w") as f:
            json.dump(self.index_dict, f)

    def __generate_new_id(self):
        max_id = -1
        for data_folder in self.index:
            max_id = max(max_id, data_folder.datafolder_id)
        return max_id + 1

    def new_datafolder(
        self,
        name: str,
        dataset_type: str,
        path_to_env: str,
        n_datapoints: int,
        identifiers: list,
    ):
        data_folder = DataFolder(
            self,
            self.__generate_new_id(),
            name,
            dataset_type,
            path_to_env,
            n_datapoints,
            identifiers,
        )
        os.mkdir(data_folder.path)
        self.index.append(data_folder)
        self.__write_index()

    def delete_datafolder(self, data_folder):
        self.index.remove(data_folder)
        self.__write_index()

    def get_dataset_file_manager(self):
        pass

    def __del__(self):
        self.__class__.thereis_instance = False

    def __str__(self):
        string = ""
        for data_folder in self.index:
            string += str(data_folder)
        return string

    def reset_all(self):
        user_decision = input(f"All {len(self.index)} datafolders will be removed, and the index eliminated, are you sure? [y/n]: ")
        if user_decision.lower() == "y":
            shutil.rmtree(self.base_folder)


class DataFolder:
    def __init__(
        self,
        manager: DataFoldersManager,
        datafolder_id: int,
        name: str,
        dataset_type: str,
        path_to_env: str,
        n_datapoints: int,
        identifiers: list,
    ):
        self.manager = manager
        self.datafolder_id = datafolder_id
        self.name = name
        self.dataset_type = dataset_type
        self.path_to_env = path_to_env
        self.n_datapoints = n_datapoints
        self.identifiers = identifiers

    def to_dict(self):
        return {
            "dataset_id": self.datafolder_id,
            "name": self.name,
            "dataset_type": self.dataset_type,
            "path_to_env": self.path_to_env,
            "n_datapoints": self.n_datapoints,
            "identifiers": self.identifiers,
        }

    @property
    def folder(self):
        return str(self.datafolder_id)

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

    def __str__(self):
        return str(self.to_dict())

class DatasetIOManager:
    def __init__(self, datafolder_manager: DataFoldersManager,):
        pass