import os
import json
import shutil
import copy


class DataFoldersManager:
    instances = []

    def __init__(self, datafolder_group="voxel_datasets", is_writer=False):
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
            self.index.append(DataFolder.from_dict(self, manager_dict))

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
        user_decision = input(
            f"All {len(self.index)} datafolders will be removed, and the index eliminated, are you sure? [y/n]: "
        )
        if user_decision.lower() == "y":
            shutil.rmtree(self.base_folder)

    def available_keys(self):
        combined_dict = dict()
        for data_folder_dict in self.index_dict:
            combined_dict = combine_dicts(combined_dict, data_folder_dict)
        return combined_dict


class DataFolder:
    def __init__(
        self,
        manager: DataFoldersManager,
        datafolder_id: int,
        name: str,
        dataset_type: str,
        path_to_env: str,
        n_datapoints: int,
        identifiers: dict,
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

    @property
    def file_names(self):
        file_names = os.listdir(self.path)
        file_names.sort()
        return

    @property
    def file_paths(self):
        return [os.path.join(self.path, file_name) for file_name in self.file_names]

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


def dict_comparison_AllOfSmallerInLarger(larger_dict: dict, smaller_dict: dict):
    result = True
    for key in smaller_dict.keys():
        if key in larger_dict.keys():
            if isinstance(smaller_dict[key], dict):
                result = result and (
                    dict_comparison_AllOfSmallerInLarger(
                        larger_dict[key], smaller_dict[key]
                    )
                )
            else:
                result = result and (smaller_dict[key] == larger_dict[key])
        else:
            result = result and False
    return result


def dict_comparison_AnyOfSmallerInLarger(larger_dict: dict, smaller_dict: dict):
    result = False
    for key in smaller_dict.keys():
        if key in larger_dict.keys():
            if isinstance(smaller_dict[key], dict):
                result = result and (
                    dict_comparison_AnyOfSmallerInLarger(
                        larger_dict[key], smaller_dict[key]
                    )
                )
            else:
                result = result and (smaller_dict[key] == larger_dict[key])
        else:
            result = result and False
        if result:
            return result
    return result


def dict_to_dict_of_sets(dic: dict):
    dict_of_sets = dict()
    for key in dic.keys():
        value = dic[key]
        if isinstance(value, set):
            dict_of_sets[key] = value
        elif isinstance(value, dict):
            dict_of_sets[key] = dict_to_dict_of_sets(value)
        else:
            dict_of_sets[key] = {
                value,
            }
    return dict_of_sets


def combine_dicts(dict_1: dict, dict_2: dict):
    combined_dict = dict_to_dict_of_sets(dict_1)
    for key2 in dict_2.keys():
        value2 = dict_2[key2]
        if key2 in combined_dict:
            if isinstance(value2, dict):
                combined_dict[key2] = combine_dicts(combined_dict[key2], value2)
            else:
                combined_dict[key2].add(value2)
        else:
            if isinstance(value2, dict):
                combined_dict[key2] = dict_to_dict_of_sets(value2)
            else:
                combined_dict[key2] = {
                    value2,
                }
    return combined_dict


class DatasetInputManager:
    """This class handles the file management for reading datasets. At creation, specify the desired characteristics and it will select the
    datafolders that fit the description"""

    def __init__(
        self,
        datafolder_manager: DataFoldersManager,
        wanted_characteristics: dict,
        unwanted_characteristics: dict,
    ):
        self.datafolder_manager = datafolder_manager
        self.wanted_characteristics = wanted_characteristics
        self.unwanted_characteristics = unwanted_characteristics
        self.filter_datafolders()

    def filter_datafolders(self):
        self.selected_datafolders: list(DataFolder) = []
        for data_folder in self.datafolder_manager.index:
            if dict_comparison_AllOfSmallerInLarger(
                data_folder.to_dict, self.wanted_characteristics
            ):
                self.selected_datafolders.append(data_folder)
        idxs_to_remove = []
        for idx, data_folder in enumerate(self.selected_datafolders):
            if dict_comparison_AnyOfSmallerInLarger(
                data_folder.to_dict(), self.unwanted_characteristics
            ):
                idxs_to_remove.append(idx)
        idxs_to_remove.sort(reverse=True)
        for idx in idxs_to_remove:
            self.selected_datafolders.pop(idx)

    @property
    def file_paths(self):
        paths = []
        for data_folder in self.selected_datafolders:
            paths += data_folder.file_paths
        return paths


class DatasetOutputManager:
    """This class is the one that handles the files to write the dataset"""

    def __init__(self, datafolders_manager: DataFoldersManager):
        pass
