import os
import json
import shutil
import time


class DataFoldersManager:
    instances = []

    @classmethod
    def get_current_instance(cls):
        if len(cls.instances) == 0:
            return cls()
        else:
            return cls.instances[0]

    def __init__(self, datafolder_group="voxel_datasets"):
        self.datafolder_group = datafolder_group
        self.index_dict = []
        self.index: list[DataFolder] = []
        if len(self.__class__.instances) == 1:
            raise Exception(
                f"There can only be one instance of the class {self.__class__}."
            )
        else:
            self.__class__.instances.append(self)
        self.base_folder = os.path.join(
            os.environ["HOME"], ".datasets", datafolder_group
        )
        self.block_file_path = os.path.join(self.base_folder, "block")
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
        while os.path.isfile(self.block_file_path):
            time.sleep(0.1)
        with open(self.block_file_path, "w+") as f:
            f.write("a")
        self.index_dict = []
        for data_folder in self.index:
            self.index_dict.append(data_folder.to_dict())
        with open(self.index_file_path, "w") as f:
            json.dump(self.index_dict, f)
        os.remove(self.block_file_path)

    def __generate_new_id(self):
        max_id = -1
        for data_folder in self.index:
            max_id = max(max_id, data_folder.datafolder_id)
        return max_id + 1

    def new_datafolder(
        self,
        dataset_name: str,
        dataset_type: str,
        path_to_env: str,
        identifiers: list,
    ):
        data_folder = DataFolder(
            self,
            self.__generate_new_id(),
            dataset_name,
            dataset_type,
            path_to_env,
            identifiers,
        )
        os.mkdir(data_folder.path)
        self.index.append(data_folder)
        self.__write_index()
        return data_folder

    def delete_datafolder(self, data_folder):
        assert isinstance(data_folder, DataFolder)
        shutil.rmtree(data_folder.path)
        self.index.remove(data_folder)
        self.__write_index()
        self.__read_index()

    def delete_datafolders(self, data_folders):
        assert isinstance(data_folders, list)
        for data_folder in data_folders:
            self.delete_datafolder(data_folder)

    def __del__(self):
        self.__class__.instances.remove(self)

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

    def filter_datafolders(
        self,
        wanted_characteristics: dict = dict(),
        unwanted_characteristics: dict = dict(),
    ):
        selected_datafolders: list[DataFolder] = []
        for data_folder in self.index:
            if dict_comparison_AllOfSmallerInLarger(
                data_folder.to_dict(), wanted_characteristics
            ):
                selected_datafolders.append(data_folder)
        idxs_to_remove = []
        for idx, data_folder in enumerate(selected_datafolders):
            if dict_comparison_AnyOfSmallerInLarger(
                data_folder.to_dict(), unwanted_characteristics
            ):
                idxs_to_remove.append(idx)
        idxs_to_remove.sort(reverse=True)
        for idx in idxs_to_remove:
            selected_datafolders.pop(idx)
        return selected_datafolders

    def gen_new_name(self, name):
        changed_name = name
        current_names = [datafolder.dataset_name for datafolder in self.index]
        counter = 1
        while True:
            if changed_name in current_names:
                changed_name = name + f"_{counter}"
                counter += 1
            else:
                return changed_name


class DataFolder:
    def __init__(
        self,
        manager: DataFoldersManager,
        datafolder_id: int,
        dataset_name: str,
        dataset_type: str,
        path_to_env: str,
        identifiers: dict,
    ):
        self.manager = manager
        self.datafolder_id = datafolder_id
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.path_to_env = path_to_env
        self.identifiers = identifiers

    def to_dict(self):
        return {
            "datafolder_id": self.datafolder_id,
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type,
            "path_to_env": self.path_to_env,
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
            dict["datafolder_id"],
            dict["dataset_name"],
            dict["dataset_type"],
            dict["path_to_env"],
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
        if not result:
            return result
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
        wanted_characteristics: dict,
        unwanted_characteristics: dict,
        datafolders_manager: DataFoldersManager = DataFoldersManager.get_current_instance(),
    ):
        self.datafolders_manager = datafolders_manager
        self.wanted_characteristics = wanted_characteristics
        self.unwanted_characteristics = unwanted_characteristics
        self.paths = None
        self.filter_datafolders()

    def filter_datafolders(self):
        self.selected_datafolders = self.datafolders_manager.filter_datafolders(
            self.wanted_characteristics, self.unwanted_characteristics
        )

    @property
    def file_paths(self):
        if self.paths is None:
            self.paths = []
            for data_folder in self.selected_datafolders:
                self.paths += data_folder.file_paths
        return self.paths

    @property
    def n_datapoints(self):
        return len(self.file_paths)


class DatasetOutputManager:
    """This class is the one that handles the files to write the dataset"""

    def __init__(
        self,
        dataset_name: str,
        dataset_type: str,
        identifiers: dict,
        datafolders_manager: DataFoldersManager = DataFoldersManager.get_current_instance(),
    ):
        self.datafolders_manager = datafolders_manager
        self.dataset_name = dataset_name
        if self.dataset_name in [
            datafolder.dataset_name for datafolder in self.datafolders_manager.index
        ]:
            user_decision = input(
                "The proposed name is already in use, overwrite [y/n]?"
            )
            if user_decision.lower() == "y":
                self.datafolders_manager.delete_datafolders(
                    self.datafolders_manager.filter_datafolders(
                        wanted_characteristics={"dataset_name": self.dataset_name}
                    )
                )
            else:
                self.dataset_name = self.datafolders_manager.gen_new_name(
                    self.dataset_name
                )
        self.dataset_type = dataset_type
        self.identifyers = identifiers

    def new_datafolder(self, path_to_env: str):
        self.current_datafolder = self.datafolders_manager.new_datafolder(
            self.dataset_name, self.dataset_type, path_to_env, self.identifyers
        )
        self.current_datafolder_dtp_counter = 0

    def get_path_to_new_train_sample(self, extension="npz"):
        filename = f"{self.current_datafolder_dtp_counter:010d}.{extension}"
        self.current_datafolder_dtp_counter += 1
        return os.path.join(self.current_datafolder.path, filename)
