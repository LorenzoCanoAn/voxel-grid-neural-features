# Voxel grid datasets

This package defines a python file handler to interact with the file-structure of a voxel dataset.

The overall idea is to have a single folder in $HOME/.datasets/.voxel datasets, that contains an index.json file and all the datafolders with the actual data. The index.json contains the information of each of the data folders.

Each data folder has a unique ID, and has been taken in an specific environment. 

A dataset can be made of different dataset, and it is possible to assemble datasets by assembling by uising different datafolders.