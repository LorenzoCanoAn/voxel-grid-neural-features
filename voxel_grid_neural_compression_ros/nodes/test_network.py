#!/bin/python
import rospy
from voxel_grid_neural_compression.neural_netrworks import VoxelGridCompressor
import torch
from voxelgrid_msgs.msg import VoxelGridFloat32MultiarrayStamped
import numpy as np


def voxel_msg_to_tensor(msg: VoxelGridFloat32MultiarrayStamped):
    xdim = msg.voxel_grid.array.layout.dim[0].size
    ydim = msg.voxel_grid.array.layout.dim[1].size
    zdim = msg.voxel_grid.array.layout.dim[2].size
    return torch.reshape(torch.tensor(msg.voxel_grid.array.data), (xdim, ydim, zdim))


class TestNetworkNode:
    def __init__(self):
        self.first_embedding = None
        rospy.init_node("network_test_node")
        self.path_to_model = rospy.get_param("~path_to_model")
        self.voxel_topic = rospy.get_param("~voxel_topic")
        self.network = VoxelGridCompressor()
        self.network.load_state_dict(torch.load(self.path_to_model))
        self.network = self.network.to("cuda")
        rospy.Subscriber(
            self.voxel_topic, VoxelGridFloat32MultiarrayStamped, self.callback
        )

    def callback(self, msg: VoxelGridFloat32MultiarrayStamped):
        voxel_grid = voxel_msg_to_tensor(msg)
        voxel_grid = torch.reshape(
            voxel_grid,
            [1, 1, voxel_grid.shape[0], voxel_grid.shape[1], voxel_grid.shape[2]],
        ).to("cuda")
        embedding = self.network.compress(voxel_grid).to("cuda")
        if self.first_embedding is None:
            self.first_embedding = embedding
        distance = self.network.compare(self.first_embedding, embedding)
        print(f"distance: {distance.item():3.3f}")

    def run(self):
        rospy.spin()


def main():
    node = TestNetworkNode()
    node.run()


if __name__ == "__main__":
    main()
