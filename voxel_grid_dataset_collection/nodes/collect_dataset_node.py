#!/usr/bin/python
import rospy
from gazebo_msgs.srv import SetModelState, SetModelStateResponse, SetModelStateRequest
from voxel_grid_dataset.dataset import TwoVoxelGridsTwoTransformsDataset
import json
import os
import threading
import math
import numpy as np
from tqdm import tqdm
from shapely import Polygon, Point
import time
from voxelgrid_msgs.msg import VoxelGridFloat32MultiarrayStamped


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return qx, qy, qz, qw


def quaternion_to_euler(q):
    (x, y, z, w) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]


def gen_poses_from_free_space(free_space_info, dist) -> (Point, Point):
    # TODO: what hapens if there are more than one free space
    for free_space in free_space_info:
        if free_space["type"] == "polygon":
            return gen_two_points_in_polygon(free_space, dist)


def random_points_in_polygon(polygon: Polygon, number: int):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


def gen_2d_point_around_point(x, y, len):
    len = np.random.uniform(0, len)
    angle = np.random.uniform(0, 2 * np.pi)
    dx = len * np.math.cos(angle)
    dy = len * np.math.sin(angle)
    return x + dx, y + dy


def gen_two_points_in_polygon(polygon_info, dist) -> (Point, Point):
    polygon = Polygon(np.array(polygon_info["points"])[:, :2])
    i_point = random_points_in_polygon(polygon, 1)[0]
    assert isinstance(i_point, Point)
    while True:
        x, y = gen_2d_point_around_point(i_point.x, i_point.y, dist)
        if polygon.contains(Point(x, y)):
            break
    return i_point, Point(x, y)


class TopicStorage:
    def __init__(self, topic_name, topic_type, time_to_sleep=0.05):
        print(f"created topic storage for topic {topic_name}")
        rospy.Subscriber(topic_name, topic_type, callback=self.callback)
        self.time_to_sleep = time_to_sleep
        self.msg_recieved = False
        self.last_msg = None

    def callback(self, msg):
        self.msg_recieved = True
        self.last_msg = msg

    def block(self, n_msgs=1):
        for i in range(n_msgs):
            self.msg_recieved = False
            while not self.msg_recieved:
                time.sleep(self.time_to_sleep)
        return self.last_msg


class DatasetCollectionNode:
    def __init__(self):
        rospy.init_node("dataset_collection_node")
        self.paths_to_envs = rospy.get_param("~paths_to_envs").split(",")
        self.n_samples_per_env = rospy.get_param("~n_samples_per_env")
        self.robot_name = rospy.get_param("~robot_name")
        self.voxelgrid_topic = rospy.get_param("~voxelgrid_topic")
        self.dataset_name = rospy.get_param("~dataset_name")
        self.max_dist_between_poses = rospy.get_param("~max_dist_between_poses")
        self.ros_thread = threading.Thread(target=self.ros_thread_target)
        self.dataset = TwoVoxelGridsTwoTransformsDataset(
            self.dataset_name,
            mode="write",
            identifiers={
                "voxel_size": rospy.get_param("~voxel_size"),
                "max_x": rospy.get_param("~max_x"),
                "max_y": rospy.get_param("~max_y"),
                "max_z": rospy.get_param("~max_z"),
            },
        )
        self.move_robot_service_proxy = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        self.voxelgrid_msg_storage = TopicStorage(
            self.voxelgrid_topic, VoxelGridFloat32MultiarrayStamped
        )

    def move_robot(self, *args):
        if len(args) == 1:
            x, y, z, roll, pitch, yaw = args[0]
        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)
        rqst = SetModelStateRequest()
        rqst.model_state.model_name = self.robot_name
        rqst.model_state.reference_frame = ""
        rqst.model_state.pose.position.x = x
        rqst.model_state.pose.position.y = y
        rqst.model_state.pose.position.z = z
        rqst.model_state.pose.orientation.x = qx
        rqst.model_state.pose.orientation.y = qy
        rqst.model_state.pose.orientation.z = qz
        rqst.model_state.pose.orientation.w = qw
        self.move_robot_service_proxy.call(rqst)

    def ros_thread_target(self):
        rospy.spin()

    def run(self):
        self.ros_thread.start()
        for path_to_env in tqdm(self.paths_to_envs):
            self.dataset.new_env(path_to_env)
            free_space_file = os.path.join(path_to_env, "free_space.json")
            with open(free_space_file, "r") as f:
                free_space_info = json.load(f)
            for i in tqdm(range(self.n_samples_per_env)):
                xy1, xy2 = gen_poses_from_free_space(
                    free_space_info, self.max_dist_between_poses
                )
                label1 = (
                    xy1.x,
                    xy1.y,
                    0.181821,
                    0,
                    0,
                    np.random.uniform(0, 2 * np.pi),
                )
                label2 = (
                    xy2.x,
                    xy2.y,
                    0.181821,
                    0,
                    0,
                    np.random.uniform(0, 2 * np.pi),
                )
                self.move_robot(label1)
                vg1 = self.voxelgrid_msg_storage.block(n_msgs=1)
                self.move_robot(label2)
                vg2 = self.voxelgrid_msg_storage.block(n_msgs=1)
                assert isinstance(vg1, VoxelGridFloat32MultiarrayStamped)
                assert isinstance(vg2, VoxelGridFloat32MultiarrayStamped)
                # TODO: The pose should have an adjustable z
                self.dataset.write_datapoint(
                    np.reshape(
                        np.array(vg1.voxel_grid.array.data),
                        (
                            vg1.voxel_grid.array.layout.dim[0].size,
                            vg1.voxel_grid.array.layout.dim[1].size,
                            vg1.voxel_grid.array.layout.dim[2].size,
                        ),
                    ),
                    np.reshape(
                        np.array(vg2.voxel_grid.array.data),
                        (
                            vg2.voxel_grid.array.layout.dim[0].size,
                            vg2.voxel_grid.array.layout.dim[1].size,
                            vg2.voxel_grid.array.layout.dim[2].size,
                        ),
                    ),
                    label1,
                    label2,
                )
        rospy.signal_shutdown("Dataset collected")
        self.ros_thread.join()


def main():
    node = DatasetCollectionNode()
    node.run()


if __name__ == "__main__":
    main()
