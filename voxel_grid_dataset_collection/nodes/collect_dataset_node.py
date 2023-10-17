import rospy
from voxel_grid_dataset.dataset_io import DatasetOutputManager
import threading

class DatasetCollectionNode:
    def __init__(self):
        rospy.init_node("dataset_collection_node")
        self.ros_thread = threading.Thread(target=self.ros_thread_target)

    def ros_thread_target(self):
        rospy.spin() 
    
    def run(self):
        self.ros_thread.start()
        for i in range(1):
            pass
        
        self.ros_thread.join()


def main():
    node = DatasetCollectionNode()
    node.run()
    

if __name__ == "__main__":
    main()
