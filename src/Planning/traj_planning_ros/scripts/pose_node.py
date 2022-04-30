#!/usr/bin/env python
#!/usr/bin/env python
import rospy
from IPython import display
from threading import Lock
import matplotlib.pyplot as plt

# TODO: import the message type for the subscribed topic
# Use '''rostopic type ''' as shown in the readme to determine
# the proper type
from geometry_msgs.msg import PoseStamped


class PoseSubscriber:
    '''
    This class subscribes to the ros "/zed2/zed_node/pose" topic, and 
    save the most recent 200 position [x,y] in to lists

    '''
    def __init__(self):
        self.x_traj = []
        self.y_traj = []
        # Lock to avoid thread race
        self.lock = Lock()
        '''
        TODO: Here you need to finish the rest of your subscriber class
        As we have done before, you may need to create some objects and 
        add some functions

        Hint 1: you need to use self.lock.acquire() before putting data to 
        the self.x_traj and self.y_traj lists
        After you finish adding value to lists, use self.lock.release()
        Checkout https://docs.python.org/3/library/threading.html#lock-objects 
        for more info
        
        Hint 2: You can do  self.x_traj.pop(0) to pop the first node in the list
        in order to maintain the maximum list size.
        '''

        rospy.init_node("camera_listener", anonymous=True)
        self.camera_subscriber = rospy.Subscriber("/nx15/zed2/zed_node/pose", PoseStamped, callback=self.callback)


    def callback(self, data):
        self.lock.acquire()
        self.x_traj.append(data.pose.position.x)
        self.y_traj.append(data.pose.position.y)
        # print(self.x_traj)
        if (len(self.x_traj) > 200):
            self.x_traj.pop(0)
        if (len(self.y_traj) > 200):
            self.y_traj.pop(0)

        print(data.pose)
        self.lock.release()
 
if __name__ == "__main__":
    listener = PoseSubscriber()
    plt.ion()
    plt.show()
    plt.figure(figsize=(5, 5))

    while not rospy.is_shutdown():
        
        display.clear_output(wait = True)
        display.display(plt.gcf())
        plt.clf()

        # avoid thread race 
        listener.lock.acquire()
        plt.scatter(listener.x_traj, listener.y_traj)
        listener.lock.release()
        plt.xlim((-20, 20))
        plt.ylim((-20, 20))
        plt.pause(0.001)
        plt.pause(0.001)
