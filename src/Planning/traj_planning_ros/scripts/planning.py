#!/usr/bin/env python

import threading
import rospy
import numpy as np
from iLQR import iLQR, Track, EllipsoidObj
from realtime_buffer import RealtimeBuffer
from traj_msgs.msg import TrajMsg
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from rc_control_msgs.msg import RCControl
import yaml, csv
import time
import math


class State():
    def __init__(self, state, t) -> None:
        self.state = state
        self.t = t
        
class Plan():
    def __init__(self, x, u, K, t0, dt, N) -> None:
        self.nominal_x = x
        self.nominal_u = u
        self.K = K
        self.t0 = t0
        self.dt = dt
        self.N = N
    
    def get_policy(self, t):
        k = int(np.floor((t-self.t0).to_sec()/self.dt))
        if k>= self.N-1:
            rospy.logwarn("Try to retrive policy beyond horizon")
            x_k = self.nominal_x[:,-1]
            x_k[2:] = 0
            u_k = np.zeros(2)
            K_k = np.zeros((2,4))
        else:
            x_k = self.nominal_x[:,k]
            u_k = self.nominal_u[:,k]
            K_k = self.K[:,:,k]

        return x_k, u_k, K_k

        
class Planning_MPC():

    def __init__(self,
                track_file=None,
                pose_topic='/zed2/zed_node/odom',
                leader_pose_topic='/nx1/zed2/zed_node/odom',
                control_topic='/planning/trajectory',
                params_file='modelparams.yaml'):
        '''
        Main class for the MPC trajectory planner
        Input:
            freq: frequence to publish the control input to ESC and Servo
            T: prediction time horizon for the MPC
            N: number of integration steps in the MPC
        '''
        # load parameters
        with open(params_file) as file:
            self.params = yaml.load(file, Loader=yaml.FullLoader)

        # parameters for the ocp solver
        self.T = self.params['T']
        self.N = self.params['N']
        self.d_open_loop = np.array(self.params['d_open_loop'])
        self.replan_dt = self.T / (self.N - 1)

        # create track
        if track_file is None:
            # make a circle with 1.5m radius
            r = 1

            theta = np.linspace(0, 2 * np.pi, 100, endpoint=True)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            self.track = Track(np.array([x, y]), 0.5, 0.5, True)
        else:
            x = []
            y = []
            with open(track_file, newline='') as f:
                spamreader = csv.reader(f, delimiter=',')
                for i, row in enumerate(spamreader):
                    if i > 0:
                        x.append(float(row[0]))
                        y.append(float(row[1]))

            center_line = np.array([x, y])
            self.track = Track(center_line=center_line,
                            width_left=self.params['track_width_L'],
                            width_right=self.params['track_width_R'],
                            loop=True)

        # set up the optimal control solver

        self.ocp_solver = iLQR(self.track, params=self.params)

        rospy.loginfo("Successfully initialized the solver with horizon " +
                    str(self.T) + "s, and " + str(self.N) + " steps.")

        self.state_buffer = RealtimeBuffer()
        self.plan_buffer = RealtimeBuffer()
        self.leader_state_buffer = RealtimeBuffer()
        self.leader_waypoints = np.zeros([4,self.N])

        # set up publiser to the reference trajectory and subscriber to the pose
        self.control_pub = rospy.Publisher(control_topic, RCControl, queue_size=1)

        
        self.pose_sub = rospy.Subscriber(pose_topic,
                                        Odometry,
                                        self.odom_sub_callback,
                                        queue_size=1)
        

        self.leader_pose_sub = rospy.Subscriber(leader_pose_topic,
                                        Odometry,
                                        self.leader_odom,
                                        queue_size=1)

        self.counter = 0
        # start planning thread
        # threading.Thread(target=self.pid_thread).start()

    def leader_odom(self, odomMsg):
        """
        Subscriber callback function of the robot pose
        """
        cur_t = odomMsg.header.stamp
        # postion
        x = odomMsg.pose.pose.position.x
        y = odomMsg.pose.pose.position.y

        # pose
        r = Rotation.from_quat([
            odomMsg.pose.pose.orientation.x, odomMsg.pose.pose.orientation.y,
            odomMsg.pose.pose.orientation.z, odomMsg.pose.pose.orientation.w
        ])

        rot_vec = r.as_rotvec()
        psi = rot_vec[2]
        
        # get previous state
        prev_state = self.leader_state_buffer.readFromRT()

        # linear velocity
        if prev_state is not None:
            dx = x - prev_state.state[0]
            dy = y - prev_state.state[1]
            #if dx**2 + dy**2 > 0.5:
            #    return
            dt = (cur_t-prev_state.t).to_sec()
            v = np.sqrt(dx * dx + dy * dy) / dt
        else:
            v = 0

        leader_cur_X = np.array([x, y, v, psi])


            
        self.leader_state_buffer.writeFromNonRT(State(leader_cur_X, cur_t))
        leader_cur_X = leader_cur_X.reshape((4,1))


        if self.counter %5 == 0:
            self.leader_waypoints = np.append(self.leader_waypoints[:,1:self.N],leader_cur_X, axis=1)
        self.counter += 1
        #rospy.loginfo(leader_cur_X)

        # write the new pose to the buffer

    def odom_sub_callback(self, odomMsg):
        """
        Subscriber callback function of the robot pose
        """
        cur_t = odomMsg.header.stamp
        # postion
        x = odomMsg.pose.pose.position.x
        y = odomMsg.pose.pose.position.y

        # pose
        r = Rotation.from_quat([
            odomMsg.pose.pose.orientation.x, odomMsg.pose.pose.orientation.y,
            odomMsg.pose.pose.orientation.z, odomMsg.pose.pose.orientation.w
        ])

        rot_vec = r.as_rotvec()
        psi = rot_vec[2]
        
        # get previous state
        prev_state = self.state_buffer.readFromRT()

        # linear velocity
        if prev_state is not None:
            dx = x - prev_state.state[0]
            dy = y - prev_state.state[1]
            dt = (cur_t-prev_state.t).to_sec()
            v = np.sqrt(dx * dx + dy * dy) / dt
        else:
            v = 0
        cur_X = np.array([x, y, v, psi])
        # obtain the latest leader state
        leader_X = self.leader_waypoints[:,-2]  # follow old 
        # get the control policy
        
        u = self.pid(cur_X, leader_X)
        throttle, steer = u
        self.publish_control(throttle, steer, cur_t)
        # write the new pose to the buffer
        # self.state_buffer.writeFromNonRT(State(cur_X, cur_t))

    def publish_control(self, throttle, steer, cur_t):
        control = RCControl()
        control.header.stamp = cur_t
        
        ## CHANGE THIS
        control.throttle = throttle
        control.steer = steer
        control.reverse = False
        #rospy.loginfo(control)
        self.control_pub.publish(control)


    def pid_thread(self):
        time.sleep(3)
        rospy.loginfo("iLQR Planning publishing thread started")
        while not rospy.is_shutdown():
            # determine if we need to publish
            
            cur_state = self.state_buffer.readFromRT()
            prev_plan = self.plan_buffer.readFromRT()
            if cur_state is None:
                continue
            since_last_pub = self.replan_dt if prev_plan is None else (
                cur_state.t - prev_plan.t0).to_sec()
            if since_last_pub >= self.replan_dt:

                leader_waypoints = np.array(self.leader_waypoints)
                rospy.loginfo(leader_waypoints)
                if prev_plan is None:
                    u_init = None
                else:
                    u_init = np.zeros((2, self.N))
                    u_init[:, :-1] = prev_plan.nominal_u[:, 1:]

                # add in obstacle to solver
                # ego_a = 0.5 / 2.0
                # ego_b = 0.2 / 2.0
                # ego_q = np.array([0, 5.6])[:, np.newaxis] 
                # ego_Q = np.diag([ego_a**2, ego_b**2])
                # static_obs = EllipsoidObj(q=ego_q, Q=ego_Q)
                # static_obs_list = [static_obs for _ in range(self.N)]
                
                sol_x, sol_u, _, _, sol_K, _, _ = self.ocp_solver.solve(
                    cur_state.state, leader_waypoints, u_init, record=True, obs_list=[])
                # print(np.round(sol_x,2))
                # print(np.round(sol_u[1,:],2))
                cur_plan = Plan(sol_x, sol_u, sol_K, cur_state.t, self.replan_dt, self.N)
                self.plan_buffer.writeFromNonRT(cur_plan)
                

    def pidManager(self, waypoints):
        pass
    
    def pid(self, startState, goalState):
        """
        Args:
            startState (numpy array): (x, y, v, theta)
            goalState (numpy array): (x,y, v, theta)
        """
        dist = np.linalg.norm(startState[:2] - goalState[:2])
        thetaErr = self.calcThetaError(startState, goalState)
        vErr = startState[2] - goalState[2]

        kp_throttle = .2
        kd_throttle = .000001
        kp_steer = 1.5
        throttle = kp_throttle * dist + kd_throttle * vErr
        steer = -kp_steer*thetaErr

        steer = np.clip(steer, -1, 1)
        throttle = np.clip(throttle, -.2, .2)

        if (dist < 0.3):
            throttle = 0

        if self.counter % 50 == 0:
            rospy.loginfo(f"startState:{startState}")
            rospy.loginfo(f"goalState:{goalState}")
            rospy.loginfo(f"thetaErr:{np.rad2deg(thetaErr)}")
            rospy.loginfo(f"dist:{dist}")
            rospy.loginfo(f"throttle:{throttle}")
            rospy.loginfo(f"steer:{steer}")

        steer = steer + 0.2 # steering correction
        return np.array([throttle, steer])

    def calcThetaError(self,startState: np.ndarray, goalState: np.ndarray) -> float:
        """Calculates the directional angle error between the direction the car
        is facing and where it should be facing if it were to travel straight 
        to the goal point
        
        Args:
            startState (numpy array): the (x,y, v, theta) state that the car is currently in
            goalState (numpy array): the (x,y, v, theta) state that the car should travel to
        
        Returns:
            float: The signed angle theta between the car's direction and the direction towards the goal

        """
        return goalState[3] - startState[3] 
        # This step is highly important - get the vector pointing from the car
        # to the target point
        car2goal = startState[:2] - goalState[:2]
        _,_,_, theta = startState
        car2goal = car2goal / np.linalg.norm(car2goal)
        carHeading = np.array([np.cos(theta), np.sin(theta)])

        # Calculate the signed angle between two vectors (can potentially be
        # done with the "perpendicular dot product" but here is accomplished
        # by utilizing code inspired by p5.js source code for the angleBetween
        # function)
        theta = math.acos(min(1, max(-1, carHeading.dot(car2goal))))
        theta = theta * np.sign(np.cross(carHeading, car2goal)) or 1
        return theta

    def run(self):
        rospy.spin() 