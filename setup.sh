#!/bin/bash
source ~/Documents/lab2_env/bin/activate
source client.sh
source devel/setup.bash
sudo /usr/sbin/nvpmodel -m 8
roslaunch traj_planning_ros ilqr_planning.launch