#!/bin/sh

if [ -z "$ROS_DISTRO" ]; then
  echo "ROS not installed. Check the installation ros_melodic"
fi


echo "# SET ROS Melodic" >> ~/.bashrc
echo "source ~/driving_ws/devel/setup.bash" >> ~/.bashrc
echo "# SET ROS NETWORK" >> ~/.bashrc
echo "export ROS_HOSTNAME=localhost" >> ~/.bashrc
echo "export ROS_MASTER_URI=http://localhost:11311" >> ~/.bashrc
echo "# SET ROS alias command" >> ~/.bashrc
echo "alias dw='cd ~/driving_ws'" >> ~/.bashrc
echo "alias ds='cd ~/driving_ws/src'" >> ~/.bashrc
echo "alias dm='cd ~/driving_ws && catkin_make && source devel/setup.bash'" >> ~/.bashrc
