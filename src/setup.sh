#!/bin/bash

echo "start update & upgrade ..."
sudo apt update -y
sudo apt upgrade -y
sudo apt autoremove -y

#echo sudo apt install ros-melodic-rosbridge-server

