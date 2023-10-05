# driving_ws
This workspace was created for line detect.

OS : ubuntu 18.04
ROS: ROS melodic

# Set up assignment1
1. Open the Terminal
2. set-up env
<pre>
  <code>
      $ ./setup.sh
      $ ./setup_bashrc.sh
  </code>
</pre>

start roslaunch & simulation
<pre>
  <code>
    $ $ roslaunch rosbridge_server rosbridge_websocket.launch
    $ cd driving_ws/src/pre/xycar_sim_driving
    $ ./xycar3Dsimulator.x86_64
  </code>
</pre>

http://xytron.co.kr/?page_id=394
