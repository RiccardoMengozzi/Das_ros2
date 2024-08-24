# Das_ros2
The project applies distributed optimal control algorithms to 2 different topics: non linear classification problems and distributed autonomous control. The second task, in particular, is about autonomous surveillance by a team of robots, this has been also simulated in ros2.

# Requirements
-matplotlib
-numpy
-networkx
-ros2 foxy or humble
-rviz2

# Basic Instructions
A report is present in the folder that explains well how the code works and the final results of the project.

The project is divided into 2 script folders. The "Project" folder contains tasks from 1.1 to 2.1, the "src" folder contains the ros2 simulation of the task 2.2.
All the tasks but the 2.2 work by simply running the proper .py file.

For the ros2 simulation you need to open 2 terminals: one to run the script, the other one to open rviz2 for the simulation.

# Task 2.2 usage

open rviz2:
```console
rviz2
```
Then open the project rviz2 configuration: File -> Open Config and selecting src\agent\rviz_visu.rviz

Run the launch file:
```console
ros2 launch agent aggregative_optimization.launch.py
```


