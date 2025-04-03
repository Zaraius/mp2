# Module 2: Implementation of Inverse Kinematics in Simulator
By Xavier Nishikawa, Zaraius Billimoria, & Julian Shah
based on code from Kenechukwu Mbanisi

This project is the mini project 2 code for the Fundamentals of Robotics class taught by Kenechukwu Mbanisi (Kene). 
It builds of the simulator built by Kene to implement analytical and numerical inverse kinematics for a 5-DOF robot
arm. While there is also a framework for 2-DOF and SCARA type robot arms, it isn't used. 

### Disclosure for AI

No AI models were used in developing the code.

<img src = "media/FPK.png">
An example for what the visualization tool looks like

## Requirements

* Step 1: Install Python 3 (if not already installed)
* Step 2: Locally download this repository from Github
* Step 3: Install all required Python packages using pip (or manually)

## How to Run

- To run the main scripts, use the command below
``` bash
$ python3 main_arm.py --robot_type 5-dof
# this configures the five-DOF arm
```


### Usage Guide

<img src = "media/arm-kinematics-viz-tool.png">



## **Forward position kinematics (FPK)**

Our implementation of inverse numerical kinematics uses forward kinematics to calculate the error of the robot position.
The forward kinematic implementation is copied from Kene's implementation and is not solely our work. Interfacing with
the FPK can be done through the sliders in the GUI (graphical user interface). 

## **Inverse position kinematics (IPK)**

### Analytical

The ananlytical inverse kinematics uses a geometric approach to solve for the desired joint angles of a 5-DOF Hiwonder
robot arm. This results in multiple solutions, with only one or two feasible for physical constraints to meet. Such 
constraints are, for example, joint limits and link lengths.

You may notice that there are two buttons in the Viz tool to solve for the analytical inverse kinematics. When there are
two real solutions for the robot to move two, both buttons will appropriately move the robot. However, there isn't always
two solutions because of joint limits and joint lengths provide physical limitations. Assuming that an imputted point is
within a set of real solutions for a 5-DOF arm, there will always be at least one solution that will work (one of the buttons).

Note: you can use the FPK to move to a real point, input those points into the IK area, and reset the arm before solving 
the IK solutions. This will ensure that you are inputting a real point.

<video src="ikPose2.mp4" width="320" height="240" controls>video</video>


### Numerical

The numerical inverse kinematics approach involves the Newton-Raphson method for optimization. In essence, we calculate
the current position of the end effector (EE) in relation to the desired position and multiply the error by the jacobian
to get new joint positions. This then happens until the EE is within an error threshold or an iteration limit is reached.
It is possible that upon inputting a desired and reachable point that it requires a few presses of the NUM SOLVE button
until the final solution is reached. This is because the path to the desired position may involve passing through
singularities which will erratically move the robot due to a product of a pseudo inverse jacobian used for calculations.

Note: orientation is not accounted for in this approach.

<video src="numericalPose2.mp4" width="320" height="240" controls>video</video>
