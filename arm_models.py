from math import sqrt, sin, cos, atan, atan2, degrees
import time
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from helper_fcns.utils import (
    EndEffector,
    rotm_to_euler,
    euler_to_rotm,
    check_joint_limits,
    dh_to_matrix,
    near_zero,
    wraptopi,
)

PI = 3.1415926535897932384
# np.set_printoptions(precision=3)


class Robot:
    """
    Represents a robot manipulator with various kinematic configurations.
    Provides methods to calculate forward kinematics, inverse kinematics, and velocity kinematics.
    Also includes methods to visualize the robot's motion and state in 3D.

    Attributes:
        num_joints (int): Number of joints in the robot.
        ee_coordinates (list): List of end-effector coordinates.
        robot (object): The robot object (e.g., TwoDOFRobot, ScaraRobot, etc.).
        origin (list): Origin of the coordinate system.
        axes_length (float): Length of the axes for visualization.
        point_x, point_y, point_z (list): Lists to store coordinates of points for visualization.
        show_animation (bool): Whether to show the animation or not.
        plot_limits (list): Limits for the plot view.
        fig (matplotlib.figure.Figure): Matplotlib figure for 3D visualization.
        sub1 (matplotlib.axes._subplots.Axes3DSubplot): Matplotlib 3D subplot.
    """

    def __init__(self, type="2-dof", show_animation: bool = True):
        """
        Initializes a robot with a specific configuration based on the type.

        Args:
            type (str, optional): Type of robot (e.g., '2-dof', 'scara', '5-dof'). Defaults to '2-dof'.
            show_animation (bool, optional): Whether to show animation of robot movement. Defaults to True.
        """
        if type == "2-dof":
            self.num_joints = 2
            self.ee_coordinates = ["X", "Y"]
            self.robot = TwoDOFRobot()

        elif type == "scara":
            self.num_joints = 3
            self.ee_coordinates = ["X", "Y", "Z", "Theta"]
            self.robot = ScaraRobot()

        elif type == "5-dof":
            self.num_joints = 5
            self.ee_coordinates = ["X", "Y", "Z", "RotX", "RotY", "RotZ"]
            self.robot = FiveDOFRobot()
            
        elif type == "6-dof":
            self.num_joints = 6
            self.ee_coordinates = ["X", "Y", "Z", "RotX", "RotY", "RotZ"]
            self.robot = SixDOFRobot()

        self.origin = [0.0, 0.0, 0.0]
        self.axes_length = 0.04
        self.point_x, self.point_y, self.point_z = [], [], []
        self.waypoint_x, self.waypoint_y, self.waypoint_z = [], [], []
        self.waypoint_rotx, self.waypoint_roty, self.waypoint_rotz = [], [], []
        self.show_animation = show_animation
        self.plot_limits = [1, 1, 1.2]

        if self.show_animation:
            self.fig = Figure(figsize=(12, 10), dpi=100)
            self.sub1 = self.fig.add_subplot(1, 1, 1, projection="3d")
            self.fig.suptitle("Manipulator Kinematics Visualization", fontsize=16)

        # initialize figure plot
        self.init_plot()

    def init_plot(self):
        """Initializes the plot by calculating the robot's points and calling the plot function."""
        self.robot.calc_robot_points()
        self.plot_3D()

    def update_plot(self, pose=None, angles=None, soln=0, numerical=False):
        """
        Updates the robot's state based on new pose or joint angles and updates the visualization.

        Args:
            pose (EndEffector, optional): Desired end-effector pose for inverse kinematics.
            angles (list, optional): Joint angles for forward kinematics.
            soln (int, optional): The inverse kinematics solution to use (0 or 1).
            numerical (bool, optional): Whether to use numerical inverse kinematics.
        """
        if pose is not None:  # Inverse kinematics case
            if not numerical:
                self.robot.calc_inverse_kinematics(pose, soln=soln)
            else:
                self.robot.calc_numerical_ik(pose, tol=0.02, ilimit=50)
        elif angles is not None:  # Forward kinematics case
            self.robot.calc_forward_kinematics(angles, radians=False)
        else:
            return
        self.plot_3D()

    def move_velocity(self, vel):
        """
        Moves the robot based on a given velocity input.

        Args:
            vel (list): Velocity input for the robot.
        """
        self.robot.calc_velocity_kinematics(vel)
        self.plot_3D()

    def draw_line_3D(self, p1, p2, format_type: str = "k-"):
        """
        Draws a 3D line between two points.

        Args:
            p1 (list): Coordinates of the first point.
            p2 (list): Coordinates of the second point.
            format_type (str, optional): The format of the line. Defaults to "k-".
        """
        self.sub1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], format_type)

    def draw_ref_line(self, point, axes=None, ref="xyz"):
        """
        Draws reference lines from a given point along specified axes.

        Args:
            point (list): The coordinates of the point to draw from.
            axes (matplotlib.axes, optional): The axes on which to draw the reference lines.
            ref (str, optional): Which reference axes to draw ('xyz', 'xy', or 'xz'). Defaults to 'xyz'.
        """
        line_width = 0.7
        if ref == "xyz":
            axes.plot(
                [point[0], self.plot_limits[0]],
                [point[1], point[1]],
                [point[2], point[2]],
                "b--",
                linewidth=line_width,
            )  # X line
            axes.plot(
                [point[0], point[0]],
                [point[1], self.plot_limits[1]],
                [point[2], point[2]],
                "b--",
                linewidth=line_width,
            )  # Y line
            axes.plot(
                [point[0], point[0]],
                [point[1], point[1]],
                [point[2], 0.0],
                "b--",
                linewidth=line_width,
            )  # Z line
        elif ref == "xy":
            axes.plot(
                [point[0], self.plot_limits[0]],
                [point[1], point[1]],
                "b--",
                linewidth=line_width,
            )  # X line
            axes.plot(
                [point[0], point[0]],
                [point[1], self.plot_limits[1]],
                "b--",
                linewidth=line_width,
            )  # Y line
        elif ref == "xz":
            axes.plot(
                [point[0], self.plot_limits[0]],
                [point[2], point[2]],
                "b--",
                linewidth=line_width,
            )  # X line
            axes.plot(
                [point[0], point[0]], [point[2], 0.0], "b--", linewidth=line_width
            )  # Z line

    def plot_waypoints(self):
        """
        Plots the waypoints in the 3D visualization
        """
        # draw the points
        self.sub1.plot(
            self.waypoint_x, self.waypoint_y, self.waypoint_z, "or", markersize=8
        )

    def update_waypoints(self, waypoints: list):
        """
        Updates the waypoints into a member variable
        """
        for i in range(len(waypoints)):
            self.waypoint_x.append(waypoints[i][0])
            self.waypoint_y.append(waypoints[i][1])
            self.waypoint_z.append(waypoints[i][2])
            # self.waypoint_rotx.append(waypoints[i][3])
            # self.waypoint_roty.append(waypoints[i][4])
            # self.waypoint_rotz.append(waypoints[i][5])

    def plot_3D(self):
        """
        Plots the 3D visualization of the robot, including the robot's links, end-effector, and reference frames.
        """
        self.sub1.cla()
        self.point_x.clear()
        self.point_y.clear()
        self.point_z.clear()

        EE = self.robot.ee

        # draw lines to connect the points
        for i in range(len(self.robot.points) - 1):
            self.draw_line_3D(self.robot.points[i], self.robot.points[i + 1])
        #     print("Robot points:")
        # for i, p in enumerate(self.robot.points):
        #     print(f"  Point {i}: {p}")


        # draw the points
        for i in range(len(self.robot.points)):
            # print(f"{self.robot.points[i][0]=}")
            # print(f"{self.robot.points[i][1]=}")
            # print(f"{self.robot.points[i][2]=}")
            self.point_x.append(self.robot.points[i][0])
            self.point_y.append(self.robot.points[i][1])
            self.point_z.append(self.robot.points[i][2])
        self.sub1.plot(
            self.point_x,
            self.point_y,
            self.point_z,
            marker="o",
            markerfacecolor="m",
            markersize=12,
        )

        # draw the waypoints
        self.plot_waypoints()

        # draw the waypoints
        self.plot_waypoints()

        # draw the EE
        self.sub1.plot(EE.x, EE.y, EE.z, "bo")
        # draw the base reference frame
        self.draw_line_3D(
            self.origin,
            [self.origin[0] + self.axes_length, self.origin[1], self.origin[2]],
            format_type="r-",
        )
        self.draw_line_3D(
            self.origin,
            [self.origin[0], self.origin[1] + self.axes_length, self.origin[2]],
            format_type="g-",
        )
        self.draw_line_3D(
            self.origin,
            [self.origin[0], self.origin[1], self.origin[2] + self.axes_length],
            format_type="b-",
        )
        # draw the EE reference frame
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[0], format_type="r-")
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[1], format_type="g-")
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[2], format_type="b-")
        # draw reference / trace lines
        self.draw_ref_line([EE.x, EE.y, EE.z], self.sub1, ref="xyz")

        # add text at bottom of window
        pose_text = "EE:[ "
        pose_text += f"X: {round(EE.x,4)},  "
        pose_text += f"Y: {round(EE.y,4)},  "
        pose_text += f"Z: {round(EE.z,4)},  "
        pose_text += f"RotX: {round(EE.rotx,4)},  "
        pose_text += f"RotY: {round(EE.roty,4)},  "
        pose_text += f"RotZ: {round(EE.rotz,4)}  "
        pose_text += " ]"

        theta_text = "Joint Positions (deg/m):     ["
        for i in range(self.num_joints):
            theta_text += f" {round(np.rad2deg(self.robot.theta[i]),2)}, "
        theta_text += " ]"

        textstr = pose_text + "\n" + theta_text
        self.sub1.text2D(
            0.2, 0.02, textstr, fontsize=13, transform=self.fig.transFigure
        )

        self.sub1.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        self.sub1.set_ylim(-self.plot_limits[1], self.plot_limits[1])
        self.sub1.set_zlim(0, self.plot_limits[2])
        self.sub1.set_xlabel("x [m]")
        self.sub1.set_ylabel("y [m]")


class TwoDOFRobot:
    """
    Represents a 2-degree-of-freedom (DOF) robot arm with two joints and one end effector.
    Includes methods for calculating forward kinematics (FPK), inverse kinematics (IPK),
    and velocity kinematics (VK).

    Attributes:
        l1 (float): Length of the first arm segment.
        l2 (float): Length of the second arm segment.
        theta (list): Joint angles.
        theta_limits (list): Joint limits for each joint.
        ee (EndEffector): The end effector object.
        points (list): List of points representing the robot's configuration.
        num_dof (int): The number of degrees of freedom (2 for this robot).
    """

    def __init__(self):
        """
        Initializes a 2-DOF robot with default arm segment lengths and joint angles.
        """
        self.l1 = 0.30  # Length of the first arm segment
        self.l2 = 0.25  # Length of the second arm segment

        self.theta = [0.0, 0.0]  # Joint angles (in radians)
        self.theta_limits = [[-PI, PI], [-PI + 0.261, PI - 0.261]]  # Joint limits

        self.ee = EndEffector()  # The end-effector object
        self.num_dof = 2  # Number of degrees of freedom
        self.points = [None] * (self.num_dof + 1)  # List to store robot points

    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculates the forward kinematics for the robot based on the joint angles.

        Args:
            theta (list): Joint angles.
            radians (bool, optional): Whether the angles are in radians or degrees. Defaults to False.
        """
        if not radians:
            # Convert degrees to radians if the input is in degrees
            self.theta[0] = np.deg2rad(theta[0])
            self.theta[1] = np.deg2rad(theta[1])
        else:
            self.theta = theta

        # Ensure that the joint angles respect the joint limits
        for i, th in enumerate(self.theta):
            self.theta[i] = np.clip(
                th, self.theta_limits[i][0], self.theta_limits[i][1]
            )

        # Update the robot configuration (i.e., the positions of the joints and end effector)
        self.calc_robot_points()

    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculates the inverse kinematics (IK) for a given end effector position.

        Args:
            EE (EndEffector): The end effector object containing the target position (x, y).
            soln (int, optional): The solution branch to use. Defaults to 0 (first solution).
        """

        x, y = EE.x, EE.y
        l1, l2 = self.l1, self.l2

        # Move robot slightly out of zero configuration to avoid singularity
        if all(th == 0.0 for th in self.theta):
            self.theta = [
                self.theta[i] + np.random.rand() * 0.01 for i in range(self.num_dof)
            ]

        try:
            if soln == 0:
                # Solution 0: Calculate joint angles using cosine rule
                beta = np.arccos((l1**2 + l2**2 - x**2 - y**2) / (2 * l1 * l2))
                self.theta[1] = PI - beta
                c2 = np.cos(self.theta[1])
                s2 = np.sin(self.theta[1])

                alpha = atan2((l2 * s2), (l1 + l2 * c2))
                gamma = atan2(y, x)
                self.theta[0] = gamma - alpha
            elif soln == 1:
                # Solution 1: Alternative calculation for theta_1 and theta_2
                beta = np.arccos((l1**2 + l2**2 - x**2 - y**2) / (2 * l1 * l2))
                self.theta[1] = beta - PI
                c2 = np.cos(self.theta[1])
                s2 = np.sin(self.theta[1])

                alpha = atan2((l2 * s2), (l1 + l2 * c2))
                gamma = atan2(y, x)
                self.theta[0] = gamma - alpha
            else:
                raise ValueError("Invalid IK solution specified!")

            if not check_joint_limits(self.theta, self.theta_limits):
                print(
                    f"\n [ERROR] Joint limits exceeded! \n \
                      Desired joints are {self.theta} \n \
                      Joint limits are {self.theta_limits}"
                )
                raise ValueError

        except RuntimeWarning:
            print("[ERROR] Joint limits exceeded (runtime issue).")

        # Calculate robot points based on the updated joint angles
        self.calc_robot_points()

    def calc_numerical_ik(self, EE: EndEffector, tol=0.01, ilimit=50):
        """
        Calculates numerical inverse kinematics (IK) based on input end effector coordinates.

        Args:
            EE (EndEffector): The end effector object containing the target position (x, y).
            tol (float, optional): The tolerance for the solution. Defaults to 0.01.
            ilimit (int, optional): The maximum number of iterations. Defaults to 50.
        """

        x, y = EE.x, EE.y

        ########################################

        # insert your code here

        ########################################

        self.calc_robot_points()

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy].
        """

        # move robot slightly out of zeros singularity
        if all(th == 0.0 for th in self.theta):
            self.theta = [
                self.theta[i] + np.random.rand() * 0.02 for i in range(self.num_dof)
            ]

        # Calculate joint velocities using the inverse Jacobian
        vel = vel[:2]  # Consider only the first two components of the velocity
        thetadot = self.inverse_jacobian() @ vel

        # print(f'thetadot: {thetadot}')

        # (Corrective measure) Ensure joint velocities stay within limits
        self.thetadot_limits = [[-PI, PI], [-PI, PI]]
        thetadot = np.clip(
            thetadot,
            [limit[0] for limit in self.thetadot_limits],
            [limit[1] for limit in self.thetadot_limits],
        )

        # print(f'Limited thetadot: {thetadot}')

        # Update the joint angles based on the velocity
        self.theta[0] += 0.02 * thetadot[0]
        self.theta[1] += 0.02 * thetadot[1]

        # Ensure joint angles stay within limits
        self.theta = np.clip(
            self.theta,
            [limit[0] for limit in self.theta_limits],
            [limit[1] for limit in self.theta_limits],
        )

        # Update robot points based on the new joint angles
        self.calc_robot_points()

    def jacobian(self, theta: list = None):
        """
        Returns the Jacobian matrix for the robot. If theta is not provided,
        the function will use the object's internal theta attribute.

        Args:
            theta (list, optional): The joint angles for the robot. Defaults to self.theta.

        Returns:
            np.ndarray: The Jacobian matrix (2x2).
        """
        # Use default values if arguments are not provided
        if theta is None:
            theta = self.theta

        return np.array(
            [
                [
                    -self.l1 * sin(theta[0]) - self.l2 * sin(theta[0] + theta[1]),
                    -self.l2 * sin(theta[0] + theta[1]),
                ],
                [
                    self.l1 * cos(theta[0]) + self.l2 * cos(theta[0] + theta[1]),
                    self.l2 * cos(theta[0] + theta[1]),
                ],
            ]
        )

    def inverse_jacobian(self):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        J = self.jacobian()
        print(f"Determinant of J: {np.linalg.det(J)}")
        # return np.linalg.inv(self.jacobian())
        return np.linalg.pinv(self.jacobian())

    def solve_forward_kinematics(self, theta: list, radians=False):
        """
        Evaluates the forward kinematics based on the joint angles.

        Args:
            theta (list): The joint angles [theta_1, theta_2].
            radians (bool, optional): Whether the input angles are in radians (True) or degrees (False).
                                      Defaults to False (degrees).

        Returns:
            list: The end effector position [x, y].
        """
        if not radians:
            theta[0] = np.deg2rad(theta[0])
            theta[1] = np.deg2rad(theta[1])

        # Compute forward kinematics
        x = self.l1 * cos(theta[0]) + self.l2 * cos(theta[0] + theta[1])
        y = self.l1 * sin(theta[0]) + self.l2 * sin(theta[0] + theta[1])

        return [x, y]

    def calc_robot_points(self):
        """
        Calculates the positions of the robot's joints and the end effector.

        Updates the `points` list, storing the coordinates of the base, shoulder, elbow, and end effector.
        """
        # Base position
        self.points[0] = [0.0, 0.0, 0.0]
        # Shoulder joint
        self.points[1] = [
            self.l1 * cos(self.theta[0]),
            self.l1 * sin(self.theta[0]),
            0.0,
        ]
        # Elbow joint
        self.points[2] = [
            self.l1 * cos(self.theta[0]) + self.l2 * cos(self.theta[0] + self.theta[1]),
            self.l1 * sin(self.theta[0]) + self.l2 * sin(self.theta[0] + self.theta[1]),
            0.0,
        ]

        # Update end effector position
        self.ee.x = self.points[2][0]
        self.ee.y = self.points[2][1]
        self.ee.z = self.points[2][2]
        self.ee.rotz = self.theta[0] + self.theta[1]

        # End effector axes
        self.EE_axes = np.zeros((3, 3))
        self.EE_axes[0] = (
            np.array(
                [
                    cos(self.theta[0] + self.theta[1]),
                    sin(self.theta[0] + self.theta[1]),
                    0,
                ]
            )
            * 0.075
            + self.points[2]
        )
        self.EE_axes[1] = (
            np.array(
                [
                    -sin(self.theta[0] + self.theta[1]),
                    cos(self.theta[0] + self.theta[1]),
                    0,
                ]
            )
            * 0.075
            + self.points[2]
        )
        self.EE_axes[2] = np.array([0, 0, 1]) * 0.075 + self.points[2]


class ScaraRobot:
    """
    A class representing a SCARA (Selective Compliance Assembly Robot Arm) robot.
    This class handles the kinematics (forward, inverse, and velocity kinematics)
    and robot configuration, including joint limits and end-effector calculations.
    """

    def __init__(self):
        """
        Initializes the SCARA robot with its geometry, joint variables, and limits.
        Sets up the transformation matrices and robot points.
        """
        # Geometry of the robot (link lengths in meters)
        self.l1 = 0.35  # Base to 1st joint
        self.l2 = 0.18  # 1st joint to 2nd joint
        self.l3 = 0.15  # 2nd joint to 3rd joint
        self.l4 = 0.30  # 3rd joint to 4th joint (tool or end-effector)
        self.l5 = 0.12  # Tool offset

        # Joint variables (angles in radians)
        self.theta = [0.0, 0.0, 0.0]

        # Joint angle limits (min, max) for each joint
        self.theta_limits = [
            [-np.pi, np.pi],
            [-np.pi + 0.261, np.pi - 0.261],
            [0, self.l1 + self.l3 - self.l5],
        ]

        # End-effector (EE) object to store EE position and orientation
        self.ee = EndEffector()

        # Number of degrees of freedom and number of points to store robot configuration
        self.num_dof = 3
        self.num_points = 7
        self.points = [None] * self.num_points

        # Transformation matrices (DH parameters and resulting transformation)
        self.DH = np.zeros((5, 4))  # Denavit-Hartenberg parameters (theta, d, a, alpha)
        self.T = np.zeros((self.num_dof, 4, 4))  # Transformation matrices

    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate Forward Kinematics (FK) based on the given joint angles.

        Args:
            theta (list): Joint angles (in radians if radians=True, otherwise in degrees).
            radians (bool): Whether the input angles are in radians (default is False).
        """
        if not radians:
            self.theta[0] = np.deg2rad(theta[0])
            self.theta[1] = np.deg2rad(theta[1])
            self.theta[2] = theta[2]  # No need to convert z-axis theta
        else:
            self.theta = theta

        # Apply joint limits after updating joint angles
        for i, th in enumerate(self.theta):
            self.theta[i] = np.clip(
                th, self.theta_limits[i][0], self.theta_limits[i][1]
            )

        # DH parameters for each joint
        ee_home = self.l3 - self.l5
        self.DH[0] = [self.theta[0], self.l1, self.l2, 0]
        self.DH[1] = [self.theta[1], ee_home, self.l4, 0]
        self.DH[2] = [0, -self.theta[2], 0, np.pi]

        # Calculate transformation matrices for each joint
        for i in range(self.num_dof):
            self.T[i] = dh_to_matrix(self.DH[i])

        # Calculate robot points (e.g., end-effector position)
        self.calc_robot_points()

    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculate Inverse Kinematics (IK) based on the input end-effector coordinates.

        Args:
            EE (EndEffector): End-effector object containing desired position (x, y, z).
            soln (int): Solution index (0 or 1), for multiple possible IK solutions.
        """
        x, y, z = EE.x, EE.y, EE.z
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5

        # Slightly perturb the robot from singularity at zero configuration
        if all(th == 0.0 for th in self.theta):
            self.theta = [
                self.theta[i] + np.random.rand() * 0.01 for i in range(self.num_dof)
            ]

        try:
            # Select inverse kinematics solution
            if soln == 0:
                # Calculate theta_3
                self.theta[2] = l1 + l3 - l5 - z

                # Using the cosine rule, calculate theta_2
                beta = np.arccos((l2**2 + l4**2 - x**2 - y**2) / (2 * l2 * l4))
                self.theta[1] = np.pi - beta

                # Using trigonometry, find theta_1
                c2 = np.cos(self.theta[1])
                s2 = np.sin(self.theta[1])
                alpha = np.arctan2(l4 * s2, l2 + l4 * c2)
                gamma = np.arctan2(y, x)
                self.theta[0] = gamma - alpha

            elif soln == 1:
                # Alternate solution for theta_1 and theta_2
                self.theta[2] = l1 + l3 - l5 - z
                beta = np.arccos((l2**2 + l4**2 - x**2 - y**2) / (2 * l2 * l4))
                self.theta[1] = beta - np.pi

                c2 = np.cos(self.theta[1])
                s2 = np.sin(self.theta[1])
                alpha = np.arctan2(l4 * s2, l2 + l4 * c2)
                gamma = np.arctan2(y, x)
                self.theta[0] = gamma - alpha

            else:
                raise ValueError("Invalid IK solution specified!")

            # Check joint limits and ensure validity
            if not check_joint_limits(self.theta, self.theta_limits):
                print(
                    f"\n [ERROR] Joint limits exceeded! \n \
                      Desired joints are {self.theta} \n \
                      Joint limits are {self.theta_limits}"
                )
                raise ValueError

        except Exception as e:
            print(f"Error in Inverse Kinematics: {e}")
            return

        # Recalculate Forward Kinematics to update the robot's configuration
        self.calc_forward_kinematics(self.theta, radians=True)

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate velocity kinematics and update joint velocities.

        Args:
            vel (array): Linear velocities (3D) of the end-effector.
        """
        # Ensure small deviation from singularity at the zero configuration
        if all(th == 0.0 for th in self.theta):
            self.theta = [
                self.theta[i] + np.random.rand() * 0.01 for i in range(self.num_dof)
            ]

        # Compute the inverse of the Jacobian to update joint velocities
        thetadot = self.inverse_jacobian() @ vel

        # Update joint angles based on computed joint velocities
        self.theta[0] += 0.02 * thetadot[0]
        self.theta[1] += 0.02 * thetadot[1]
        self.theta[2] -= 0.01 * thetadot[2]

        # Apply joint limits after updating joint angles
        for i, th in enumerate(self.theta):
            self.theta[i] = np.clip(
                th, self.theta_limits[i][0], self.theta_limits[i][1]
            )

        # Recalculate robot points based on updated joint angles
        self.calc_robot_points()

    def jacobian(self):
        """
        Compute the Jacobian matrix for the robot's end-effector.

        Returns:
            numpy.ndarray: 3x3 Jacobian matrix.
        """
        ee_home = self.l3 - self.l5
        self.DH[0] = [self.theta[0], self.l1, self.l2, 0]
        self.DH[1] = [self.theta[1], ee_home, self.l4, 0]
        self.DH[2] = [0, -self.theta[2], 0, np.pi]

        # Update transformation matrices
        for i in range(self.num_dof):
            self.T[i] = dh_to_matrix(self.DH[i])

        # Extract transformation matrices
        T0_1 = self.T[0]
        T0_2 = self.T[0] @ self.T[1]
        T0_3 = self.T[0] @ self.T[1] @ self.T[2]
        O0 = np.array([0, 0, 0, 1])

        jacobian = np.zeros((3, 3))

        # calculate Jv(1)
        # z0 X r, r = (Oc - O0)
        r = (T0_3 @ O0 - O0)[:3]
        # z0 = T0_1[:3,:3] @ np.array([0, 0, 1])
        z0 = np.array([0, 0, 1])
        Jv1 = np.linalg.cross(z0, r)

        # calculate Jv(2)
        r = (T0_3 @ O0 - T0_1 @ O0)[:3]
        z1 = T0_1[:3, :3] @ np.array([0, 0, 1])
        Jv2 = np.linalg.cross(z1, r)

        # calculate Jv(3)
        r = (T0_3 @ O0 - T0_2 @ O0)[:3]
        z2 = T0_2[:3, :3] @ np.array([0, 0, 1])
        Jv3 = z2

        jacobian[:, 0] = Jv1
        jacobian[:, 1] = Jv2
        jacobian[:, 2] = Jv3

        return jacobian

    def inverse_jacobian(self):
        """
        Compute the inverse of the Jacobian matrix.

        Returns:
            numpy.ndarray: Inverse Jacobian matrix.
        """
        return np.linalg.inv(self.jacobian())

    def calc_robot_points(self):
        """
        Calculate the main robot points (links and end-effector position) using the current joint angles.
        Updates the robot's points array and end-effector position.
        """
        # Calculate transformation matrices for each joint and end-effector
        self.points[0] = np.array([0, 0, 0, 1])
        self.points[1] = np.array([0, 0, self.l1, 1])
        self.points[2] = self.T[0] @ self.points[0]
        self.points[3] = self.points[2] + np.array([0, 0, self.l3, 1])
        self.points[4] = self.T[0] @ self.T[1] @ self.points[0] + np.array(
            [0, 0, self.l5, 1]
        )
        self.points[5] = self.T[0] @ self.T[1] @ self.points[0]
        self.points[6] = self.T[0] @ self.T[1] @ self.T[2] @ self.points[0]

        self.EE_axes = (
            self.T[0] @ self.T[1] @ self.T[2] @ np.array([0.075, 0.075, 0.075, 1])
        )
        self.T_ee = self.T[0] @ self.T[1] @ self.T[2]

        # End-effector (EE) position and axes
        self.ee.x = self.points[-1][0]
        self.ee.y = self.points[-1][1]
        self.ee.z = self.points[-1][2]
        rpy = rotm_to_euler(self.T_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy

        # EE coordinate axes
        self.EE_axes = np.zeros((3, 3))
        self.EE_axes[0] = self.T_ee[:3, 0] * 0.075 + self.points[-1][0:3]
        self.EE_axes[1] = self.T_ee[:3, 1] * 0.075 + self.points[-1][0:3]
        self.EE_axes[2] = self.T_ee[:3, 2] * 0.075 + self.points[-1][0:3]


class FiveDOFRobot:
    """
    A class to represent a 5-DOF robotic arm with kinematics calculations, including
    forward kinematics, inverse kinematics, velocity kinematics, and Jacobian computation.

    Attributes:
        l1, l2, l3, l4, l5: Link lengths of the robotic arm.
        theta: List of joint angles in radians.
        theta_limits: Joint limits for each joint.
        ee: End-effector object for storing the position and orientation of the end-effector.
        num_dof: Number of degrees of freedom (5 in this case).
        points: List storing the positions of the robot joints.
        DH: Denavit-Hartenberg parameters for each joint.
        T: Transformation matrices for each joint.
    """

    def __init__(self):
        """Initialize the robot parameters and joint limits."""
        # Link lengths
        # self.l1, self.l2, self.l3, self.l4, self.l5 = 0.30, 0.15, 0.18, 0.15, 0.12
        self.l1, self.l2, self.l3, self.l4, self.l5 = (
            0.155,
            0.099,
            0.095,
            0.055,
            0.105,
        )  # from hardware measurements

        # Joint angles (initialized to zero)
        self.theta = [0, 0, 0, 0, 0]

        # Joint limits (in radians)
        self.theta_limits = [
            [-np.pi, np.pi],
            [-np.pi / 3, np.pi],
            [-np.pi + np.pi / 12, np.pi - np.pi / 4],
            [-np.pi + np.pi / 12, np.pi - np.pi / 12],
            [-np.pi, np.pi],
        ]

        self.thetadot_limits = [
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
        ]

        # End-effector object
        self.ee = EndEffector()

        # Robot's points
        self.num_dof = 5
        self.points = [None] * (self.num_dof + 1)

        # Denavit-Hartenberg parameters and transformation matrices
        self.DH = np.zeros((5, 4))
        self.T = np.zeros((self.num_dof, 4, 4))

    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate forward kinematics based on the provided joint angles.

        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """
        if not radians:
            # Convert degrees to radians
            self.theta = np.deg2rad(theta)
        else:
            self.theta = theta

        # Apply joint limits
        self.theta = [
            np.clip(th, self.theta_limits[i][0], self.theta_limits[i][1])
            for i, th in enumerate(self.theta)
        ]

        # Set the Denavit-Hartenberg parameters for each joint
        self.DH[0] = [self.theta[0], self.l1, 0, np.pi / 2]
        self.DH[1] = [self.theta[1] + np.pi / 2, 0, self.l2, np.pi]
        self.DH[2] = [self.theta[2], 0, self.l3, np.pi]
        self.DH[3] = [self.theta[3] - np.pi / 2, 0, 0, -np.pi / 2]
        self.DH[4] = [self.theta[4], self.l4 + self.l5, 0, 0]

        # Compute the transformation matrices
        for i in range(self.num_dof):
            self.T[i] = dh_to_matrix(self.DH[i])

        # Calculate robot points (positions of joints)
        self.calc_robot_points()

    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculate inverse kinematics to determine the joint angles based on end-effector position.

        Args:
            EE: EndEffector object containing desired position and orientation.
            soln: Optional parameter for multiple solutions (not implemented).
        """

        # initializing a list for each theta
        theta1_list = []
        theta2_list = []
        theta3_list = []
        theta4_list = []
        theta5_list = []

        # calculating theta 1 value and then adding it to the list
        theta_1 = np.arctan2(EE.y, EE.x)
        theta1_list.append(theta_1)

        # calculating and adding the other theta 1 value
        if theta_1 > 0:
            theta1_list.append(theta_1 - np.pi)  # angle wrapping case 1
        else:
            theta1_list.append(theta_1 + np.pi)  # angle wrapping case 2

        # find rotation of EE matrix
        EE_euler = (EE.rotx, EE.roty, EE.rotz)
        EE_rot = np.array(euler_to_rotm(EE_euler))

        # find position of joint 3 in base frame
        P_EE = np.array([EE.x, EE.y, EE.z])
        k = np.array([0, 0, 1])

        # subtracting l4 and l5 in the z direction from the end effector pos
        P_3 = P_EE - ((self.l4 + self.l5) * (EE_rot @ k))

        # Solve decoupled kinematic problem in frame 1 for theta 2 & 3
        P3_x, P3_y, P3_z = P_3[0], P_3[1], P_3[2]
        L = np.sqrt(P3_x**2 + P3_y**2 + (P3_z - self.l1) ** 2)  # solve for L in 3D

        # Check if point is reachable
        max_reach = self.l2 + self.l3  # maximum reach
        min_reach = abs(self.l2 - self.l3)  # minimum reach

        if L > max_reach or L < min_reach:
            print(
                f"Point unreachable: Distance {L:.3f} is outside arm's reach [{min_reach:.3f}, {max_reach:.3f}]"
            )

        delta_x = np.sqrt(P3_x**2 + P3_y**2)  # solve for x displacement from 1 to 3
        delta_z = P3_z - self.l1  # solve for z displacement from 1 to 3
        alpha = np.arctan2(delta_z, delta_x)

        # calculate phi using law of cosine
        phi = np.arccos((self.l2**2 + self.l3**2 - L**2) / (2 * self.l2 * self.l3))
        # adding positive and negative variations of theta 3 to a list
        theta3_list.append(np.pi - phi)
        theta3_list.append(-(np.pi - phi))

        # calculating theta 2
        # first solution of theta 2
        gamma1 = self.l3 * sin(theta3_list[0])
        beta1 = np.arcsin(gamma1 / L)
        theta2_list.append(-(alpha - beta1 - np.pi / 2))

        # second solution of theta 2
        gamma2 = self.l3 * sin(theta3_list[1])
        beta2 = np.arcsin(gamma2 / L)
        theta2_list.append(alpha + beta2 - np.pi / 2)


        # We compute R_0_3 by plugging in thetas 1 2 and 3 into the rotation matrix
        # multiply wrist pos: R_3_5 = (R_0_3)^T * R_0_6 (transpose is the same thing as inverse in this case)
        # R_0_5 is the end effector orientation (found from the eulertorotm function of the end effector)
        # Calculate R_3_5, gives a 3x3 matrix of values. These values mean nothing unless you know the functions 
        # they are associated with. We can matrix multiply symbolically R_3_4 and R_4_5 which 
        # will symbolically make R_3_5. With these, there are a few cos and sin functions that standalone with a theta 
        # (ex. r_3_5[1,2] = sin(theta4)). With some simple inverse trigonometric functions, we can find what the theta values are.       

        # looping through all combinations of theta 1 2 to get all combinations of theta 4 and 5
        for i in range(2):
            for j in range(2):
                theta_1 = theta1_list[i]
                theta_2 = theta2_list[j]
                theta_3 = theta3_list[j]
                # DH matrix for current theta values
                dh1 = dh_to_matrix([theta_1, self.l1, 0, np.pi / 2])
                dh2 = dh_to_matrix([theta_2 + np.pi / 2, 0, self.l2, 0])
                dh3 = dh_to_matrix([-theta_3, 0, self.l3, 0])

                # getting the rotation matrix of 0 to 3 from dh matrix
                t_0_3 = dh1 @ dh2 @ dh3
                r_0_3 = t_0_3[:3, :3]
                # getting the rotation matrix from 3 to 5
                r_3_5 = np.transpose(r_0_3) @ EE_rot

                # EXPLAIN THIS IN WORDS
                theta_4 = wraptopi(np.arctan2(r_3_5[1, 2], r_3_5[0, 2]))

                theta_5 = wraptopi(np.arctan2(r_3_5[2, 0], r_3_5[2, 1]))

                # adding theta 4 and 5 to their list
                theta4_list.append(theta_4)
                theta5_list.append(theta_5)

        # adding all combinations of theta values into the solutions list
        solutions = []
        solutions.append(
            [
                theta1_list[1],
                theta2_list[0],
                theta3_list[0],
                theta4_list[2],
                theta5_list[1],
            ]
        )
        solutions.append(
            [
                theta1_list[1],
                theta2_list[1],
                theta3_list[1],
                theta4_list[2],
                theta5_list[1],
            ]
        )
        solutions.append(
            [
                theta1_list[0],
                theta2_list[0],
                theta3_list[0],
                theta4_list[1],
                theta5_list[2],
            ]
        )
        solutions.append(
            [
                theta1_list[0],
                theta2_list[1],
                theta3_list[1],
                theta4_list[1],
                theta5_list[2],
            ]
        )

        # initializing error list
        error_list = []
        # looping through each solution
        for i, solution in enumerate(solutions):
            # calculationg the position of this solution
            position = [EE.x, EE.y, EE.z, EE.rotx, EE.roty, EE.rotz]
            self.calc_forward_kinematics(solution, radians=True)
            calc_pos = [
                self.ee.x,
                self.ee.y,
                self.ee.z,
                self.ee.rotx,
                self.ee.roty,
                self.ee.rotz,
            ]
            # calculating the error from desired and actual position and adding it to the list
            error_list.append(
                [np.linalg.norm(np.array(position) - np.array(calc_pos)), i]
            )
        # sorting the error
        sorted_indices = np.argsort(np.array(error_list)[:, 0])

        # displaying the 4 positions in sorted order for each button
        if soln == 0:
            self.calc_forward_kinematics(solutions[sorted_indices[0]], radians=True)
        elif soln == 1:
            self.calc_forward_kinematics(solutions[sorted_indices[1]], radians=True)
        elif soln == 2:
            self.calc_forward_kinematics(solutions[sorted_indices[2]], radians=True)
        elif soln == 3:
            self.calc_forward_kinematics(solutions[sorted_indices[3]], radians=True)

    def calc_numerical_ik(self, EE: EndEffector, tol=0.01, ilimit=50):
        """Calculate numerical inverse kinematics based on input coordinates.

        Using Newton-Raphson method of optimization, move towards the desired position using
        the inverse jacobian times the positional error.

        Args
            EE: EnderEffector object containing desired EE positions
            tol: float defining allowable tolerance between desired position and true position
            ilimit: int defining maximum number of iterations to find allowable solution
        """

        # find desired EE position
        pos_des = [EE.x, EE.y, EE.z]

        # make sure that the loop doesn't run forever by defining iteration limit
        for _ in range(ilimit):
            # solve for error
            pos_current = self.solve_forward_kinematics(self.theta)
            error = pos_des - pos_current[0:3]
            # If error outside tol, recalculate theta (Newton-Raphson)
            if np.linalg.norm(error) > tol:
                self.theta = self.theta + np.dot(
                    self.inverse_jacobian(pseudo=True), error
                )
            # If error is within tolerence: break
            else:
                break


        self.calc_forward_kinematics(self.theta, radians=True)

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate the joint velocities required to achieve the given end-effector velocity.

        Args:
            vel: Desired end-effector velocity (3x1 vector).
        """
        # Avoid singularity by perturbing joint angles slightly
        if all(th == 0.0 for th in self.theta):
            self.theta = [th + np.random.rand() * 0.01 for th in self.theta]

        # Calculate the joint velocity using the inverse Jacobian
        # thetadot = self.inverse_jacobian(pseudo=True) @ vel
        thetadot = self.damped_inverse_jacobian() @ vel

        # (Corrective measure) Ensure joint velocities stay within limits
        thetadot = np.clip(
            thetadot,
            [limit[0] for limit in self.thetadot_limits],
            [limit[1] for limit in self.thetadot_limits],
        )

        # Update joint angles
        self.theta[0] += 0.02 * thetadot[0]
        self.theta[1] += 0.02 * thetadot[1]
        self.theta[2] += 0.02 * thetadot[2]
        self.theta[3] += 0.02 * thetadot[3]
        self.theta[4] += 0.02 * thetadot[4]

        # Recompute robot points based on updated joint angles
        self.calc_forward_kinematics(self.theta, radians=True)

    def jacobian(self, theta: list = None):
        """
        Compute the Jacobian matrix for the current robot configuration.

        Args:
            theta (list, optional): The joint angles for the robot. Defaults to self.theta.
        Returns:
            Jacobian matrix (3x5).
        """
        # Use default values if arguments are not provided
        if theta is None:
            theta = self.theta

        # Define DH parameters
        DH = np.zeros((5, 4))
        DH[0] = [theta[0], self.l1, 0, np.pi / 2]
        DH[1] = [theta[1] + np.pi / 2, 0, self.l2, np.pi]
        DH[2] = [theta[2], 0, self.l3, np.pi]
        DH[3] = [theta[3] - np.pi / 2, 0, 0, -np.pi / 2]
        DH[4] = [theta[4], self.l4 + self.l5, 0, 0]

        # Compute transformation matrices
        T = np.zeros((self.num_dof, 4, 4))
        for i in range(self.num_dof):
            T[i] = dh_to_matrix(DH[i])

        # Precompute transformation matrices for efficiency
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ T[i])

        # Define O0 for calculations
        O0 = np.array([0, 0, 0, 1])

        # Initialize the Jacobian matrix
        jacobian = np.zeros((3, self.num_dof))

        # Calculate the Jacobian columns
        for i in range(self.num_dof):
            T_curr = T_cumulative[i]
            T_final = T_cumulative[-1]

            # Calculate position vector r
            r = (T_final @ O0 - T_curr @ O0)[:3]

            # Compute the rotation axis z
            z = T_curr[:3, :3] @ np.array([0, 0, 1])

            # Compute linear velocity part of the Jacobian
            jacobian[:, i] = np.cross(z, r)

        # Replace near-zero values with zero, primarily for debugging purposes
        return near_zero(jacobian)

    def inverse_jacobian(self, pseudo=False):
        """
        Compute the inverse of the Jacobian matrix using either pseudo-inverse or regular inverse.

        Args:
            pseudo: Boolean flag to use pseudo-inverse (default is False).

        Returns:
            The inverse (or pseudo-inverse) of the Jacobian matrix.
        """

        J = self.jacobian()
        JT = np.transpose(J)
        manipulability_idx = np.sqrt(np.linalg.det(J @ JT))

        if pseudo:
            return np.linalg.pinv(self.jacobian())
        else:
            return np.linalg.inv(self.jacobian())

    def damped_inverse_jacobian(self, q=None, damping_factor=0.025):
        if q is not None:
            J = self.jacobian(q)
        else:
            J = self.jacobian()

        JT = np.transpose(J)
        I = np.eye(3)
        return JT @ np.linalg.inv(J @ JT + (damping_factor**2) * I)

    def dh_to_matrix(self, dh_params: list) -> np.ndarray:
        """Converts Denavit-Hartenberg parameters to a transformation matrix.

        Args:
            dh_params (list): Denavit-Hartenberg parameters [theta, d, a, alpha].

        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        theta, d, a, alpha = dh_params
        return np.array(
            [
                [
                    cos(theta),
                    -sin(theta) * cos(alpha),
                    sin(theta) * sin(alpha),
                    a * cos(theta),
                ],
                [
                    sin(theta),
                    cos(theta) * cos(alpha),
                    -cos(theta) * sin(alpha),
                    a * sin(theta),
                ],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

    def solve_forward_kinematics(self, theta: list, radians=True):
        """Solves the new transformation matrix from base to EE at current theta values"""

        # Convert degrees to radians
        if not radians:
            for i in range(len(theta)):
                theta[i] = np.deg2rad(theta[i])

        # DH parameters = [theta, d, a, alpha]
        DH = np.zeros((5, 4))
        DH[0] = [theta[0], self.l1, 0, np.pi / 2]
        DH[1] = [theta[1] + np.pi / 2, 0, self.l2, np.pi]
        DH[2] = [theta[2], 0, self.l3, np.pi]
        DH[3] = [theta[3] - np.pi / 2, 0, 0, -np.pi / 2]
        DH[4] = [theta[4], self.l4 + self.l5, 0, 0]

        T = np.zeros((self.num_dof, 4, 4))
        for i in range(self.num_dof):
            T[i] = dh_to_matrix(DH[i])

        return T[0] @ T[1] @ T[2] @ T[3] @ T[4] @ np.array([0, 0, 0, 1])

    def calc_robot_points(self):
        """Calculates the main arm points using the current joint angles"""

        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, 6):
            self.points[i] = T_cumulative[i] @ self.points[0]

        # Calculate EE position and rotation
        self.EE_axes = T_cumulative[-1] @ np.array(
            [0.075, 0.075, 0.075, 1]
        )  # End-effector axes
        self.T_ee = T_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        self.ee.x, self.ee.y, self.ee.z = self.points[-1][:3]

        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = rotm_to_euler(self.T_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[0], rpy[1], rpy[2]

        # Calculate the EE axes in space (in the base frame)
        self.EE = [self.ee.x, self.ee.y, self.ee.z]
        # print(f"{self.points=}")
        self.EE_axes = np.array(
            [self.T_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)]
        )



class SixDOFRobot:
    """
    A class to represent a 6-DOF robotic arm with kinematics calculations, including
    forward kinematics, inverse kinematics, velocity kinematics, and Jacobian computation.

    Attributes:
        l1, l2, l3, l4, l5, l6: Link lengths of the robotic arm.
        theta: List of joint angles in radians.
        theta_limits: Joint limits for each joint.
        ee: End-effector object for storing the position and orientation of the end-effector.
        num_dof: Number of degrees of freedom (6 in this case).
        points: List storing the positions of the robot joints.
        DH: Denavit-Hartenberg parameters for each joint.
        T: Transformation matrices for each joint.
    """

    def __init__(self):
        """Initialize the robot parameters and joint limits."""

        # Joint angles (initialized to zero)
        self.theta = [0, 0, 0, 0, 0, 0]
    
        self.l1 = (128.3 + 115.0) / 1000   # 243.3 mm
        self.l2 = 30.0 / 1000         # offset
        self.l3 = 280.0 / 1000      # long horizontal reach
        self.l4 = 20.0  / 1000     # small offset
        self.l5 = (140.0 + 105.0)  / 1000  # vertical section
        self.l6 = (28.5 + 28.5)  / 1000    # vertical section
        self.l7 = (105.0 + 130.0) / 1000   # vertical section to EE

        # Joint limits (in radians)
        self.theta_limits = [
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi]
        ]

        self.thetadot_limits = [
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
        ]

        # End-effector object
        self.ee = EndEffector()

        # Robot's points
        self.num_dof = 6
        self.points = [None] * (self.num_dof + 1)

        # Denavit-Hartenberg parameters and transformation matrices
        self.DH = np.zeros((self.num_dof, 4))
        self.T = np.zeros((self.num_dof, 4, 4))

    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate forward kinematics based on the provided joint angles.

        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """
        if not radians:
            # Convert degrees to radians
            self.theta = np.deg2rad(theta)
        else:
            self.theta = theta

        # Apply joint limits
        self.theta = [
            np.clip(th, self.theta_limits[i][0], self.theta_limits[i][1])
            for i, th in enumerate(self.theta)
        ]

        self.DH[0] = [self.theta[0], (128.3 + 115.0) / 1000, 0.0, np.pi / 2]
        self.DH[1] = [self.theta[1] + np.pi / 2, 30.0 / 1000, 280.0 / 1000, np.pi]
        self.DH[2] = [self.theta[2] + np.pi / 2, 20.0 / 1000, 0.0, np.pi / 2]
        self.DH[3] = [self.theta[3] + np.pi / 2, (140.0 + 105.0) / 1000, 0.0, np.pi / 2]
        self.DH[4] = [self.theta[4] + np.pi, (28.5 + 28.5) / 1000, 0.0, np.pi / 2]
        self.DH[5] = [self.theta[5] + np.pi / 2, (105.0 + 130.0) / 1000, 0.0, 0.0]


        # Compute the transformation matrices
        for i in range(self.num_dof):
            self.T[i] = dh_to_matrix(self.DH[i])

        # Calculate robot points (positions of joints)
        self.calc_robot_points()

    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculate inverse kinematics to determine the joint angles based on end-effector position.

        Args:
            EE: EndEffector object containing desired position and orientation.
            soln: Optional parameter for multiple solutions (not implemented).
        """

        # initializing a list for each theta
        theta1_list = []
        theta2_list = []
        theta3_list = []
        theta4_list = []
        theta5_list = []
        

        # calculating theta 1 value and then adding it to the list
        theta_1 = np.arctan2(EE.y, EE.x)
        theta1_list.append(theta_1)

        # calculating and adding the other theta 1 value
        if theta_1 > 0:
            theta1_list.append(theta_1 - np.pi)  # angle wrapping case 1
        else:
            theta1_list.append(theta_1 + np.pi)  # angle wrapping case 2

        # find rotation of EE matrix
        EE_euler = (EE.rotx, EE.roty, EE.rotz)
        EE_rot = np.array(euler_to_rotm(EE_euler))

        # find position of joint 3 in base frame
        P_EE = np.array([EE.x, EE.y, EE.z])
        k = np.array([0, 0, 1])

        # subtracting l4 and l5 in the z direction from the end effector pos
        # P_3 = P_EE - ((self.l4 + self.l5) * (EE_rot @ k))
        P_3 = P_EE - ((self.l7) * (EE_rot @ k))


        # Solve decoupled kinematic problem in frame 1 for theta 2 & 3
        P3_x, P3_y, P3_z = P_3[0], P_3[1], P_3[2]
        L = np.sqrt(P3_x**2 + P3_y**2 + (P3_z - self.l1) ** 2)  # solve for L in 3D

        # Check if point is reachable
        max_reach = self.l2 + self.l3  # maximum reach
        min_reach = abs(self.l2 - self.l3)  # minimum reach

        if L > max_reach or L < min_reach:
            print(
                f"Point unreachable: Distance {L:.3f} is outside arm's reach [{min_reach:.3f}, {max_reach:.3f}]"
            )

        delta_x = np.sqrt(P3_x**2 + P3_y**2)  # solve for x displacement from 1 to 3
        delta_z = P3_z - self.l1  # solve for z displacement from 1 to 3
        alpha = np.arctan2(delta_z, delta_x)

        # calculate phi using law of cosine
        phi = np.arccos((self.l2**2 + self.l3**2 - L**2) / (2 * self.l2 * self.l3))
        # adding positive and negative variations of theta 3 to a list
        theta3_list.append(np.pi - phi)
        theta3_list.append(-(np.pi - phi))

        # calculating theta 2
        # first solution of theta 2
        gamma1 = self.l3 * sin(theta3_list[0])
        beta1 = np.arcsin(gamma1 / L)
        theta2_list.append(-(alpha - beta1 - np.pi / 2))

        # second solution of theta 2
        gamma2 = self.l3 * sin(theta3_list[1])
        beta2 = np.arcsin(gamma2 / L)
        theta2_list.append(alpha + beta2 - np.pi / 2)


        # We compute R_0_3 by plugging in thetas 1 2 and 3 into the rotation matrix
        # multiply wrist pos: R_3_5 = (R_0_3)^T * R_0_6 (transpose is the same thing as inverse in this case)
        # R_0_5 is the end effector orientation (found from the eulertorotm function of the end effector)
        # Calculate R_3_5, gives a 3x3 matrix of values. These values mean nothing unless you know the functions 
        # they are associated with. We can matrix multiply symbolically R_3_4 and R_4_5 which 
        # will symbolically make R_3_5. With these, there are a few cos and sin functions that standalone with a theta 
        # (ex. r_3_5[1,2] = sin(theta4)). With some simple inverse trigonometric functions, we can find what the theta values are.       

        # looping through all combinations of theta 1 2 to get all combinations of theta 4 and 5
        for i in range(2):
            for j in range(2):
                theta_1 = theta1_list[i]
                theta_2 = theta2_list[j]
                theta_3 = theta3_list[j]
                # DH matrix for current theta values
                dh1 = dh_to_matrix([theta_1, self.l1, 0, np.pi / 2])
                dh2 = dh_to_matrix([theta_2 + np.pi / 2, 0, self.l2, 0])
                dh3 = dh_to_matrix([-theta_3, 0, self.l3, 0])

                # getting the rotation matrix of 0 to 3 from dh matrix
                t_0_3 = dh1 @ dh2 @ dh3
                r_0_3 = t_0_3[:3, :3]
                # getting the rotation matrix from 3 to 5
                r_3_5 = np.transpose(r_0_3) @ EE_rot

                # EXPLAIN THIS IN WORDS
                theta_4 = wraptopi(np.arctan2(r_3_5[1, 2], r_3_5[0, 2]))

                theta_5 = wraptopi(np.arctan2(r_3_5[2, 0], r_3_5[2, 1]))

                # adding theta 4 and 5 to their list
                theta4_list.append(theta_4)
                theta5_list.append(theta_5)

        # adding all combinations of theta values into the solutions list
        solutions = []
        solutions.append(
            [
                theta1_list[1],
                theta2_list[0],
                theta3_list[0],
                theta4_list[2],
                theta5_list[1],
            ]
        )
        solutions.append(
            [
                theta1_list[1],
                theta2_list[1],
                theta3_list[1],
                theta4_list[2],
                theta5_list[1],
            ]
        )
        solutions.append(
            [
                theta1_list[0],
                theta2_list[0],
                theta3_list[0],
                theta4_list[1],
                theta5_list[2],
            ]
        )
        solutions.append(
            [
                theta1_list[0],
                theta2_list[1],
                theta3_list[1],
                theta4_list[1],
                theta5_list[2],
            ]
        )

        # initializing error list
        error_list = []
        # looping through each solution
        for i, solution in enumerate(solutions):
            # calculationg the position of this solution
            position = [EE.x, EE.y, EE.z, EE.rotx, EE.roty, EE.rotz]
            self.calc_forward_kinematics(solution, radians=True)
            calc_pos = [
                self.ee.x,
                self.ee.y,
                self.ee.z,
                self.ee.rotx,
                self.ee.roty,
                self.ee.rotz,
            ]
            # calculating the error from desired and actual position and adding it to the list
            error_list.append(
                [np.linalg.norm(np.array(position) - np.array(calc_pos)), i]
            )
        # sorting the error
        sorted_indices = np.argsort(np.array(error_list)[:, 0])

        # displaying the 4 positions in sorted order for each button
        if soln == 0:
            self.calc_forward_kinematics(solutions[sorted_indices[0]], radians=True)
        elif soln == 1:
            self.calc_forward_kinematics(solutions[sorted_indices[1]], radians=True)
        elif soln == 2:
            self.calc_forward_kinematics(solutions[sorted_indices[2]], radians=True)
        elif soln == 3:
            self.calc_forward_kinematics(solutions[sorted_indices[3]], radians=True)

    def calc_numerical_ik(self, EE: EndEffector, tol=0.01, ilimit=100):
        """
        Numerical inverse kinematics for position-only control (x, y, z).
        Uses pseudoinverse of position Jacobian (3x6).
        """

        pos_des = np.array([EE.x, EE.y, EE.z])

        for i in range(ilimit):
            pos_current = self.solve_forward_kinematics(self.theta)[:3]
            error = pos_des - pos_current
            print(f"[{i}] Error: {error}, Norm: {np.linalg.norm(error):.4f}")

            if np.linalg.norm(error) < tol:
                break

            J_full = self.jacobian()
            J_pos = J_full[:3, :]  # Extract top 3 rows (∂x, ∂y, ∂z)

            delta_theta = 0.05 * np.linalg.pinv(J_pos) @ error
            self.theta = self.theta + delta_theta

            # Enforce joint limits
            self.theta = np.clip(
                self.theta,
                [lim[0] for lim in self.theta_limits],
                [lim[1] for lim in self.theta_limits]
            )

        # Recompute final FK after convergence
        self.calc_forward_kinematics(self.theta, radians=True)

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate the joint velocities required to achieve the given end-effector velocity.

        Args:
            vel: Desired end-effector velocity (3x1 vector).
        """
        # Avoid singularity by perturbing joint angles slightly
        if all(th == 0.0 for th in self.theta):
            self.theta = [th + np.random.rand() * 0.01 for th in self.theta]

        # Calculate the joint velocity using the inverse Jacobian
        # thetadot = self.inverse_jacobian(pseudo=True) @ vel
        thetadot = self.damped_inverse_jacobian() @ vel
        # (Corrective measure) Ensure joint velocities stay within limits
        thetadot = np.clip(
            thetadot,
            [limit[0] for limit in self.thetadot_limits],
            [limit[1] for limit in self.thetadot_limits],
        )
        # Update joint angles
        self.theta[0] += 0.02 * thetadot[0]
        self.theta[1] += 0.02 * thetadot[1]
        self.theta[2] += 0.02 * thetadot[2]
        self.theta[3] += 0.02 * thetadot[3]
        self.theta[4] += 0.02 * thetadot[4]
        self.theta[5] += 0.02 * thetadot[5]


        # Recompute robot points based on updated joint angles
        self.calc_forward_kinematics(self.theta, radians=True)

    def jacobian(self, theta: list = None):
        """
        Compute the Jacobian matrix for the current robot configuration.

        Args:
            theta (list, optional): The joint angles for the robot. Defaults to self.theta.
        Returns:
            Jacobian matrix (3x5).
        """
        # Use default values if arguments are not provided
        if theta is None:
            theta = self.theta

        # Define DH parameters
        DH = np.zeros((6, 4))
        
        DH[0] = [self.theta[0], (128.3 + 115.0) / 1000, 0.0, np.pi / 2]
        DH[1] = [self.theta[1] + np.pi / 2, 30.0 / 1000, 280.0 / 1000, np.pi]
        DH[2] = [self.theta[2] + np.pi / 2, 20.0 / 1000, 0.0, np.pi / 2]
        DH[3] = [self.theta[3] + np.pi / 2, (140.0 + 105.0) / 1000, 0.0, np.pi / 2]
        DH[4] = [self.theta[4] + np.pi, (28.5 + 28.5) / 1000, 0.0, np.pi / 2]
        DH[5] = [self.theta[5] + np.pi / 2, (105.0 + 130.0) / 1000, 0.0, 0.0]

        # Compute transformation matrices
        T = np.zeros((self.num_dof, 4, 4))
        for i in range(self.num_dof):
            T[i] = dh_to_matrix(DH[i])

        # Precompute transformation matrices for efficiency
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ T[i])

        # Define O0 for calculations
        O0 = np.array([0, 0, 0, 1])

        # Initialize the Jacobian matrix
        jacobian = np.zeros((3, self.num_dof))

        # Calculate the Jacobian columns
        for i in range(self.num_dof):
            T_curr = T_cumulative[i]
            T_final = T_cumulative[-1]

            # Calculate position vector r
            r = (T_final @ O0 - T_curr @ O0)[:3]

            # Compute the rotation axis z
            z = T_curr[:3, :3] @ np.array([0, 0, 1])

            # Compute linear velocity part of the Jacobian
            jacobian[:, i] = np.cross(z, r)
      # Replace near-zero values with zero, primarily for debugging purposes
        return near_zero(jacobian)

    def inverse_jacobian(self, pseudo=False):
        """
        Compute the inverse of the Jacobian matrix using either pseudo-inverse or regular inverse.

        Args:
            pseudo: Boolean flag to use pseudo-inverse (default is False).

        Returns:
            The inverse (or pseudo-inverse) of the Jacobian matrix.
        """

        J = self.jacobian()
        JT = np.transpose(J)
        manipulability_idx = np.sqrt(np.linalg.det(J @ JT))

        if pseudo:
            return np.linalg.pinv(self.jacobian())
        else:
            return np.linalg.inv(self.jacobian())

    def damped_inverse_jacobian(self, q=None, damping_factor=0.025):
        if q is not None:
            J = self.jacobian(q)
        else:
            J = self.jacobian()
        JT = np.transpose(J)
        I = np.eye(3)
        return JT @ np.linalg.inv(J @ JT + (damping_factor**2) * I)

    def dh_to_matrix(self, dh_params: list) -> np.ndarray:
        """Converts Denavit-Hartenberg parameters to a transformation matrix.

        Args:
            dh_params (list): Denavit-Hartenberg parameters [theta, d, a, alpha].

        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        theta, d, a, alpha = dh_params
        return np.array(
            [
                [
                    cos(theta),
                    -sin(theta) * cos(alpha),
                    sin(theta) * sin(alpha),
                    a * cos(theta),
                ],
                [
                    sin(theta),
                    cos(theta) * cos(alpha),
                    -cos(theta) * sin(alpha),
                    a * sin(theta),
                ],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

    def solve_forward_kinematics(self, theta: list, radians=True):
        """Solves the transformation matrix from base to EE at current theta values"""

        if not radians:
            theta = [np.deg2rad(t) for t in theta]  # convert in place safely

        # DH parameters = [theta, d, a, alpha]
        DH = np.zeros((6, 4))
        DH[0] = [theta[0], (128.3 + 115.0) / 1000, 0.0, np.pi / 2]
        DH[1] = [theta[1] + np.pi / 2, 30.0 / 1000, 280.0 / 1000, np.pi]
        DH[2] = [theta[2] + np.pi / 2, 20.0 / 1000, 0.0, np.pi / 2]
        DH[3] = [theta[3] + np.pi / 2, (140.0 + 105.0) / 1000, 0.0, np.pi / 2]
        DH[4] = [theta[4] + np.pi, (28.5 + 28.5) / 1000, 0.0, np.pi / 2]
        DH[5] = [theta[5] + np.pi / 2, (105.0 + 130.0) / 1000, 0.0, 0.0]

        T = np.zeros((self.num_dof, 4, 4))
        for i in range(self.num_dof):
            T[i] = dh_to_matrix(DH[i])

        T_total = T[0]
        for i in range(1, self.num_dof):
            T_total = T_total @ T[i]

        pos_ee = T_total @ np.array([0, 0, 0, 1])
        return pos_ee[:3]


    def calc_robot_points(self):
        """Calculates the main arm points using the current joint angles"""

        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, 7):
            self.points[i] = T_cumulative[i] @ self.points[0]

        # Calculate EE position and rotation
        self.EE_axes = T_cumulative[-1] @ np.array(
            [0.075, 0.075, 0.075, 1]
        )  # End-effector axes
        self.T_ee = T_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        self.ee.x, self.ee.y, self.ee.z = self.points[-1][:3]

        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = rotm_to_euler(self.T_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[0], rpy[1], rpy[2]

        # Calculate the EE axes in space (in the base frame)
        self.EE = [self.ee.x, self.ee.y, self.ee.z]
        # print(f"{self.points=}")
        self.EE_axes = np.array(
            [self.T_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)]
        )