import matplotlib.pyplot as plt
import numpy as np

# Kevin's simulation of the magnetic field due to three sets of rectangular Helmholtz coils centered at the origin
# in general, if a vector [input] to a function is not working, try using [[input]] because these
# functions like having 2d inputs

# This is the number-crunching file with no graphing
# I split up the simulation and graphing files to make it easier to edit

k_m = 10 ** -7  # magnetic constant / 4 pi
background_field = np.array([-0.15, -0.06, 0.42])  # this is in Gauss

# useful dictionaries for calculations and organization. they help keep track of the direction of the positive current
# through each loop of wire

direction_dict = {'x': [0, 1, 2], 'y': [1, 2, 0], 'z': [2, 0, 1]}
side_dict = {'+': 1, '-': -1}
basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # basis vectors


class Wire:  # not used in the main code, but if you want to simulate a single wire, use this
    def __init__(self, start, finish, current, para_ax):
        self.start = np.array(start)
        self.finish = np.array(finish)
        self.current = current
        self.para_ax = para_ax  # parallel vector to the wire

    def b_field_wire(self, point):  # for the single wire, start and finish are vectors
        wire = self.start - self.finish

        # these variables simplify calculation, so that I do not need to project any vectors
        # wire_orth lets me find the perpendicular distance from any point to the wire
        # para_ax lets me find the distance between the point projected onto the wire and the wire edge

        # read the comments in the loop class for more detail

        para_ax = self.para_ax
        wire_orth = np.array([1, 1, 1])
        wire_orth[para_ax] = 0

        start_dist = self.start - point
        finish_dist = self.finish - point
        perp_vect = start_dist * wire_orth

        start_dist_norm = np.sqrt(start_dist[:, 0:1] ** 2 + start_dist[:, 1:2] ** 2 + start_dist[:, 2:3] ** 2)
        finish_dist_norm = np.sqrt(finish_dist[:, 0:1] ** 2 + finish_dist[:, 1:2] ** 2 + finish_dist[:, 2:3] ** 2)
        perp_dist_norm = np.sqrt(perp_vect[:, 0:1] ** 2 + perp_vect[:, 1:2] ** 2 + perp_vect[:, 2:3] ** 2)

        points_para_ax = point[:, para_ax:para_ax + 1]

        # val is the magnitude of the b field due to one wire
        # it is just the solved biot-savart integral for straight wires

        sin_theta2 = (self.finish[para_ax] - points_para_ax) / finish_dist_norm
        sin_theta1 = (self.start[para_ax] - points_para_ax) / start_dist_norm

        val = k_m * self.current / perp_dist_norm * np.abs(sin_theta1 - sin_theta2)

        # drct is the direction of the b field due to one wire
        # it is the cross product of the wire and the perpendicular vector from the wire to the point

        drct = np.cross(wire, perp_vect)
        drct_norm = np.array([[np.linalg.norm(i)] for i in drct])
        vect = drct / drct_norm * val * 10000  # converts tesla to gauss
        return vect


class Loop:  # a rectangle of wires, one face of a rectangular prism
    def __init__(self, center, length, direction, current, side):
        self.center = np.array(center)
        self.length = np.array(length)
        self.current = current
        self.direction = direction
        self.axis = direction_dict[direction][0]
        self.side = np.array([1, 1, 1])
        self.side[self.axis] = side_dict[side]  # this changes the sign of the direction coordinate
        self.vertices = []
        self.vertices.append(self.center + self.length / 2 * self.side)  # vertex 0
        self.vertices.append(self.vertices[0] - self.length * basis[direction_dict[direction][1]])  # vertex 1
        self.vertices.append(self.vertices[1] - self.length * basis[direction_dict[direction][2]])  # vertex 2
        self.vertices.append(self.vertices[2] + self.length * basis[direction_dict[direction][1]])  # vertex 3

    def b_field_wire(self, start, finish, point):
        # this is the meat of the code. It calculates the b field due to a single side of a rectangular loop ("a wire")
        # for the loop, start and finish are integers - the vertex numbers for the list of [vertices] (self.vertices)

        wire = self.vertices[start] - self.vertices[finish]  # the wire direction

        # these variables below simplify calculation, so that I do not need to project any vectors
        # para_ax lets me find the distance between the point projected onto the wire and the wire edge
        # wire_orth lets me find the perpendicular distance from any point to the wire

        para_ax = direction_dict[self.direction][start % 2 + 1]  # just an integer, (0, 1, 2) corresponds to (x, y, z)
        wire_orth = np.array([1, 1, 1])
        wire_orth[para_ax] = 0

        start_dist = self.vertices[start] - point
        finish_dist = self.vertices[finish] - point
        perp_vect = start_dist * wire_orth

        # the [point] variable must be a 2d matrix of size (x, 3). it is a list of points
        # these below calculate the magnitude of each vector in the above arrays. they are used in the final calculation

        start_dist_norm = np.sqrt(start_dist[:, 0:1] ** 2 + start_dist[:, 1:2] ** 2 + start_dist[:, 2:3] ** 2)
        finish_dist_norm = np.sqrt(finish_dist[:, 0:1] ** 2 + finish_dist[:, 1:2] ** 2 + finish_dist[:, 2:3] ** 2)
        perp_dist_norm = np.sqrt(perp_vect[:, 0:1] ** 2 + perp_vect[:, 1:2] ** 2 + perp_vect[:, 2:3] ** 2)

        points_para_ax = point[:, para_ax:para_ax + 1]

        # val is the magnitude of the b field due to a wire
        # it is just the solved biot-savart integral for straight wires

        sin_theta2 = (self.vertices[finish][para_ax] - points_para_ax) / finish_dist_norm
        sin_theta1 = (self.vertices[start][para_ax] - points_para_ax) / start_dist_norm

        val = k_m * self.current / perp_dist_norm * np.abs(sin_theta1 - sin_theta2)

        # drct is the direction of the b field due to a wire
        # it is the cross product of the wire and the perpendicular vector from the wire to the point

        drct = np.cross(wire, perp_vect)
        drct_norm = np.sqrt(drct[:, 0:1] ** 2 + drct[:, 1:2] ** 2 + drct[:, 2:3] ** 2)
        vect = drct / drct_norm * val * 10000  # the factor of 10000 converts tesla to gauss

        return vect

    def wire_plot(self, start, finish):  # plots a single side of the loop
        plt.plot([self.vertices[start][0], self.vertices[finish][0]],
                 [self.vertices[start][1], self.vertices[finish][1]],
                 [self.vertices[start][2], self.vertices[finish][2]], 'k')


class Coil:  # a coil is two loops, defined by a rectangular prism with a center vector and a length, and a direction
    # center: a position vector
    # length: a vector with [x_length, y_length, z_length]. center is similar
    # direction: where you want the b-field to point, 'x', 'y', or 'z'
    # current: the total current * turns for one side of the coil
    # (positive current will create a b-field in the positive direction)
    def __init__(self, center, length, direction, current):
        self.center = np.array(center)
        self.length = np.array(length)
        self.direction = direction
        self.current = current
        self.p_loop = Loop(center, length, direction, current, '+')
        self.n_loop = Loop(center, length, direction, current, '-')

    def b_field_coil(self, point):  # adds up the field due to each wire in each loop of the coil
        vect_coil = 0
        for i in range(4):
            vect_coil = vect_coil + self.p_loop.b_field_wire(i, (i + 1) % 4, point) \
                        + self.n_loop.b_field_wire(i, (i + 1) % 4, point)
        return vect_coil

    def coil_plot(self):  # plots the coil
        for i in range(4):
            self.p_loop.wire_plot(i, (i + 1) % 4)
            self.n_loop.wire_plot(i, (i + 1) % 4)


##################################################################################################################
# Here is where I define my coils. Each coil has four parameters: [center] point, [length] of sides, direction, and
# current. Positive current is defined as the current that produces a field in the positive x/y/z direction.
# the origin (0, 0, 0) is treated as the center of the observation region. if the coils are centered, the
# [center] parameter should be [0, 0, 0]

xCoil = Coil([0, 0, 0], [0.195, 0.385, 0.17], 'x', 0.9)
yCoil = Coil([0, 0, 0], [0.28, 0.42, 0.28], 'y', 2)
zCoil = Coil([0, -0.15, 0], [0.8, 1.06, 0.515], 'z', 115)
##################################################################################################################


def b_field(point):  # sums up the b_field due to every coil
    arr_point = np.asarray(point)
    vect_total = xCoil.b_field_coil(arr_point) + yCoil.b_field_coil(arr_point) + zCoil.b_field_coil(arr_point)
    return vect_total + background_field


def print_b_field(point):
    data = b_field(point)
    for i in range(np.shape(data)[0]):
        print('field at ' + str(point[i]) + ' is ' + str(data[i]))
