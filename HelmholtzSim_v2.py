import matplotlib.pyplot as plt
import numpy as np

# Kevin's simulation of the magnetic field due to three sets of rectangular Helmholtz coils centered at the origin
# for general troubleshooting, if a vector [input] to a function is not working, try using [[input]] because these
# functions like having 2d inputs

# This file is the Graphing and Data files combined. It is more compact but unwieldy

k_m = 10 ** -7  # magnetic constant / 4 pi
background_field = np.array([-0.05, -0.05, 0])  # this is in Gauss

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


class Loop:
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


class Coil:  # a coil is two loops

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


def b_field(point):
    arr_point = np.asarray(point)
    vect_total = xCoil.b_field_coil(arr_point) + yCoil.b_field_coil(arr_point) + zCoil.b_field_coil(arr_point)
    return vect_total + background_field


def print_b_field(point):
    data = b_field(point)
    for i in range(np.shape(data)[0]):
        print('field at ' + str(point[i]) + ' is ' + str(data[i]))


print_b_field(np.array([[0, 0, 0], [-0.01, 0, 0], [0.01, 0, 0], [0, -0.15, 0], [0, 0.15, 0], [-0.01, -0.15, -0.1],
                        [0, 0.15, 0.01], [0, 0.15, -0.01]]))

# all variables and code below here are for the purpose of graphing
origin_vect = b_field([[0, 0, 0]])  # useful for scaling the plots
origin_mag = np.linalg.norm(origin_vect)


def upper_bound(data):  # gets rid of the super long vectors in the 3d vector field
    limited_data = data
    for i in range(np.shape(data)[0]):
        if np.linalg.norm(data[i]) > origin_mag * 2:
            limited_data[i] = data[i] / np.linalg.norm(data[i]) * origin_mag * 2
    return limited_data


# actual graphing begins here. first I graph the coils themselves
ax = plt.axes(projection='3d')
xCoil.coil_plot()
yCoil.coil_plot()
zCoil.coil_plot()
plt.title('Helmholtz Coils')
ax.set_xlim3d(-0.8, 0.8)
ax.set_ylim3d(-1.1, 0.5)
ax.set_zlim3d(-0.8, 0.8)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()

# next, is the coils with the magnetic field vectors
ax = plt.axes(projection='3d')

mesh = np.asarray(np.meshgrid(np.arange(-0.6, 0.6, 1.2 / 10), np.arange(-0.85, 0.55, 1.4 / 10),
                              np.arange(-0.4, 0.4, 0.8 / 6)))
coords = np.reshape(mesh.T, (-1, 3))
b_coords = b_field(coords)
b_limited = upper_bound(b_coords)  # gets rid of super long vectors

x_reshaped = np.reshape(b_limited[:, 0:1], np.shape(mesh)[3:0:-1])  # returns the 'list of vectors' format to the
y_reshaped = np.reshape(b_limited[:, 1:2], np.shape(mesh)[3:0:-1])  # meshgrid format used in graphing
z_reshaped = np.reshape(b_limited[:, 2:3], np.shape(mesh)[3:0:-1])
b_reshaped = np.swapaxes(np.array([x_reshaped, y_reshaped, z_reshaped]), 1, 3)

x, y, z = mesh
u, v, w = b_reshaped

ax.quiver(x, y, z, u, v, w, length=origin_mag / 60)  # makes the field at the origin the same length every run

xCoil.coil_plot()
yCoil.coil_plot()
zCoil.coil_plot()

plt.title('Magnetic Field')
ax.set_xlim3d(-0.8, 0.8)
ax.set_ylim3d(-1.1, 0.5)
ax.set_zlim3d(-0.8, 0.8)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()

# these are one dimensional plots along the y-axis, in this case the path of the beam through the chamber
y_range = np.asarray([[i] for i in np.arange(-0.15, 0.15, 0.005)])
y_coords = np.insert(y_range, 0, 0, axis=1)
b_y_coords0 = b_field(np.insert(y_coords, 2, -0.01, axis=1))  # these are a list of vectors parallel to the y axis,
b_y_coords1 = b_field(np.insert(y_coords, 2, 0, axis=1))  # with a slight offset (-0.01, 0, and 0.01) in the
b_y_coords2 = b_field(np.insert(y_coords, 2, 0.01, axis=1))  # z-direction

plt.plot(y_range, b_y_coords0[:, 1:2], 'r', label='z = -0.01')
plt.plot(y_range, b_y_coords1[:, 1:2], 'g', label='z = 0.00')
plt.plot(y_range, b_y_coords2[:, 1:2], 'b', label='z = 0.01')
plt.title('y component of b-field vs y displacement')
plt.xlabel("Distance (m)")
plt.ylabel("b-field (gauss)")
plt.legend()
plt.tick_params(axis='both', direction='in', top='true', right='true')
plt.show()

plt.plot(y_range, b_y_coords0[:, 2:3], 'r', label='z = -0.01')
plt.plot(y_range, b_y_coords1[:, 2:3], 'g', label='z = 0.00')
plt.plot(y_range, b_y_coords2[:, 2:3], 'b', label='z = 0.01')
plt.title('z component of b-field vs y displacement')
plt.xlabel("Distance (m)")
plt.ylabel("b-field (gauss)")
plt.legend()
plt.tick_params(axis='both', direction='in', top='true', right='true')
plt.show()

plt.plot(y_range, b_y_coords0[:, 1:2] / b_y_coords0[:, 2:3], 'r', label='z = -0.01')
plt.plot(y_range, b_y_coords1[:, 1:2] / b_y_coords1[:, 2:3], 'g', label='z = 0.00')
plt.plot(y_range, b_y_coords2[:, 1:2] / b_y_coords2[:, 2:3], 'b', label='z = 0.01')
plt.title('y / z component of b-field vs y displacement')
plt.xlabel("Distance (m)")
plt.ylabel("b-field (gauss)")
plt.legend()
plt.tick_params(axis='both', direction='in', top='true', right='true')
plt.show()

# these are global variables for the color plots. there is probably a better way to do this
cx, cz, cy, b_reshaped_xz, b_reshaped_yz = np.array([0, 0, 0, 0, 0])


def color_domain(x_lim, y_lim, z_lim, gridnumber):  # this function changes the domain of the color maps
    global cx, cy, cz, b_reshaped_xz, b_reshaped_yz
    mesh_xz = np.asarray(np.meshgrid(np.arange(-x_lim, x_lim, x_lim / gridnumber),
                                     np.arange(-z_lim, z_lim, z_lim / gridnumber)))
    cx, cz = mesh_xz
    coords_xz = np.reshape(mesh_xz.T, (-1, 2))
    coords_xz_pad = np.insert(coords_xz, 1, 0, axis=1)
    b_coords_xz = b_field(coords_xz_pad)
    x_reshaped_xz = np.reshape(b_coords_xz[:, 0:1], np.shape(mesh_xz)[2:0:-1])
    y_reshaped_xz = np.reshape(b_coords_xz[:, 1:2], np.shape(mesh_xz)[2:0:-1])
    z_reshaped_xz = np.reshape(b_coords_xz[:, 2:3], np.shape(mesh_xz)[2:0:-1])
    b_reshaped_xz = np.swapaxes(np.array([x_reshaped_xz, y_reshaped_xz, z_reshaped_xz]), 1, 2)

    mesh_yz = np.asarray(np.meshgrid(np.arange(-y_lim, y_lim, y_lim / gridnumber),
                                     np.arange(-z_lim, z_lim, z_lim / gridnumber)))
    cy, dummy = mesh_yz
    coords_yz = np.reshape(mesh_yz.T, (-1, 2))
    coords_yz_pad = np.insert(coords_yz, 0, 0, axis=1)
    b_coords_yz = b_field(coords_yz_pad)
    x_reshaped_yz = np.reshape(b_coords_yz[:, 0:1], np.shape(mesh_yz)[2:0:-1])
    y_reshaped_yz = np.reshape(b_coords_yz[:, 1:2], np.shape(mesh_yz)[2:0:-1])
    z_reshaped_yz = np.reshape(b_coords_yz[:, 2:3], np.shape(mesh_yz)[2:0:-1])
    b_reshaped_yz = np.swapaxes(np.array([x_reshaped_yz, y_reshaped_yz, z_reshaped_yz]), 1, 2)


# this function give a color plot of the x, y, z components of the b field over the xz and yz planes
# you can call the dimensions you want to look at, or the scaling of the colors

def color_plot(lims, c_centers, color_range):
    # [lims] is the x, y, z viewing range. the range goes from -lim to +lim
    # [c_centers] is the center of the color map for the x, y, z components
    # color_range is the scaling of the color map
    b_xz_x, b_xz_y, b_xz_z = b_reshaped_xz
    b_yz_x, b_yz_y, b_yz_z = b_reshaped_yz
    domains_list = [cx, cy, cz]
    b_fields_list = [[b_xz_x, b_xz_y, b_xz_z], [b_yz_x, b_yz_y, b_yz_z]]
    titles_list = ['x', 'y', 'z']
    plt_num = 1
    fig = plt.figure(figsize=(12.5, 7.5))
    fig.subplots_adjust(wspace=0.5, hspace=0.4)
    for i in range(2):
        for j in range(3):
            plt.subplot(2, 3, plt_num)
            plt.pcolormesh(domains_list[i], cz, b_fields_list[i][j],
                           cmap=plt.cm.get_cmap('magma'), shading='auto',
                           vmin=c_centers[0][j] - color_range * 2,
                           vmax=c_centers[0][j] + color_range * 2)
            plt.axis([-lims[i], lims[i], -lims[2], lims[2]])
            plt.title(titles_list[j] + '-component on the ' + titles_list[i] + 'z plane', y=1.1)
            plt.colorbar()
            plt_num += 1
    plt.show()


x_bound = 0.8 * max(xCoil.length[0], yCoil.length[0], zCoil.length[0])  # automatically sets a bound for a color map
y_bound = 0.8 * max(xCoil.length[1], yCoil.length[1], zCoil.length[1])
z_bound = 0.8 * max(xCoil.length[2], yCoil.length[2], zCoil.length[2])

color_domain(x_bound, y_bound, z_bound, 24)  # sets the domain to the entire coil, with 48 subdivisions per axis

color_plot([x_bound, y_bound, z_bound], [[0, 0, 0]], origin_mag)
color_plot([x_bound, y_bound, z_bound], [[0, 0, 0]], origin_mag / 100)  # a more zoomed in color scale
color_plot([x_bound, y_bound, z_bound], origin_vect, origin_mag / 100)  # color scale centered at origin values

color_domain(0.01, 0.15, 0.01, 24)  # domain change to the viewing region

# the color scale for these are centered at the value of the b field @ (0, 0, 0)
color_plot([0.01, 0.15, 0.01], origin_vect, 0.1)
color_plot([0.01, 0.15, 0.01], origin_vect, 0.05)
color_plot([0.01, 0.15, 0.01], origin_vect, 0.025)
color_plot([0.01, 0.15, 0.01], origin_vect, 0.01)
