import matplotlib.pyplot as plt
import numpy as np
import HelmholtzData as Sim

# This is the graphing file
# I split up the simulation and graphing files to make it easier to edit

#######################################################################################################################
# Here I name all the functions and objects used for graphing
xCoil, yCoil, zCoil = Sim.xCoil, Sim.yCoil, Sim.zCoil
b_field = Sim.b_field
print_b = Sim.print_b_field
origin_vect = b_field([[0, 0, 0]])  # useful for scaling the plots
origin_mag = np.linalg.norm(origin_vect)


def upper_bound(data):  # gets rid of the super long vectors in the 3d vector field
    limited_data = data
    for i in range(np.shape(data)[0]):
        if np.linalg.norm(data[i]) > origin_mag * 2:
            limited_data[i] = data[i] / np.linalg.norm(data[i]) * origin_mag * 2
    return limited_data


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
    # the functions I defined in HelmholtzData take inputs with the shape (n, 3). So, I have to reshape the mesh into
    # a list of vectors corresponding to every point in the mesh. the output is the same shape - a list of vectors
    # corresponding to the b-field at every point. each the position of each output vector lines up with the input,
    # so I can reshape the output back into a mesh for graphing

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
#######################################################################################################################


# actual graphing begins here
print_b(np.array([[0, 0, 0], [-0.01, 0, 0], [0.01, 0, 0], [0, -0.15, 0], [0, 0.15, 0], [-0.01, -0.15, -0.1],
                  [0, 0.15, 0.01], [0, 0.15, -0.01]]))  # prints b-field at all these points

# this is the graph of the coils themselves
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

# returns the 'list of vectors' format to the 'meshgrid' style used in graphing
x_reshaped = np.reshape(b_limited[:, 0:1], np.shape(mesh)[3:0:-1])
y_reshaped = np.reshape(b_limited[:, 1:2], np.shape(mesh)[3:0:-1])
z_reshaped = np.reshape(b_limited[:, 2:3], np.shape(mesh)[3:0:-1])
b_reshaped = np.swapaxes(np.array([x_reshaped, y_reshaped, z_reshaped]), 1, 3)  # like .T in another direction

x, y, z = mesh
u, v, w = b_reshaped

ax.quiver(x, y, z, u, v, w, length=origin_mag / 60)  # this is the vector field

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

# color plots below
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
