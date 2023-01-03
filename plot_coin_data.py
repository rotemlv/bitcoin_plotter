import pandas as pd  # for storing and manipulating the data we get back
import numpy as np  # numerical python, I usually need this somewhere
import matplotlib.pyplot as plt  # for charts and such

from write_coin_data_to_file import write_data_to_file

# some constants
ONE_DAY_24HR_DATA = 480  # This fits to the data grabbed in grab_btc_data_to_file
_debug = False  # use to print matrices etc ...
# parameters for matrix - subject to change:
_total_data_size = ONE_DAY_24HR_DATA  # this must change given the number if data points in read file!
_sector_start, _sector_end = 0, _total_data_size  # sector_end must be <= total_data_size
_sector_size = 100  # not in use
_sector_jump_increment = 60  # an attempt to linearize segments of coin prices


# the basic story:
# we got a few dots: say (2,44599.97), representin' the price of btc in time '2',
# we want to find the best fit line for all points in a certain sector (say, time 0 to 10)
# we got points 0 to 10 -> (a1,b1),(a2,b2), ... , (a10,b10)
# using line formula we get y = ax + b
# by plugging the values for each point to x,y respectively, we get a matrix as such
# [ a1 1] [a]  =  [ b1]
# [ a2 1] [b]     [ b2]
# ...               ...
# [a10 1]         [ b10]
# findin' a,b (the line coefficients) is done by the following algebraic calculation:
# let matrix be A and vector be x,b respectively.
# Ax = b -> (A^t)Ax = (A^t)b -> vector a1,a2,... is independent of vector 1,1,..,1 as it represents time,
# hence we can invert (A^t)A.
# -> x = ((A^t)A)^(-1)(A^t)b
# and we got the answer
# the end :)

# TODO: is it even sensible ??? V
# say we don't care about the constant of the line, y = ax + b without the 'b'
# same process but ignore the 1's:
# [ a1 ] [a]  =  [ b1]
# [ a2 ]         [ b2]
# ...               ...
# [a10 ]         [ b10]
# which means we can move [a] (a scalar), or just solve the same way:
# Ax = b -> AtAx = Atb -> x = (AtA)^(-1)(At)b -> AtA^(-1) is just 1/(|A|^2) which we mark as |A|^(-2)
# getting: x = |A|^(-2)(At)b where (At)(b) is also a scalar (dot product between A and b)
# so, simplifying: x = (|A|^(-2))*dot(A,b), where |A| = sqrt(dot(A,A))
# another way: x = (1/dot(A,A))*dot(A,b) which means a = x0 = x, and the line is y = ax (where this x is not a vector)

# matrices
def print_matrix(matrix):
    """(my) Standard print for float matrix"""
    for row in matrix:
        print("[", end="")
        for i, element in enumerate(row):
            print("%3g," % element, end="") if i != len(row) - 1 else print("%3g" % element, end="")
        print("]")


def read_file(file_name):
    """OLD - do not use"""
    tuple_example = []
    with open(file_name, 'r') as file:
        for line in file:
            test = line.split(',')
            tuple_example.append((test[0], int(test[1]), float(test[2][:-1])))
    return tuple(tuple_example)


def read_file_better(file_name):
    """Open file (place in this py file's location and read from it to a list of tuples"""
    # tuple_example = []
    date_times, int_times, prices = [], [], []
    with open(file_name, 'r') as file:
        for line in file:
            test = line.split(',')
            date_times.append(test[0])
            int_times.append(int(test[1]))
            prices.append(float(test[2][:-1]))
    return {'dates': date_times, 'times': int_times, 'prices': prices}


def separate_data_x_y_diff(un_arranged_data):
    """Return the data dict separated to 3 parts: times, prices, difference.
    Where: times = market-close time (vector), prices = close-price (vector), difference=time-delta (scalar).
    *vector is currently a regular python list."""
    # start index with 0, increment by diff
    # changed if len(data) < 2 to this:
    if len(un_arranged_data['times']) < 2:
        raise Exception(f'Cannot operate on a single point of data {un_arranged_data} in normalize_data()')
    x_axis, y_axis = [], []
    diff = un_arranged_data['times'][-1] - un_arranged_data['times'][-2]
    for i in range(0, len(un_arranged_data['times'])):
        x_axis.append(i)
        y_axis.append(un_arranged_data['prices'][i])
    return x_axis, y_axis, diff  # diff is time delta


def plot_graphs(data_to_plot, coefficient_a, coefficient_b, diff):
    """Plot a graph (currently representing crypto prices) along the best fit line
    for specific sector of data (sectors set in main and global for now)."""
    # plot the thing
    plt.xlabel(f"Prices from {filename}")
    # plot the line
    ls = [(str(pd.to_datetime(data_to_plot['times'][_sector_start] + i * diff, unit='ms')),
           coefficient_a * i + coefficient_b) for i in range(_sector_start, _sector_end)]
    plt.plot([e[0] for e in ls], [e[1] for e in ls])
    plt.show()


def plot_graphs_2(data_to_plot, coefficient_a, coefficient_b, diff):
    """A faster plot not using the pandas time conversion function"""
    # plot the thing

    plt.xlabel(f"Prices from {filename}")
    # plot the line
    # data['times'][sector_start] is start value, diff is diff :)
    ls = [(data_to_plot['times'][_sector_start] + (i - _sector_start) * diff,
           coefficient_a * i + coefficient_b) for i in range(_sector_start, _sector_end)]
    plt.plot([e[0] for e in ls], [e[1] for e in ls])
    plt.show()


def plot_coin_prices_without_show(data_to_plot):
    # plot the thing
    plt.xlabel(f"Prices from {filename}")
    plt.plot(data_to_plot['times'], data_to_plot['prices'])


def plot_lines_without_show(data_to_plot, coefficient_a, coefficient_b, diff, start, end):
    # plot the thing
    plt.xlabel(f"Prices from {filename}")

    # plot the line
    # data['times'][sector_start] is start value, diff is diff :)
    ls = [(data_to_plot['times'][start] + (i - start) * diff, coefficient_a * i + coefficient_b)
          for i in range(start, end)]

    # the line width stuff is used to make the smaller changes thicker
    plt.plot([e[0] for e in ls], [e[1] for e in ls],
             linewidth=0.5 + 2 * (_total_data_size - (end - start)) / (0.7 * _total_data_size))


def plot_segments(data_to_plot, normalized_data, start=_sector_start, edge=_sector_end):
    # TODO: this function does way too much
    while edge <= 500:
        # refactored
        x = get_current_vector(edge, normalized_data, start)

        a = float(list(x[0])[0])  # get the coefficient out of the vector (sure it can be done better)
        b = float(list(x[1])[0])

        plot_lines_without_show(data_to_plot, a, b, normalized_data[2], start, edge)
        start += _sector_jump_increment
        edge += _sector_jump_increment

    plot_coin_prices_without_show(data_to_plot)
    plt.show()


def get_current_vector(edge, normalized_data, start):
    matrix = []
    vector = []
    for i in range(start, edge):
        matrix.append([normalized_data[0][i], 1])
        vector.append([normalized_data[1][i]])
    matrix_a = np.asmatrix(matrix)  # turn matrix to a numpy matrix
    vector_b = np.asmatrix(vector)
    matrix_a_transpose_dot_a = (matrix_a.transpose() @ matrix_a)  # create required matrices (can be shortened)
    matrix_inverse_a_t_a = np.linalg.inv(matrix_a_transpose_dot_a)
    x = matrix_inverse_a_t_a @ matrix_a.transpose() @ vector_b  # find the line coefficients (y = x0*t+x1)
    if _debug:
        print_matrix(matrix)
        print_matrix(vector)
        print(x)
    return x


def show_lines_of_diff_lengths(original_data, normalized_data, start, end, inc=100):
    # TODO: ditto
    while start < end:
        matrix = []
        vector = []
        for i in range(start, end):
            matrix.append([normalized_data[0][i], 1])
            vector.append([normalized_data[1][i]])

        mat_a = np.asmatrix(matrix)  # turn matrix to a numpy matrix
        vec_b = np.asmatrix(vector)
        mat_a_t_a = (mat_a.transpose() @ mat_a)  # create required matrices (can be shortened)
        mat_a_t_a_inverse = np.linalg.inv(mat_a_t_a)
        vec_x = mat_a_t_a_inverse @ mat_a.transpose() @ vec_b  # find the line coefficients (y = x0*t+x1)

        if _debug:
            print_matrix(matrix)
            print_matrix(vector)
            print(vec_x)

        a = float(list(vec_x[0])[0])  # get the coefficient out of the vector (yeah sure it can be done better)
        b = float(list(vec_x[1])[0])
        plot_lines_without_show(original_data, a, b, normalized_data[2], start, end)
        start += inc

    plot_coin_prices_without_show(original_data)
    plt.show()


def make_line_without_constant(original_data, normalized_data, st, en, inc=100):
    # TODO: This is logically faulty, leave this here until you can thoroughly explain why
    # explanation so far:
    # In order to get a linear approx for a line, we need two pieces of information: slope and a point.
    # without the constant we only got the slope, which only allows us to get the difference in the 1st dimension
    # (time), which is why this result always increases.
    # -> we only gain the information contained in the times vector (1st col of the matrix)
    # while ignoring the second column.
    # In conclusion: when trying to re-invent linear algebra, prepare for failure.
    while st < en:
        matrix = []  # this matrix is a vector here
        vector = []
        for i in range(st, en):
            matrix.append([normalized_data[0][i]])
            vector.append([normalized_data[1][i]])

        # another way: x = (1/dot(A,A))*dot(A,b)
        mat_a = np.asmatrix(matrix)  # turn matrix to a numpy matrix
        vec_b = np.asmatrix(vector)
        a_t_a = (mat_a.transpose() @ mat_a)  # create required matrices (can be shortened)
        a_t_a_inv = np.linalg.inv(a_t_a)
        vec_x = a_t_a_inv @ mat_a.transpose() @ vec_b  # find the line coefficients (y = x0*t+x1)

        # this is x
        # x = (1/(A.transpose()@A)) * (A.transpose() @ b)
        if _debug:
            print_matrix(matrix)
            print_matrix(vector)
            print(vec_x)
        a = float(list(vec_x[0])[0])  # get the coefficient out of the vector (sure it can be done better)
        # b = float(list(x[1])[0])

        plot_lines_without_show(original_data, a, 0, normalized_data[2], st, en)
        st += inc

    plot_coin_prices_without_show(original_data)
    plt.show()


if __name__ == '__main__':
    filename = write_data_to_file()

    # check validity of global variables
    with open(filename, 'r') as tmp_open:
        line_count = tmp_open.read().count('\n')
        if line_count != _total_data_size or _sector_end > _total_data_size:
            raise Exception(f'Trying to read {_total_data_size} data points all the way to {_sector_end} from a file with'
                            f'{line_count} data points')
    data = read_file_better(filename)

    data_x = separate_data_x_y_diff(data)
    # make_line_without_constant(data, data_x)
    show_lines_of_diff_lengths(data, data_x, _sector_start, _sector_end, _sector_jump_increment)

