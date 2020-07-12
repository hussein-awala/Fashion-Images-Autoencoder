def get_pool_sizes(image_shape):
    width = image_shape[0]
    height = image_shape[1]
    kernel_sizes = []
    while True:
        if width % 2 == 0 and height % 2 == 0:
            kernel_sizes.append(2)
            width /= 2
            height /= 2
        elif width % 3 == 0 and height % 3 == 0:
            kernel_sizes.append(3)
            width /= 3
            height /= 3
        elif width % 5 == 0 and height % 5 == 0:
            kernel_sizes.append(5)
            width /= 5
            height /= 5
        elif width % 7 == 0 and height % 7 == 0:
            kernel_sizes.append(7)
            width /= 7
            height /= 7
        else:
            break
    return kernel_sizes


def find_shape(image_shape, vector_size, filter_size):
    factor = image_shape[0] / image_shape[1]
    goal = vector_size / filter_size
    i = 1
    while True:
        res = i * i * factor
        if res == goal:
            break
        if res > goal:
            print("error")
            return None
        i += 1
    shape = (int(i * factor), i, filter_size)
    return shape
