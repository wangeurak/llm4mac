import os
import re


# write & read file
def check_file(path):
    if os.path.getsize(path) == 0:
        return 0
    else:
        return 1


def read_file(path, datatype):
    with open(path, "r") as f:
        data = f.read()
        data_list_raw = re.split(" ", data)
        del(data_list_raw[0])

        if datatype == 0 or datatype == 1:
            data_list = [int(x) for x in data_list_raw]
        else:
            data_list = [float(x) for x in data_list_raw]

    open(path, "w").close()
    if datatype == 0:
        return data_list
    else:
        return data_list[0]


def write_file(path, data):
    with open(path, "w") as f:
        f.write(' ' + str(data))


def add_file(path, data):
    with open(path, "a") as f:
        f.write(' ' + str(data))
