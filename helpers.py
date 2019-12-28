# get fist position of a number in a list
def get_id_of_num_in_list(list, num):
    for i in range(len(list)):
        if list[i] == num:
            return i
    return -1