import random
def create_map_with_path(size=4):
    lake = [['F' for _ in range(size)] for _ in range(size)]
    for i in range(1, size-1):
        for j in range(1, size-1):
            if random.random() < 0.3:  
                lake[i][j] = 'H'
    lake[0][0] = 'S'
    lake[size-1][size-1] = 'G'
    return lake

def create_map_with_no_path(size=4):
    lake = [['H' for _ in range(size)] for _ in range(size)]
    lake[0][0] = 'S'
    lake[size - 1][size - 1] = 'G'
    return lake
