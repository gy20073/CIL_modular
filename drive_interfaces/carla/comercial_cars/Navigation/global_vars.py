def init():
    global diff_angle_global
    diff_angle_global = -2

def set(value):
    global diff_angle_global
    diff_angle_global = value
    #print("set to ", value)

def get():
    #print("read got value of ", diff_angle_global)
    return diff_angle_global