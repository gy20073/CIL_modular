import pickle, sys

# fake the speed, command, loc, ori
content = [[0.0, "d", ['debug_direct_pixel', 981, 659], [0, -1.0]]]

with open(sys.argv[1], "wb") as f:
    pickle.dump(content, f)
