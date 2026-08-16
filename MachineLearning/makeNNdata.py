""" Author: Nikola D. Lilov, 12 Aug 2026

    Desc: Generates simple data for training a neural network. It gives
    three inputs - X, Y, Z coordinates in the unit cube - and two outputs.
    The first ouptut is whether or not the given coordinates correspond to
    a point in a set sphere (RADIUS = 0.725). The second output is even
    simpler - whether or not X is within [-0.5; +0.5] or not.

    Usage:      python makeNNdata.py [# of data points] [save_path] [give output? (Y/n)] [answer_path]
    Defaults:   python makeNNdata.py        250         dataset.txt         True            ans.txt
    
    ^ Explanation ^: How many data points to generate; Where to save the generated points;
            Should I also give the answers (is this for training or for inferencing);
            If not to the last one, where should I save the answers instead?
"""


import csv
import random
from sys import argv

def bool_(arg):
    if arg.lower() in ("false","f","no","n","0"):
        return False
    else:
        return True

NUM_POINTS = int(argv[1]) if len(argv) > 1 else 250
OUTPUT_FILE = argv[2] if len(argv) > 2 else "dataset.txt"
GIVE_OUTPUT = bool_(argv[3]) if len(argv) > 3 else True
ANS_FILE = argv[4] if len(argv) > 4 else "ans.txt"
#V_s = 4/3*pi*r^3.   Note that the volume of the unit cube is V_c=8 - all possibilities of x,y,z in [-1,1].
RADIUS = 0.725 #V_s/V_c = 0.2
R2=RADIUS**2
BOUNDRY = 0.5 #out2

files = {"file": open(OUTPUT_FILE, "w", newline="")}
if not GIVE_OUTPUT:
    files["ans"] = open(ANS_FILE, "w", newline="")

try:
    writer = csv.writer(files["file"], delimiter=' ')
    writer2 = csv.writer(files["ans"], delimiter=' ') if not GIVE_OUTPUT else None
    
    avg1,avg2=0,0
    for _ in range(NUM_POINTS):
        in1 = random.uniform(-1.0, 1.0)
        in2 = random.uniform(-1.0, 1.0)
        in3 = random.uniform(-1.0, 1.0)
        writer.writerow([round(in1, 15),round(in2, 15),round(in3, 15)])

        # Outside sphere -> -1.0
        # Inside sphere -> +1.0
        if R2 > (in1**2 + in2**2 + in3**2):
            out1 = 1.0
        else:
            out1 = -1.0

        if BOUNDRY < abs(in1):
            out2 = 1.0
        else:
            out2 = -1.0

        avg1+=out1
        avg2+=out2
        
        if GIVE_OUTPUT: writer.writerow([round(out1,3),round(out2,3)])
        else: writer2.writerow([round(out1,3),round(out2,3)])

    print(f"Generated {NUM_POINTS} points in {OUTPUT_FILE}.")
    if not GIVE_OUTPUT:
        print(f"No output included in {OUTPUT_FILE}. Instead it was stored in {ANS_FILE}.")
    print(f"Average of output1: {avg1/NUM_POINTS}")
    print(f"Average of output2: {avg2/NUM_POINTS}")

except IOError as e:
    print(f"Error opening files: {e}. Aborting...")
except Exception as e:
    print(f"Unexpected error: {e}. Aborting...")

finally:
    for f in files.values():
        f.close()
