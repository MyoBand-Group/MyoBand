# Author: Nikola D. Lilov, 12 Aug 2026

import csv
import random
from sys import argv

NUM_POINTS = int(argv[1]) if len(argv) > 1 else 100
OUTPUT_FILE = argv[2] if len(argv) > 2 else "data.csv"
#V_s = 4/3*pi*r^3.   Note that the volume of the unit cube is V_c=8 - all possibilities of x,y,z in [-1,1].
RADIUS = 0.725 #V_s/V_c = 0.2
R2=RADIUS**2

with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.writer(file,delimiter=' ')

    avg1,avg2=0,0
    for _ in range(NUM_POINTS):
        in1 = random.uniform(-1.0, 1.0)
        in2 = random.uniform(-1.0, 1.0)
        in3 = random.uniform(-1.0, 1.0)

        
        # Outside sphere -> -1.0
        # Inside sphere -> +1.0
        if R2 > (in1**2 + in2**2 + in3**2):
            output1 = 1.0
        else:
            output1 = -1.0

        if 0.5 < abs(in1):
            output2 = 1.0
        else:
            output2 = -1.0

        avg1+=output1
        avg2+=output2


        writer.writerow([round(in1, 6),round(in2, 6),round(in3, 6)])
        writer.writerow([round(output1,3),round(output2,3)])

print(f"Generated {NUM_POINTS} points in {OUTPUT_FILE}.")
print(f"Average of output1: {avg1/NUM_POINTS}")
print(f"Average of output2: {avg2/NUM_POINTS}")