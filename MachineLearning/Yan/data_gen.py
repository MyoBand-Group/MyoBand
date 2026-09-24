import json
import random

import matplotlib.pyplot as plt
import numpy as np


class Data:
    def __init__(self):
        self.wave = random.randint(1, 4)
        self.wform = []
        self.amp=1


def dataGen(steps=100,angle=0.1,amp=1):
    d=Data()
    d.amp=amp
    print(angle,amp)
    if d.wave == 1:
        for j in range(steps):
            d.wform.append(np.sin(angle*j)*amp) 
    if d.wave == 2:
        for j in range(steps):
            d.wform.append(np.cos(angle*j)*amp)
    if d.wave == 3:
        for j in range(steps):
            d.wform.append(np.tan(angle*j)*amp) 
    if d.wave == 4:
        for j in range(steps):
            d.wform.append((1 / np.tan(angle*j)) * amp)
    
    
    return d


data = []
for i in range(100):
    data.append(dataGen(angle=random.uniform(0.1, 1.0), amp=random.uniform(1, 5)))

serializable_data = [
    {"wave": item.wave, "wform": item.wform,"amp": item.amp}
    for item in data
]

with open("data.json", "w") as f:
    json.dump(serializable_data, f, indent=2)

plt.figure(figsize=(10, 6))
for item in data:
    plt.plot(item.wform, label=f"Wave {item.wave}")

plt.title("Generated Waveforms")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.savefig("wave_plot.png")
plt.show()

