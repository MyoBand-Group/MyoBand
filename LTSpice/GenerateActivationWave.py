#Author:Viktor Atanasov, data: May 2026
import math
import random

# Settings
duration = 3.0       # 3 seconds total
sample_rate = 1000   # 1000 data points per second
dt = 1.0 / sample_rate

# Create files
with open("electrode5.txt", "w") as f1, open("electrode6.txt", "w") as f2:
    for i in range(int(duration * sample_rate)):
        t = i * dt
        
        # Base 60Hz wall noise (tiny, 0.05mV)
        noise = 0.00005 * math.sin(2 * math.pi * 60 * t)
        
        # Muscle contraction logic (Flexing between 1.0s and 2.0s)
        if 1.0 <= t <= 2.0:
            # Envelope to smooth the start and end of the flex
            envelope = math.sin(math.pi * (t - 1.0)) 
            # Chaotic 100Hz-ish muscle firings
            muscle_ac = random.uniform(-0.0015, 0.0015) * math.sin(2 * math.pi * 90 * t)
            muscle_signal = envelope * muscle_ac
        else:
            muscle_signal = random.uniform(-0.00002, 0.00002) # Just resting tissue noise
            
        # Electrode 1 sees the signal + noise
        v1 = muscle_signal + noise + 2.5
        # Electrode 2 sees a slightly different angle of the muscle + the SAME noise
        v2 = (muscle_signal * -0.5) + noise + 2.5
        
        # Write to LTspice PWL format (Time Voltage)
        f1.write(f"{t:.4f}\t{v1:.6f}\n")
        f2.write(f"{t:.4f}\t{v2:.6f}\n")

print("Files 'electrode1.txt' and 'electrode2.txt' created successfully!")