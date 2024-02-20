import re
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt

bank_pos = np.linspace(0, 228, 400)
k_eff = []

for step in bank_pos:
    f = open("C1.inp", "w")

    for line in open("./C1B1b1.inp"):
        match = re.search("bank8", line)

        if match:
            line = re.sub("bank8", str(step), line)

        f.write(line)

    f.close()

    subprocess.run("[PLACE PATH TO PARCS] C1.inp", shell=True)

    for line in open("./C1.parcs_out"):
        match = re.search("K-Effective:\\s+\\d+.\\d+", line)

        if match:
            k_eff.append(float(re.findall(r"\\d+.\\d+", match.group(0))[0]))

k_eff = np.array(k_eff)

rod_worth = (k_eff - 1) / k_eff

plt.plot(bank_pos, rod_worth)
plt.xlabel("Bank 8 Position")
plt.ylabel("Rod Worth")
plt.savefig("rod_worth.png")

df = pd.DataFrame({"bank8_pos": bank_pos, "rod_worth": rod_worth})
df.to_csv("rod_worth.csv", index=False)
