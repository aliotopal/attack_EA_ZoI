import numpy as np
import random

roi = np.load("extracted_numbers.npy")
# print(roi)

no_of_regs = random.randrange(0, len(roi))
#
print(no_of_regs)
#
reg_idx = random.sample(range(0, len(roi)), no_of_regs)

print(reg_idx)
#
for x in reg_idx:
    reg = roi[x]
    print(reg)
