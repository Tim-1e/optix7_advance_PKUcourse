import cv2
import numpy as np
import math
import os, sys

if len(sys.argv) != 4:
    print("usage: [ref_dir] [test_dir] [result_dir]")
    exit(0)

ref_dir = sys.argv[1]
test_dir = sys.argv[2]

ref = cv2.imread(ref_dir)
test = cv2.imread(test_dir)

print(f"ref: {ref.shape}")
print(f"test: {ref.shape}")

ref = ref.astype("float32")
test = test.astype("float32")

mse = np.mean((ref - test)**2)

# 这种算法对小的值有偏向
mape = np.mean(abs(ref - test)/(ref+1e-3)) * 100

print()
print("mse: {:.5f}".format(mse))
print("mape: {:.5f}%".format(mape))

with open("./diff.txt", "w") as f:
    f.write(f"ref: {sys.argv[1]}\n")
    f.write(f"test: {sys.argv[2]}\n")
    f.write("mse: {:.5f}\n".format(mse))
    f.write("mape: {:.5f}%\n".format(mape))


# 生成热度图
heatMap = None
heatMap = np.clip(1000*abs(ref - test)/(ref+1e-3), 0, 255)
heatMap = cv2.normalize(heatMap, heatMap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

# different color scales: 
# https://stackoverflow.com/questions/59478962/how-to-convert-a-grayscale-image-to-heatmap-image-with-python-opencv
heatMap = cv2.applyColorMap(heatMap, cv2.COLORMAP_JET)
cv2.imwrite(sys.argv[3], heatMap)



