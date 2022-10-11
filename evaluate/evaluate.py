import cv2
import numpy as np
import math

ref_dir = "./1.png"
test_dir = "./2.png"

ref = cv2.imread(ref_dir)
test = cv2.imread(test_dir)

print(f"ref: {ref.shape}")
print(f"test: {ref.shape}")

ref = ref.astype("float32")
test = test.astype("float32")

mse = np.mean((ref - test)**2)

# 这种算法对小的值有偏向
mape = np.mean(abs(ref - test)/(ref+1e-5)) * 100

print()
print("mse: {:.5f}".format(mse))
print("mape: {:.5f}%".format(mape))