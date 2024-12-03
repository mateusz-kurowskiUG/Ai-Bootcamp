import torch
import numpy as np
import time


ms = 10_000


mx_a = np.random.rand(ms, ms)
mx_b = np.random.rand(ms, ms)


start = time.time()

result = np.dot(mx_a, mx_b)

end = time.time()

time_diff = end - start

print(f"diff: {time_diff}")


tensor_a = torch.from_numpy(mx_a)
tensor_b = torch.from_numpy(mx_b)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = time.time()
torch_res = torch.matmul(tensor_a, tensor_b)
end = time.time()
torch.dot
diff2 = end - start
print(f"torch diff: {diff2}")
