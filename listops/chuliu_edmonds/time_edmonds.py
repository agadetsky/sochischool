import time
import torch
from listops.chuliu_edmonds.edmonds import *


n = 50
bs = 128
lengths = np.random.choice(np.arange(10, n), bs)
np.random.seed(42)
weights = np.zeros((bs, n, n))
for i in range(bs):
    w = np.random.rand(lengths[i], lengths[i])
    weights[i:, :lengths[i], :lengths[i]] = w
weights = torch.tensor(weights).to("cuda")

# Test chr' version.
start = time.time()
res_chr = edmonds_python(weights.to("cpu").numpy(), lengths, "chr")
print(f"Chris' version took: {time.time() - start}")

# Test nx version.
start = time.time()
res_nx = edmonds_python(weights.to("cpu").numpy(), lengths, "nx")
print(f"Nx version took: {time.time() - start}")

# C++ (cpu) version.
start = time.time()
res_cpp_cpu = edmonds_cpp_pytorch(weights.to("cpu"), torch.tensor(lengths))
print(f"C++ (cpu) version took: {time.time() - start}")

# C++ (gpu) version.
start = time.time()
res_cpp_gpu = edmonds_cpp_pytorch(weights, torch.tensor(lengths))
torch.cuda.synchronize()
print(f"C++ (gpu) version took: {time.time() - start}")

np.testing.assert_almost_equal(res_chr, res_nx)
np.testing.assert_almost_equal(res_nx, res_cpp_cpu)
np.testing.assert_almost_equal(res_cpp_cpu.numpy(), res_cpp_gpu.to("cpu").numpy())
