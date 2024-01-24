import numpy
import torch
import torch.nn.functional as F
import librosa
# 创建一个示例张量
tensor = torch.tensor([1, 2, 3], dtype=torch.float32)

# 使用torch.nn.functional.normalize对张量进行标准化
normalized_tensor = F.normalize(tensor, dim=0)

# 打印原始张量和标准化后的张量
print("原始张量：", tensor)
print("标准化后的张量：", normalized_tensor)


numpy_arr = numpy.array([1, 2, 3])

normalized_numpy_arr = librosa.util.normalize(numpy_arr)
normalized_tensor = torch.tensor(normalized_tensor, dtype=torch.float32)
print("原始张量：", numpy_arr)
print("标准化后的张量：", normalized_tensor)
