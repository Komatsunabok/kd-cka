import torch
print(torch.version.cuda)  # 12.4と表示されればOK
print(torch.cuda.is_available())  # TrueならGPU認識OK
print(torch.cuda.get_device_name(0))  # GPU名を確認

"""
12.4
True
NVIDIA GeForce RTX 3060

! cuda driver support 12.9
! CUDA driver can support up to cuda runtime version 12.9 -> cuda runtime 12.4 is available 
"""

# try:
#     from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
#     from nvidia.dali.pipeline import pipeline_def
#     import nvidia.dali.types as types
#     import nvidia.dali.fn as fn
#     print("DALI is installed and working correctly!")
# except ImportError as e:
#     print("DALI is not installed or not working:", e)


# import nvidia.dali
# print(nvidia.dali.__version__)

# import multiprocessing

# print(multiprocessing.cpu_count())  # 利用可能なCPUコア数を表示