# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:35:18 2021

@author: Sterling Engle
@uid: 904341227

Developed and tested only on Windows 10 under Python 3.8.5.

GPUmodelALBERTqa.py is a RTX 3070 laptop GPU performance model of
ALBERT NLP pre-trained in SQUAD 2.0 to answer questions:
"ktrapeznikov/albert-xlarge-v2-squad-v2" to investigate performance for
machine reading comprehension of the California Penal Code.

These parameters may be specified via command line arguments:

usage: python GPUmodelALBERTqa.py [-h] [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        log output file path

References:
[1] The Hugging Face Team, "transformers / examples / question-answering",
    https://github.com/huggingface/transformers/tree/master/examples/question-answering
[2] Cloudera Fast Forward Labs, "Evaluating QA: Metrics, Predictions, and
    the Null Response",
    https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#F1
[2] Persson, Alladin, "Useful Tensor Operations",
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py
[3] PyTorch.org, "PyTorch Documentation",
    https://pytorch.org/docs/stable/index.html
[4] Stack Overflow, "Get total amount of free GPU memory and available using
    pytorch",
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
"""

import torch
import time
import argparse  # command-line parsing library
import numpy as np

# These arrays describe GPU performance specs, [GPU][specification]
g_RTX_3070_Laptop = 0
g_quantity = 1

# These are the specification array indices
s_nameId = 0
s_architecture = 1
s_chip = 2
s_clock_rate = 3
s_compute_capability_major = 4
s_compute_capability_minor = 5
s_fb_bus_width = 6
s_fbp_count = 7
s_global_l1_cache_supported = 8
s_global_memory_bus_width = 9
s_gpu_pci_ext_gpu_link_rate = 10
s_gpu_pci_ext_gpu_link_width = 11
s_l2_cache_size = 12
s_l2s_count = 13
s_limits_max_cta_per_sm = 14
s_local_l1_cache_supported = 15
s_max_access_policy_window_size = 16
s_max_block_dim_x = 17
s_max_block_dim_y = 18
s_max_block_dim_z = 19
s_max_blocks_per_multiprocessor = 20
s_max_gpu_frequency_khz = 21
s_max_grid_dim_x = 22
s_max_grid_dim_y = 23
s_max_grid_dim_z = 24
s_max_ipc_per_multiprocessor = 25
s_max_ipc_per_scheduler = 26
s_max_mem_frequency_khz = 27
s_max_persisting_l2_cache_size = 28
s_max_pitch = 29
s_max_registers_per_block = 30
s_max_registers_per_multiprocessor = 31
s_max_registers_per_thread = 32
s_max_shared_memory_per_block = 33
s_max_shared_memory_per_block_optin = 34
s_max_shared_memory_per_multiprocessor = 35
s_max_threads_per_block = 36
s_max_threads_per_multiprocessor = 37
s_max_warps_per_multiprocessor = 38
s_max_warps_per_scheduler = 39
s_multiprocessor_count = 40
s_num_l2s_per_fbp = 41
s_num_schedulers_per_multiprocessor = 42
s_ram_type = 43
s_reserved_shared_memory_per_block = 44
s_sass_level = 45
s_single_to_double_precision_perf_ratio = 46
s_sparse_cuda_array_supported = 47
s_total_constant_memory = 48
s_total_memory = 49
s_warp_size = 50
s_cuda_cores = 51
s_quantity = 52

gpu_names = ("NVIDIA GeForce RTX 3070 Laptop GPU")

# array describing GPU performance specs, [GPU][specification]
gpus = np.zeros([g_quantity, s_quantity], dtype="i8")
gpus[0, s_nameId] = g_RTX_3070_Laptop
gpus[g_RTX_3070_Laptop, s_architecture] = 368
gpus[g_RTX_3070_Laptop, s_chip] = 372
gpus[g_RTX_3070_Laptop, s_clock_rate] = 1620000
gpus[g_RTX_3070_Laptop, s_compute_capability_major] = 8
gpus[g_RTX_3070_Laptop, s_compute_capability_minor] = 6
gpus[g_RTX_3070_Laptop, s_fb_bus_width] = 256
gpus[g_RTX_3070_Laptop, s_fbp_count] = 4
gpus[g_RTX_3070_Laptop, s_global_l1_cache_supported] = True
gpus[g_RTX_3070_Laptop, s_global_memory_bus_width] = 256
gpus[g_RTX_3070_Laptop, s_gpu_pci_ext_gpu_link_rate] = 8000
gpus[g_RTX_3070_Laptop, s_gpu_pci_ext_gpu_link_width] = 16
gpus[g_RTX_3070_Laptop, s_l2_cache_size] = 4194304
gpus[g_RTX_3070_Laptop, s_l2s_count] = 32
gpus[g_RTX_3070_Laptop, s_limits_max_cta_per_sm] = 16
gpus[g_RTX_3070_Laptop, s_local_l1_cache_supported] = True
gpus[g_RTX_3070_Laptop, s_max_access_policy_window_size] = 134213632
gpus[g_RTX_3070_Laptop, s_max_block_dim_x] = 1024
gpus[g_RTX_3070_Laptop, s_max_block_dim_y] = 1024
gpus[g_RTX_3070_Laptop, s_max_block_dim_z] = 64
gpus[g_RTX_3070_Laptop, s_max_blocks_per_multiprocessor] = 16
gpus[g_RTX_3070_Laptop, s_max_gpu_frequency_khz] = 1620000
gpus[g_RTX_3070_Laptop, s_max_grid_dim_x] = 2147483647
gpus[g_RTX_3070_Laptop, s_max_grid_dim_y] = 65535
gpus[g_RTX_3070_Laptop, s_max_grid_dim_z] = 65535
gpus[g_RTX_3070_Laptop, s_max_ipc_per_multiprocessor] = 4
gpus[g_RTX_3070_Laptop, s_max_ipc_per_scheduler] = 1
gpus[g_RTX_3070_Laptop, s_max_mem_frequency_khz] = 7001000
gpus[g_RTX_3070_Laptop, s_max_persisting_l2_cache_size] = 3145728
gpus[g_RTX_3070_Laptop, s_max_pitch] = 2147483647
gpus[g_RTX_3070_Laptop, s_max_registers_per_block] = 65536
gpus[g_RTX_3070_Laptop, s_max_registers_per_multiprocessor] = 65536
gpus[g_RTX_3070_Laptop, s_max_registers_per_thread] = 255
gpus[g_RTX_3070_Laptop, s_max_shared_memory_per_block] = 49152
gpus[g_RTX_3070_Laptop, s_max_shared_memory_per_block_optin] = 101376
gpus[g_RTX_3070_Laptop, s_max_shared_memory_per_multiprocessor] = 102400
gpus[g_RTX_3070_Laptop, s_max_threads_per_block] = 1024
gpus[g_RTX_3070_Laptop, s_max_threads_per_multiprocessor] = 1536
gpus[g_RTX_3070_Laptop, s_max_warps_per_multiprocessor] = 48
gpus[g_RTX_3070_Laptop, s_max_warps_per_scheduler] = 12
gpus[g_RTX_3070_Laptop, s_multiprocessor_count] = 40
gpus[g_RTX_3070_Laptop, s_num_l2s_per_fbp] = 8
gpus[g_RTX_3070_Laptop, s_num_schedulers_per_multiprocessor] = 4
gpus[g_RTX_3070_Laptop, s_ram_type] = 17
gpus[g_RTX_3070_Laptop, s_reserved_shared_memory_per_block] = 1024
gpus[g_RTX_3070_Laptop, s_sass_level] = 8
gpus[g_RTX_3070_Laptop, s_single_to_double_precision_perf_ratio] = 32
gpus[g_RTX_3070_Laptop, s_sparse_cuda_array_supported] = True
gpus[g_RTX_3070_Laptop, s_total_constant_memory] = 65536
gpus[g_RTX_3070_Laptop, s_total_memory] = 8589934592
gpus[g_RTX_3070_Laptop, s_warp_size] = 32
gpus[g_RTX_3070_Laptop, s_cuda_cores] = 5120

# GPU specification names
gpu_specnames = ("name id",
                 "architecture",
                 "chip",
                 "clock rate",
                 "compute capability major",
                 "compute capability minor",
                 "fb bus width",
                 "fbp count",
                 "global l1 cache supported",
                 "global memory bus width",
                 "gpu pci ext gpu link rate",
                 "gpu pci ext gpu link width",
                 "l2 cache size",
                 "l2s count",
                 "limits max cta per sm",
                 "local l1 cache supported",
                 "max access policy window size",
                 "max block dim x",
                 "max block dim y",
                 "max block dim z",
                 "max blocks per multiprocessor",
                 "max gpu frequency khz",
                 "max grid dim x",
                 "max grid dim y",
                 "max grid dim z",
                 "max ipc per multiprocessor",
                 "max ipc per scheduler",
                 "max mem frequency khz",
                 "max persisting l2 cache size",
                 "max pitch",
                 "max registers per block",
                 "max registers per multiprocessor",
                 "max registers per thread",
                 "max shared memory per block",
                 "max shared memory per block optin",
                 "max shared memory per multiprocessor",
                 "max threads per block",
                 "max threads per multiprocessor",
                 "max warps per multiprocessor",
                 "max warps per scheduler",
                 "multiprocessor count",
                 "num l2s per fbp",
                 "num schedulers per multiprocessor",
                 "ram type",
                 "reserved shared memory per block",
                 "sass level",
                 "single to double precision perf ratio",
                 "sparse cuda array supported",
                 "total constant memory",
                 "total memory",
                 "warp size")

# These 3-D arrays contain [model][kernel][parameter]
# the only model is the ALBERT xlarge inference model for question answering
m_ALBERT_xlarge_qa = 0

# the m_ALBERT_xlarge_qa kernel rows are:
k_layers = 0  # number of layers
k_unrolled_elementwise_kernel_AddFunctor_large = 1
k_unrolled_elementwise_kernel_AddFunctor_medium = 2
k_unrolled_elementwise_kernel_AddFunctor_small = 3
k_unrolled_elementwise_kernel_AddFunctor_11 = 4
k_unrolled_elementwise_kernel_AddFunctor_685 = 5
k_unrolled_elementwise_kernel_copy_device_to_device_medium = 6
k_unrolled_elementwise_kernel_copy_device_to_device_small = 7
k_unrolled_elementwise_kernel_copy_device_to_device_0_11 = 8
k_vectorized_elementwise_kernel_GELU = 9
k_vectorized_elementwise_kernel_MulScalarFunctor = 10
k_vectorized_elementwise_kernel_AddFunctor = 11
k_vectorized_elementwise_kernel_AUnaryFunctor = 12
k_vectorized_elementwise_kernel_MulScalarFunctor_2 = 13
k_vectorized_elementwise_kernel_AddFunctor_5 = 14
k_vectorized_elementwise_kernel_BunaryFunctor = 15
k_indexSelectLargeIndex = 16
k_RowwiseMomentsCUDAKernel = 17
k_RowwiseMomentsCUDAKernel_8 = 18
k_LayerNormForwardCUDAKernel = 19
k_LayerNormForwardCUDAKernel_9 = 20
k_Kernel_xlarge = 21
k_Kernel_large = 22
k_Kernel_medium = 23
k_Kernel_small = 24
k_Kernel_10 = 25
k_Kernel_684 = 26
k_softmax_warp_forward = 27
k_reduce_kernel = 28

# the 32 per-kernel column parameters from NVIDIA Nsight Compute are:
p_quantity = 0  # number of times kernel used by model, or number of layers
p_function = 1
p_sub_function = 2
p_size = 3
p_first = 4
p_last = 5
p_increment = 6
p_function_call = 7
p_grid_size = 8
p_block_size = 9
p_threads = 10  # p_threads = p_grid_size * p_block_size
p_waves_per_sm = 11
p_registers_per_thread = 12
p_static_shared_memory_per_block = 13
p_dynamic_shared_memory_per_block = 14
p_driver_shared_memory_per_block = 15
p_shared_memory_configuration_size = 16
p_memory_throughput = 17
p_l1_hit_rate = 18
p_l2_hit_rate = 19
p_mem_pipes_busy_pct = 20
p_sol_sm_pct = 21
p_sol_memory_pct = 22
p_sol_l1_cache_pct = 23
p_sol_l2_cache_pct = 24
p_sol_dram_pct = 25
p_duration_usec = 26
p_elapsed_cycles = 27
p_sm_active_cycles = 28
p_sm_frequency = 29
p_dram_frequency = 30
p_threads_registers_per_thread_divided_by_duration = 31
p_avg_threads_x_registers_per_thread_each_usec = 32

param_names = (
               "quantity",
               "function",
               "sub function",
               "size",
               "first",
               "last",
               "increment",
               "function call",
               "grid size",
               "block size",
               "threads",  # threads = grid size * block size
               "waves per SM",
               "registers per thread",
               "static shared memory per block",
               "dynamic shared memory per block",
               "driver shared memory per block",
               "shared memory configuration size",
               "memory throughput",
               "L1 hit rate",
               "L2 hit rate",
               "mem pipes busy pct",
               "SOL SM pct",
               "SOL memory pct",
               "SOL L1 cache pct",
               "SOL L2 cache pct",
               "SOL DRAM pct",
               "duration usec",
               "elapsed cycles",
               "SM active cycles",
               "SM frequency",
               "DRAM frequency",
               "threads * registers per thread / duration usec",
               "average threads * registers per thread each usec"
)

# p_quantity = 0, p_function = 1, p_sub_function = 2, p_size = 3, p_first = 4
# p_last = 5, p_increment = 6, p_function_call = 7
models = np.array([[
    [24, "AutoModelForQuestionAnswering with ALBERT xLarge pretrained "
     "on SQuAD2.0 by ktrapeznikov", -1, -1, 0, 688, (1), -1,
     -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -1, -1,
     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, -1
     ],
    # 267 total unrolled_elementwise_kernel kernels
    [24, "unrolled_elementwise_kernel", "AddFunctor", "large", 33, 677, (28),
     "void unrolled_elementwise_kernel<AddFunctor<float>, Array<char*, 3>, "
     "OffsetCalculator<2, unsigned int>, OffsetCalculator<1, unsigned int>, "
     "native::memory::LoadWithoutCast, native::memory::StoreWithoutCast> "
     "(int, AddFunctor<float>, Array<char*, 3>, OffsetCalculator<2, "
     "unsigned int>, OffsetCalculator<1, unsigned int>, "
     "native::memory::LoadWithoutCast, native::memory::StoreWithoutCast)",
     9920, 64, 634880, 15.50, 24, 0, 0, 1.02, 16.38,  # Launch Statistics
     358.92, 55.14, 56.27, 21.42,  # Memory Workload Analysis
     # GPU Speed Of Light (SOL):
     51.02, 79.11, 33.93, 38.25, 79.11, 51.84, 61359, 54880.38, 1.16, 7.09,
     293926, 265000
     ],
    [24, "unrolled_elementwise_kernel", "AddFunctor", "medium", 22, 666, (28),
     "void unrolled_elementwise_kernel<AddFunctor<float>, Array<char*, 3>, "
     "OffsetCalculator<2, unsigned int>, OffsetCalculator<1, unsigned int>, "
     "native::memory::LoadWithoutCast, native::memory::StoreWithoutCast>"
     "(int, AddFunctor<float>, Array<char*, 3>, OffsetCalculator<2, "
     "unsigned int>, OffsetCalculator<1, unsigned int>, "
     "native::memory::LoadWithoutCast, native::memory::StoreWithoutCast)",
     6007, 64, 384448, 9.39, 24, 0, 0, 1.02, 16.38,  # Launch Statistics
     396.25, 37.12, 50.34, 19.95,  # Memory Workload Analysis
     47.67, 81.31, 30.40, 32.09, 81.31, 31.55, 39879, 31271.20, 1.24, 7.61,
     292449, 265000
     ],

    [120, "unrolled_elementwise_kernel", "AddFunctor", "small", 13, 680,
     (2, 2, 11, 8, 5),
     "void unrolled_elementwise_kernel<AddFunctor<float>, Array<char*, 3>, "
     "OffsetCalculator<2, unsigned int>, OffsetCalculator<1, unsigned int>, "
     "native::memory::LoadWithoutCast, native::memory::StoreWithoutCast>"
     "(int, AddFunctor<float>, Array<char*, 3>, OffsetCalculator<2, "
     "unsigned int>, OffsetCalculator<1, unsigned int>, "
     "native::memory::LoadWithoutCast, native::memory::StoreWithoutCast)",
     2480, 64, 158720, 3.88, 24, 0, 0, 1.02, 16.38,  # Launch Statistics
     198.92, 61.85, 52.98, 21.77,  # Memory Workload Analysis
     52.40, 49.42, 33.68, 37.45, 49.42, 15.04, 15103, 12850.62, 984.80, 6.29,
     253277, 265000
     ],
    [1, "unrolled_elementwise_kernel", "AddFunctor", "11", 11, 11, (0),
     "void unrolled_elementwise_kernel<AddFunctor<float>, Array<char*, 3>, "
     "OffsetCalculator<2, unsigned int>, OffsetCalculator<1, unsigned int>, "
     "native::memory::LoadWithoutCast, native::memory::StoreWithoutCast>"
     "(int, AddFunctor<float>, Array<char*, 3>, OffsetCalculator<2, "
     "unsigned int>, OffsetCalculator<1, unsigned int>, "
     "native::memory::LoadWithoutCast, native::memory::StoreWithoutCast)",
     2480, 64, 158720, 3.88, 24, 0, 0, 1.02, 16.38,
     201.32, 61.77, 51.39, 21.71,
     52.29, 49.95, 33.39, 38.48, 49.95, 15.04, 15133, 12919.15, 987.37, 6.30,
     253277, 265000
     ],
    [1, "unrolled_elementwise_kernel", "Addfunctor", "685", 685, 685, (0),
     "void unrolled_elementwise_kernel<AddFunctor<float>, Array<char*, 3>, "
     "OffsetCalculator<2, unsigned int>, OffsetCalculator<1, unsigned int>, "
     "native::memory::LoadWithoutCast, native::memory::StoreWithoutCast>"
     "(int, AddFunctor<float>, Array<char*, 3>, OffsetCalculator<2, "
     "unsigned int>, OffsetCalculator<1, unsigned int>, "
     "native::memory::LoadWithoutCast, native::memory::StoreWithoutCast)",
     3, 64, 192, 0.00, 24, 0, 0, 1.02, 16.38,
     2.99, 53.98, 59.04, 0.07,
     0.19, 0.94, 4.54, 0.63, 0.94, 5.31, 4609, 198.25, 867.06, 4.99,
     868, 700
     ],
    [24, "unrolled_elementwise_kernel", "copy_device_to_device", "medium",
     19, 663, (28),
     "void unrolled_elementwise_kernel<copy_device_to_device(TensorIterator&, "
     "bool)::{lambda()#2}::operator()() const::{lambda()#4}::operator()() "
     "const::{lambda(float)#1}, Array<char*, 2>, OffsetCalculator<1, "
     "unsigned int>, char*, native::memory::LoadWithoutCast, Array<char*, 2>"
     "::StoreWithoutCast>(int, copy_device_to_device(TensorIterator&, bool)::"
     "{lambda()#2}::operator()() const::{lambda()#4}::operator()() const::"
     "{lambda(float)#1}, Array<char*, 2>, OffsetCalculator<1, unsigned int>, "
     "char*, native::memory::LoadWithoutCast, Array<char*, 2>::"
     "StoreWithoutCast)",
     2480, 64, 158720, 3.88, 20, 0, 0, 1.02, 16.38,
     97.73, 12.98, 88.73, 6.39,
     19.87, 58.01, 89.98, 58.01, 23.06, 32.32, 35619, 32845.20, 1.08, 6.62,
     # (threads * registers per thread) / mem pipes busy pct / duration usec
     15371, 15140
     ],
    [72, "unrolled_elementwise_kernel", "copy_device_to_device", "small",
     18, 670, (6, 2, 20),
     "void unrolled_elementwise_kernel<copy_device_to_device(TensorIterator&, "
     "bool)::{lambda()#2}::operator()() const::{lambda()#4}::operator()() "
     "const::{lambda(float)#1}, Array<char*, 2>, OffsetCalculator<1, "
     "unsigned int>, char*, native::memory::LoadWithoutCast, Array<char*, 2>"
     "::StoreWithoutCast>(int, copy_device_to_device(TensorIterator&, bool)::"
     "{lambda()#2}::operator()() const::{lambda()#4}::operator()() const::"
     "{lambda(float)#1}, Array<char*, 2>, OffsetCalculator<1, unsigned int>, "
     "char*, native::memory::LoadWithoutCast, Array<char*, 2>::"
     "StoreWithoutCast)",
     2480, 64, 158720, 3.88, 20, 0, 0, 1.02, 16.38,
     185.28, 0, 50.38, 12.64,
     50.70, 43.44, 27.85, 29.48, 43.44, 16.67, 18016, 14058.92, 1.06, 6.66,
     # (threads * registers per thread) / mem pipes busy pct / duration usec
     15065, 15140
     ],
    [1, "unrolled_elementwise_kernel", "copy_device_to_device", "0",
     0, 0, (0),
     "void unrolled_elementwise_kernel<copy_device_to_device(TensorIterator&, "
     "bool)::{lambda()#2}::operator()() const::{lambda()#4}::operator()() "
     "const::{lambda(float)#1}, Array<char*, 2>, TrivialOffsetCalculator<1, "
     "unsigned int>, char*, native::memory::LoadWithCast<1>, Array<char*, 2>"
     "::StoreWithCast>(int, copy_device_to_device(TensorIterator&, bool)::"
     "{lambda()#2}::operator()() const::{lambda()#4}::operator()() const::"
     "{lambda(float)#1}, Array<char*, 2>, TrivialOffsetCalculator<1, "
     "unsigned int>, char*, native::memory::LoadWithCast<1>, Array<char*, 2>::"
     "StoreWithCast)",
     2, 64, 128, 0.00, 32, 0, 0, 1.02, 16.38,
     1.98, 0, 69.31, 0.02,
     0.07, 0.55, 6.18, 0.50, 0.55, 5.76, 5607, 145.55, 972.69, 5.6,
     711, 700
     ],
    # GELU (Gaussian Error Linear Unit) activation function:
    [24, "vectorized_elementwise_kernel", "GELU", "", 34, 678, (28),
     "void vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(TensorIterator"
     "&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() "
     "const::{lambda()#1}::operator()() const::{lambda(float)#1}, "
     "Array<char*, 2> >(int, GeluCUDAKernelImpl(TensorIterator&)::"
     "{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::"
     "{lambda()#1}:operator()() const::{lambda(float)#1}, Array<char*, 2>)",
     9920, 64, 634880, 15.50, 32, 0, 0, 1.02, 16.38,
     364.12, 0, 50.18, 5.17,
     57.39, 82.76, 31.16, 35.85, 82.76, 50.72, 58453, 57042.10, 1.13, 6.87,
     400555, 400000
     ],
    [24, "vectorized_elementwise_kernel", "MulScalarFunctor", "",
     21, 665, (28),
     "void vectorized_elementwise_kernel<4, MulScalarFunctor<float, float>, "
     "Array<char*, 2> >(int, MulScalarFunctor<float, float>, Array<char*, 2>) "
     "(in ../libtorch/lib/libtorch_cuda.so)",
     6007, 64, 384448, 9.39, 16, 0, 0, 1.02, 16.38,
     298.83, 0, 49.44, 5.66,
     5.66, 84.72, 34.26, 40.10, 84.72, 34.72, 32385, 28365.22, 916.59, 5.51,
     177165, 175000
     ],
    [48, "vectorized_elementwise_kernel", "AddFunctor", "", 29, 681, (8, 20),
     "void vectorized_elementwise_kernel<4, AddFunctor<float>, Array<char*, 3>"
     " >(int, AddFunctor<float>, Array<char*, 3>)",
     2480, 64, 158720, 3.88, 20, 0, 0, 1.02, 16.38,
     452.07, 0, 33.88, 4.44,
     4.44, 86.61, 25.32, 34.95, 86.61, 16.48, 22748, 15668.48, 1.36, 8.16,
     192621, 190000
     ],
    [1, "vectorized_elementwise_kernel", "AUnaryFunctor", "1", 1, 1, (0),
     "void vectorized_elementwise_kernel<4, AUnaryFunctor<AddFunctor<float> >,"
     " Array<char*, 2> >(int, AUnaryFunctor<AddFunctor<float> >, "
     "Array<char*, 2>)",
     2, 64, 128, 0.00, 16, 0, 0, 1.02, 16.38,
     1.94, 0, 87.05, 0.02,
     0.03, 0.55, 12.37, 0.45, 0.55, 3.36, 3231, 72.75, 961.06, 5.52,
     609, 700
     ],
    [1, "vectorized_elementwise_kernel", "MulScalarFunctor", "2", 2, 2, (0),
     "void vectorized_elementwise_kernel<4, MulScalarFunctor<float, float>, "
     "Array<char*, 2> >(int, MulScalarFunctor<float, float>, Array<char*, 2>) "
     "(in ../libtorch/lib/libtorch_cuda.so)",
     2, 64, 128, 0.00, 15, 0, 0, 1.02, 16.38,
     920, 0, 219.17, 0.02,
     0.03, 1.00, 12.42, 1.00, 0.26, 3.20, 3111, 72.47, 970.26, 5.60,
     600, 700
     ],
    [2, "vectorized_elementwise_kernel", "AddFunctor", "5, 7", 5, 7, (2),
     "void vectorized_elementwise_kernel<4, AddFunctor<float>, Array<char*, 3>"
     " >(int, AddFunctor<float>, Array<char*, 3>)",
     155, 64, 9920, 0.24, 20, 0, 0, 1.02, 16.38,
     79.65, 33.33, 36.43, 1.54,
     1.54, 21.14, 11.92, 13.09, 21.14, 4, 4059, 2080.07, 1.01, 5.89,
     49600, 50000
     ],
    [1, "vectorized_elementwise_kernel", "BunaryFunctor", "688",
     688, 688, (0),
     "void vectorized_elementwise_kernel<4, BUnaryFunctor<AddFunctor<long "
     "long> >, Array<char*, 2> >(int, BUnaryFunctor<AddFunctor<long long> >, "
     "Array<char*, 2>)",
     1, 64, 64, 0.00, 20, 0, 0, 1.02, 16.38,
     40, 0, 57.14, 0.01,
     0.01, 0.58, 24.41, 0.58, 0.01, 3.20, 3201, 36.88, 999.69, 5.76,
     400, 700
     ],
    [3, "indexSelectLargeIndex", "", "3, 4, 6", 3, 6, (1, 2),
     "void indexSelectLargeIndex<float, unsigned int, 2, 2, -2, true>"
     "(cuda::detail::TensorInfo<float, unsigned int>, "
     "cuda::detail::TensorInfo<float, unsigned int>, "
     "cuda::detail::TensorInfo<long long, unsigned int>, int, int, "
     "unsigned int, unsigned int, long long)",
     310, 128, 39680, 0.65, 32, 0, 0, 1.02, 16.38,
     32.62, 9.35, 48.77, 5.08,
     19.75, 8.87, 8.48, 8.42, 8.87, 4.99, 4922, 2925.53, 977.73, 5.74,
     254461, 254000
     ],
    [48, "RowwiseMomentsCUDAKernel", "", "", 30, 682, (8, 20),
     "void RowwiseMomentsCUDAKernel<float>(long long, float, float const*, "
     "float*, float*)",
     310, 512, 158720, 2.58, 17, 256, 0, 1.02, 8.19,
     173.40, 0.16, 1.81, 30.76,
     30.76, 41.95, 36.34, 19.15, 41.95, 14.94, 16204, 13476.58, 1.07, 6.46,
     180605, 181000
     ],
    [1, "RowwiseMomentsCUDAKernel", "", "8", 8, 8, (0),
     "void RowwiseMomentsCUDAKernel<float>(long long, float, float const*, "
     "float*, float*)",
     310, 512, 158720, 2.58, 17, 256, 0, 1.02, 8.19,
     19.02, 3.54, 21.22, 39.98,
     39.98, 39.98, 51.19, 2.33, 4.51, 8.86, 10025, 7751.02, 1.12, 6.58,
     304542, 305000
     ],
    [48, "LayerNormForwardCUDAKernel", "", "", 31, 683, (8, 20),
     "void LayerNormForwardCUDAKernel<float>(long long, float const*, "
     "float const*, float const*, float const*, float const*, float*)",
     310, 256, 79360, 1.29, 22, 0, 0, 1.02, 8.19,
     222.45, 49.73, 55.50, 40.88,
     40.88, 52.94, 43.03, 39.74, 52.94, 14.27, 15100, 14121.83, 1.04, 6.57,
     122349, 122000
     ],
    [1, "LayerNormForwardCUDAKernel", "", "9", 9, 9, (0),
     "void LayerNormForwardCUDAKernel<float>(long long, float const*, "
     "float const*, float const*, float const*, float const*, float*)",
     310, 256, 79360, 1.29, 22, 0, 0, 1.02, 8.19,
     34.68, 48.27, 63.67, 10.16,
     10.16, 10.16, 19.89, 8.39, 9.13, 4.83, 4920, 2494.10, 1.01, 5.93,
     361474, 360000
     ],
    [48, "Kernel", "cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4",
     "xlarge", 32, 679, (3, 25),
     "void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_"
     "align4>(cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4::Params)",
     320, 128, 40960, 8, 148, 0, 73.73, 1.02, 102.40,
     105.04, 0, 91.37, 14.63,
     47.10, 45.92, 36.39, 45.92, 23.88, 934.94, 1115108, 1108240.07, 1.19,
     6.87, 6484, 6400
     ],
    [96, "Kernel", "cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4",
     "large", 12, 671, (2, 2, 11, 13),
     "void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_"
     "align4>(cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4::Params)",
     80, 128, 10240, 2, 148, 0, 73.73, 1.02, 102.40,
     97.54, 0, 91.31, 14.43,
     46.44, 45.64, 35.87, 45.64, 21.96, 239.58, 286136, 278578.78, 1.19, 6.94,
     6326, 6400
     ],
    [24, "Kernel", "cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align1",
     "medium", 20, 664, (28),
     "void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_"
     "align1>(cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align1::Params)",
     320, 128, 40960, 8, 154, 0, 73.73, 1.02, 102.40,
     104.63, 73.88, 87.97, 37.63,
     37.63, 46.30, 48.86, 28.11, 23.78, 88.32, 103232, 96586.43, 1.15, 6.88,
     71420, 71000
     ],
    [24, "Kernel", "cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align1",
     "small", 25, 669, (28),
     "void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_"
     "align1>(cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align1::Params)",
     80, 128, 10240, 2, 154, 0, 73.73, 1.02, 102.40,
     183.45, 74.48, 77.39, 37.55,
     37.55, 47.95, 51.23, 34.64, 41.24, 57.09, 66442, 61075.25, 1.14, 6.95,
     27622, 28000
     ],
    [1, "Kernel", "cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4",
     "10", 10, 10, (0),
     "void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_"
     "align4>(cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4::Params)",
     80, 128, 10240, 2, 148, 0, 73.73, 1.02, 102.40,
     78.22, 0, 91.92, 10.71,
     20.68, 27.11, 27.21, 27.11, 13.91, 27.14, 40178, 26529.12, 1.46, 8.79,
     55841, 56000
     ],
    [1, "Kernel", "cutlass_80_tensorop_s1688gemm_64x64_32x4_tn_align1",
     "684", 684, 684, (0),
     "void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x4_tn_align1>"
     "(cutlass_80_tensorop_s1688gemm_64x64_32x4_tn_align1::Params)",
     5, 64, 320, 0.12, 168, 0, 65.54, 1.02, 102.40,
     22.86, 75.00, 3.73, 1.87,
     2.53, 5.13, 21.54, 1.97, 5.13, 134.30, 162299, 19970.33, 1.21, 6.96,
     400, 700
     ],
    [24, "softmax_warp_forward", "", "", 23, 667, (28),
     "void softmax_warp_forward<float, float, float, 9, false>(float*, "
     "float const*, int, int, int)",
     1240, 128, 158720, 2.58, 39, 0, 0, 1.02, 16.38,
     441.72, 31.16, 54.28, 21.36,
     37.01, 77.82, 38.66, 30.98, 77.82, 30.34, 44838, 28336.47, 1.45, 8.87,
     204024, 200000
     ],
    [2, "reduce_kernel", "", "686, 687", 686, 687, (1),
     "void reduce_kernel<512, 1, ReduceOp<float, ArgMaxOps<float>, "
     "unsigned int, long long, 4> >",
     1, 256, 256, 0.00, 32, 16, 4.10, 1.02, 65.54,
     10.56, 0, 761.21, 0.15,
     0.22, 2.57, 11.85, 0.51, 2.57, 8, 8923, 110.92, 1.11, 6.42,
     1024, 700
     ]
    ]], dtype=object)


def printlog(s):
    print(s)
    if logFile is not None:
        print(s, file=logFile)


# prints GPU memory usage and availability [4]
def printMemory():
    if torch.cuda.is_available():
        totalGPU = torch.cuda.get_device_properties(0).total_memory
        reservedGPU = torch.cuda.memory_reserved(0)
        allocatedGPU = torch.cuda.memory_allocated(0)
        freeGPU = reservedGPU - allocatedGPU  # free inside reserved
        printlog(f"     total GPU memory: {totalGPU}")
        printlog(f"  reserved GPU memory: {reservedGPU}")
        printlog(f" allocated GPU memory: {allocatedGPU}")
        printlog(f"      free GPU memory: {freeGPU}")


def printGPUspecs(gpuId):
    printlog(f"GPU Specifications for {gpu_names[gpuId: ]}:")
    printlog("")
    for i in range(1, len(gpu_specnames)):
        printlog(f"{gpu_specnames[i]}: {gpus[gpuId, i]}")


def printModelParameters(modelId):
    printlog("Model Parameters for:")
    for kernelId in range(0, len(models[modelId])):
        printlog(f"KERNEL {kernelId}")
        for i in range(0, len(param_names)):
            if models[modelId][kernelId][i] != -1:
                printlog(f"{param_names[i]}: {models[modelId][kernelId][i]}")
        printlog("")


def calcTotalDurationUsecs(modelId):
    totalDuration = 0.0
    for kernelId in range(1, len(models[modelId])):
        totalDuration += (models[modelId][kernelId][p_quantity] *
                          models[modelId][kernelId][p_duration_usec])
    printlog(f"   Model actual duration: {totalDuration:.0f} usec")
    return totalDuration


def calcModelKernelDuration(modelId, kernelId):
    kernel = models[modelId][kernelId]
    kernelDuration = 0.0
    if kernelId == k_unrolled_elementwise_kernel_copy_device_to_device_medium \
       or (kernelId ==
           k_unrolled_elementwise_kernel_copy_device_to_device_small):
        kernelDuration = kernel[p_threads] * kernel[p_registers_per_thread] / \
                         kernel[p_mem_pipes_busy_pct] / \
                         kernel[p_avg_threads_x_registers_per_thread_each_usec]
    else:
        kernelDuration = kernel[p_threads] * kernel[p_registers_per_thread] / \
                         kernel[p_avg_threads_x_registers_per_thread_each_usec]

    kernelTotal = kernel[p_quantity] * kernelDuration
    printlog(f"Kernel #{kernelId} {kernel[p_function]}-"
             f"{kernel[p_sub_function]}-"
             f"{kernel[p_size]}: {kernel[p_quantity]} * {kernelDuration:0.1f} "
             f"usec [{kernel[p_duration_usec]:0.1f} actual] = "
             f"{kernelTotal:0.1f} usec")
    return kernelTotal


def performanceModel(modelId):
    model = models[modelId]
    printlog(f"Kernel Execution Time (in usec) Performance Prediction "
             f"for {model[k_layers][p_quantity]}-layer")
    printlog(f"{model[k_layers][p_function]}")
    printlog(f"with {len(model) - 1} Kernels executed a total of "
             f"{model[k_layers][p_last] + 1} times.")
    printlog("")
    modelDuration = 0.0
    for kernelId in range(1, len(model)):
        modelDuration += calcModelKernelDuration(modelId, kernelId)
    printlog("")
    printlog(f"Model predicted duration: {modelDuration:.0f} usec")
    return modelDuration


def main():
    startTime = int(round(time.time() * 1000))  # time in ms
    ap = argparse.ArgumentParser()  # argument parser
    ap.add_argument("-o", "--output", type=argparse.FileType('a'),
                    help="log output file path")
    args = vars(ap.parse_args())
    global logFile  # output file for printlog(s)
    logFile = args['output']  # optional output file used by printlog(s)
    printlog(f"Performance modeling {gpu_names[0:]} for one question "
             "inference using:")
    printlog("")
    printlog("AutoModelForQuestionAnswering with ALBERT xLarge pretrained "
             "on SQuAD2.0 by ktrapeznikov")
    printlog("")
    printGPUspecs(g_RTX_3070_Laptop)
    printlog("")
    printModelParameters(m_ALBERT_xlarge_qa)
    predictedDuration = performanceModel(m_ALBERT_xlarge_qa)
    actualDuration = calcTotalDurationUsecs(m_ALBERT_xlarge_qa)
    percentError = abs(predictedDuration - actualDuration) / actualDuration \
        * 100.0
    printlog(f"        Percentage error: {percentError:.2f}%")

    elapsedTime = int(round(time.time() * 1000)) - startTime
    printlog("")
    printlog(f"elapsed time: {elapsedTime} ms")

    if logFile is not None:
        logFile.close()


if __name__ == "__main__":
    main()
