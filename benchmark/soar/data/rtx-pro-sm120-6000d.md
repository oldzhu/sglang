Below is the extracted content from the provided PDF, **"从硬件底层看算子优化_NVIDIA穆正阳.pdf"**, organized by its core technical sections:

### **1. General Introduction & LLM Sparsity**

The document introduces optimization strategies for Large Language Models (LLMs) on **RTX PRO GPUs**. It covers various attention mechanisms and sparsity techniques:

* 
**Traditional Softmax Attention**: $O(n^2)$ complexity.


* 
**Sparse Attention**: Pruning tokens to focus on local or important information (e.g., Sliding Window, MOBA, DSA, NSA).


* 
**Linear Attention**: Approximating softmax via kernelization and associativity.


* **Hardware Optimization Targets**:
* Maximize SM utilization and occupancy.


* Maximize memory and instruction throughput.


* Minimize memory thrashing through better management.





### **2. RTX PRO (sm120) Hardware Characteristics**

The competition utilizes the **sm120 architecture**, which requires specific toolchain compilation or capability checks. Key features include:

* **Optimization Strategy**: Similar to the Hopper architecture.
* **MMA (Matrix Multiply-Accumulate)**: Commands are at the **warp level** rather than the warpgroup level.
* **TMA (Tensor Memory Accelerator)**: Hardware support for advanced load/store operations.
* **QMMA**: Support for **mxfp8** (native FP8 compute).

**Hardware Specifications**:

| Feature | Specification |
| --- | --- |
| **BF16/FP16 Tensor Core** | 148 TFLOPS |
| **FP8 Tensor Core** | 296 TFLOPS |
| **FP4 Tensor Core** | 593 TFLOPS |
| **GPU Memory** | 84GB GDDR7 |
| **Memory Bandwidth** | 1398 GB/s |
| **L2 Cache** | 112MB |

### **3. Profiling and Analysis Tools**

The document highlights two primary NVIDIA tools for detecting and solving performance issues:

* 
**NVIDIA Nsight Systems**: Provides a system-wide performance analysis and visualizes workloads on a timeline. It can be integrated into Python workflows via JupyterLab.


* 
**NVIDIA Nsight Compute**: An interactive profiler for a "deeper dive" into specific CUDA kernels. It provides detailed metrics on:


* Compute throughput and memory workload.


* Detailed memory charts, including L1/L2 cache hit rates.


* Correlation between source code and specific instructions.





### **4. Developer Resources**

NVIDIA provides reference usage and integration guides through **TensorRT-LLM**. Specifically, it points developers to a unit test for **FP8 block scale GEMM** on GitHub.
