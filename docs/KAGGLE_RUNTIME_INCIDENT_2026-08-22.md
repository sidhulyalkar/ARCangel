# Kaggle RTX PRO 6000 runtime incident — 2026-08-22

Both FINAL C candidates reached the intended Qwen snapshot and then failed identically during FlashInfer 0.6.6 SM120 JIT. CUDA itself was healthy: PyTorch saw the RTX PRO 6000, `nvidia-smi` worked, ARC wheels imported, vLLM 0.19.0 installed, and model discovery confirmed Qwen + 27B + FP8 + safetensors.

The actual root cause was the final host-link step:

```text
/usr/bin/ld: cannot find -lcuda: No such file or directory
```

FlashInfer's generated ninja link flags request the unversioned CUDA driver linker name with `-lcuda`. Kaggle can expose only the versioned runtime `libcuda.so.1` in the driver mount while omitting `libcuda.so` from CUDA_HOME's usual linker directories.

## ARCangel fix

FINAL D performs a preflight before vLLM startup:

- dynamically resolves the live `libcuda.so.1`;
- validates runtime loading with `ctypes.CDLL`;
- creates a writable `libcuda.so` alias;
- prepends its directory to `LIBRARY_PATH` and `LD_LIBRARY_PATH`;
- configures writable FlashInfer JIT/generated-source directories;
- places the alias in CUDA's stub linker directory when writable;
- compiles a real C++ `-lcuda` probe and requires `CUDA DRIVER LINKER PASS`;
- only then launches vLLM.

Reusable implementation: `scripts/kaggle_cuda_linker_preflight.py`.

This incident is infrastructure evidence only. It does not change the D110R2/D210R2 policy conclusions or the intended S115/S120 hypothesis separation.
