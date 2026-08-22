# Infrastructure fixes

FINAL D fixes the Kaggle RTX PRO 6000 FlashInfer SM120 JIT host-link failure caused by a missing unversioned `libcuda.so` linker name. The repair is isolated from ARCangel policy logic and applied identically to S115 and S120.
