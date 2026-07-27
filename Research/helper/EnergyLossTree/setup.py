"""Build the vendored TreeFilter CUDA extension in place."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


PACKAGE_DIR = Path(__file__).resolve().parent
CSRC_DIR = PACKAGE_DIR / "csrc"


setup(
    name="tta-energy-loss-tree-cuda",
    version="0.1.0",
    description="Forward-only CUDA TreeFilter extension for EnergyLossTree",
    ext_modules=[
        CUDAExtension(
            name="_tree_filter_cuda",
            include_dirs=[str(CSRC_DIR)],
            sources=[
                str(CSRC_DIR / "tree_filter.cpp"),
                str(CSRC_DIR / "mst" / "mst.cpp"),
                str(CSRC_DIR / "mst" / "boruvka.cpp"),
                str(CSRC_DIR / "bfs" / "bfs.cu"),
                str(CSRC_DIR / "refine" / "refine.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
