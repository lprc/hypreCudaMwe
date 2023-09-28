# hypreCudaMwe
A minimal working example for HYPRE using CUDA

## Build with CMake
Make sure to set `-DHYPRE_DIR` and `-DCMAKE_CXX_COMPILER` properly. 

If HYPRE is installed system-wide, setting HYPRE_DIR might not be needed. If set, make sure to point it to the directory, where the CMake files of HYPRE sit.

Also, if your default C++ compiler works, setting CMAKE_CXX_COMPILER might not be needed. In my case, `ld` had problems finding CUDA libraries, so I just used Clang here, which works fine.
```
mkdir build
cd build
cmake -DHYPRE_DIR=/home/simon/hypreWithoutUnifiedMem/lib/cmake/HYPRE -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/clang++
cd ..
cmake --build build
```

Then running `build/testMweHypreCuda` yields
```
CUDA ERROR (code = 700, an illegal memory access was encountered) at /home/simon/projects/hypre/src/utilities/memory.c:472
```
on my machine.
