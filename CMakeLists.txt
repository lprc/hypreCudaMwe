cmake_minimum_required(VERSION 3.18)
project(HYPRE_CUDA_MWE VERSION 0.1.0 LANGUAGES CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)

# Need to add -DHYPRE_DIR=path/to/hypre/lib/cmake/HYPRE for the configure step
find_package(HYPRE REQUIRED)
if (DEFINED HYPRE_DIR)
    include_directories("${HYPRE_DIR}/../../../include")
    link_directories("${HYPRE_DIR}/../..")
endif()

add_executable(testMweHypreCuda test_mwe_hypre_cuda.cpp)
target_link_libraries(testMweHypreCuda 
    PRIVATE 
        CUDA::cudart
        CUDA::curand
        CUDA::cusparse
        HYPRE
        MPI::MPI_CXX
)
