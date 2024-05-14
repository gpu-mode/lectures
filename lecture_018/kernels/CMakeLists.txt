cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(dlrm_fused_kernels)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/src/*.cuh")

# wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcu121.zip
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third-party/libtorch)
list(APPEND CMAKE_PREFIX_PATH /home/ksharma/anaconda3/envs/cuda-learn)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(dlrm_kernels_test "${CMAKE_SOURCE_DIR}/src/dlrm_kernels_main.cc" ${gpu_source_files})
target_link_libraries(dlrm_kernels_test "${TORCH_LIBRARIES}")

add_executable(fused_kernels_lora_test "${CMAKE_SOURCE_DIR}/src/fused_kernels_lora_main.cc" ${gpu_source_files})
target_link_libraries(fused_kernels_lora_test "${TORCH_LIBRARIES}")
