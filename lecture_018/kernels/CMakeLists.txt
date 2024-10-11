cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0146 NEW)  # Suppress the FindCUDA policy warning

project(dlrm_fused_kernels LANGUAGES CXX CUDA)  # Include CUDA in the project

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/src/*.cuh")

# Set the URL for the PyTorch library
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcu121.zip")

# Set the download directory and target directory for libtorch
set(LIBTORCH_ZIP "${CMAKE_SOURCE_DIR}/third-party/libtorch.zip")
set(LIBTORCH_DIR "${CMAKE_SOURCE_DIR}/third-party/libtorch")

# Download the PyTorch library if not already downloaded
if(NOT EXISTS ${LIBTORCH_ZIP})
    file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ZIP} SHOW_PROGRESS)
endif()

# Unzip the library if not already unzipped
if(NOT EXISTS ${LIBTORCH_DIR})
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvf ${LIBTORCH_ZIP} --strip-components=1 -C ${LIBTORCH_DIR})
endif()

# Add the extracted libtorch to the CMake prefix path
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

# Try to automatically detect an active Conda environment
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Detected active Conda environment: $ENV{CONDA_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
else()
    message(WARNING "No active Conda environment detected. Please set CONDA_PREFIX_PATH manually.")
endif()

find_package(Torch REQUIRED)

# CUDA language support is already enabled through 'enable_language(CUDA)'

# Set compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(dlrm_kernels_test "${CMAKE_SOURCE_DIR}/src/dlrm_kernels_main.cc" ${gpu_source_files})
target_link_libraries(dlrm_kernels_test "${TORCH_LIBRARIES}")

add_executable(fused_kernels_lora_test "${CMAKE_SOURCE_DIR}/src/fused_kernels_lora_main.cc" ${gpu_source_files})
target_link_libraries(fused_kernels_lora_test "${TORCH_LIBRARIES}")
