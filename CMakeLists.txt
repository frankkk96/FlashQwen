cmake_minimum_required(VERSION 3.18)

# Target GPU architecture. 89 = Ada (RTX 4090). Must be set before project()/enabling CUDA.
# Override for other GPUs, e.g. -DCMAKE_CUDA_ARCHITECTURES=90 for Hopper.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(FlashQwen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

add_executable(flashqwen
  src/main.cpp
  src/cli.cpp
  src/sampler.cpp
  src/generate.cpp
  src/chat.cpp
  src/benchmark.cpp
  src/scheduler.cpp
  src/safetensors.cpp
  src/tokenizer.cpp
  src/model.cu
  src/kernels.cu
)
target_include_directories(flashqwen PRIVATE src third_party)  # third_party: RapidJSON + CLI11
set_target_properties(flashqwen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
