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

# gRPC + Protobuf for the `serve` mode (the Go OpenAI gateway is the client). Prefer a CMake
# config package (a source/conda build under /usr/local — e.g. the latest gRPC); fall back to the
# system pkg-config (the distro's libgrpc++-dev). Only stable gRPC APIs are used, so either works.
find_package(gRPC CONFIG QUIET)
find_package(Protobuf CONFIG QUIET)
if(gRPC_FOUND AND Protobuf_FOUND)
  message(STATUS "gRPC via CMake config: gRPC ${gRPC_VERSION}, protobuf ${Protobuf_VERSION}")
  set(GRPC_LIBS gRPC::grpc++ protobuf::libprotobuf)
  set(PROTOC_BIN $<TARGET_FILE:protobuf::protoc>)
  set(GRPC_PLUGIN_BIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
  pkg_check_modules(PROTOBUF REQUIRED IMPORTED_TARGET protobuf)
  message(STATUS "gRPC via pkg-config: grpc++ ${GRPCPP_VERSION}, protobuf ${PROTOBUF_VERSION}")
  set(GRPC_LIBS PkgConfig::GRPCPP PkgConfig::PROTOBUF)
  find_program(PROTOC_BIN protoc REQUIRED)
  find_program(GRPC_PLUGIN_BIN grpc_cpp_plugin REQUIRED)
endif()

# Generate the protobuf + gRPC C++ stubs from the shared proto (kept at the repo root, since the
# Go gateway generates from it too) into the build tree.
set(GEN_DIR ${CMAKE_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GEN_DIR})
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../proto)
set(PROTO ${PROTO_DIR}/inference.proto)
set(PROTO_SRCS ${GEN_DIR}/inference.pb.cc ${GEN_DIR}/inference.grpc.pb.cc)
set(PROTO_HDRS ${GEN_DIR}/inference.pb.h  ${GEN_DIR}/inference.grpc.pb.h)
add_custom_command(
  OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
  COMMAND ${PROTOC_BIN}
          -I ${PROTO_DIR}
          --cpp_out=${GEN_DIR}
          --grpc_out=${GEN_DIR}
          --plugin=protoc-gen-grpc=${GRPC_PLUGIN_BIN}
          ${PROTO}
  DEPENDS ${PROTO}
  COMMENT "protoc: inference.proto -> C++ stubs")

add_executable(flashqwen
  src/main.cpp
  src/cli.cpp
  src/sampler.cpp
  src/generate.cpp
  src/chat.cpp
  src/benchmark.cpp
  src/scheduler.cpp
  src/grpc_server.cpp
  src/safetensors.cpp
  src/kernels.cu
  src/model/model.cu
  src/model/tokenizer.cpp
  src/model/prompt.cpp
  ${PROTO_SRCS}
)
target_include_directories(flashqwen PRIVATE src third_party ${GEN_DIR})  # third_party: RapidJSON + CLI11
target_link_libraries(flashqwen PRIVATE ${GRPC_LIBS})
# Prefer the source-built gRPC/protobuf in /usr/local over any system (apt) copy at runtime.
set_target_properties(flashqwen PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                           BUILD_RPATH /usr/local/lib)
