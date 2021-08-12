cmake_minimum_required(VERSION 3.4.3)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(tau
  ${cmake_3_0_PROJ_VERSION}
  ${cmake_3_0_LANGUAGES}
  CXX)

# Find CMake file for Clang
find_package(Clang REQUIRED)
# Find CMake file for MLIR
find_package(MLIR REQUIRED)

# Add path to MLIR modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
# Add path to LLVM modules
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# import LLVM CMake functions
include(AddLLVM)
include(TableGen)
# import MLIR CMake functions
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_subdirectory(lib)
add_subdirectory(include/tau)
