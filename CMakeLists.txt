cmake_minimum_required(VERSION 3.14)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(tau
  ${cmake_3_0_PROJ_VERSION}
  ${cmake_3_0_LANGUAGES}
  CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# Shut up warning from linking with MLIR
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON CACHE BOOL "Hide inlines")

# Set path for custom Tau-specific modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

option(TAU_CLANG_PATH "Path to installed Clang/LLVM")
if(TAU_CLANG_PATH)
  set(Clang_DIR "${TAU_CLANG_PATH}/lib/cmake/clang")
  set(MLIR_DIR "${TAU_CLANG_PATH}/lib/cmake/mlir")
endif()

# Find CMake file for Clang
find_package(Clang REQUIRED)
# Find CMake file for MLIR
find_package(MLIR REQUIRED)

# FetchContent allows us to download external projects
# at configuration time
Include(FetchContent)

# Fetch Catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.7
)

set(CATCH_BUILD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Catch2)

# Fetch trompeloeil
FetchContent_Declare(
  trompeloeil
  GIT_REPOSITORY https://github.com/rollbear/trompeloeil.git
  GIT_TAG        v41
)
FetchContent_MakeAvailable(trompeloeil)

# Add path to MLIR modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
# Add path to LLVM modules
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
# Add path to Catch2 modules
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# import LLVM CMake functions
include(AddLLVM)
include(TableGen)
# import MLIR CMake functions
include(AddMLIR)
# import testing functions
include(CTest)
include(Catch)
# import Tau CMake functions
include(AddTau)


find_package(Immer REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(TAU_TOOLS_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(TAU_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TAU_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(LIT_TOOL_NAMES "llvm-lit" "lit")
find_program(
  LIT_TOOL
  NAMES ${LIT_TOOL_NAMES}
  DOC "Path to lit tool"
  )

if ((NOT LIT_TOOL) OR (NOT EXISTS "${LIT_TOOL}"))
  message(FATAL_ERROR "lit is required for testing."
    " Please, install it via pip."
    )
else()
  message(STATUS "Using lit: ${LIT_TOOL}")
endif()

set(LLVM_DEFAULT_EXTERNAL_LIT ${LIT_TOOL})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Tau's own include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})

# Always export compilation commands
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

option(TAU_ENABLE_ASAN "Enable address sanitizer")
if(TAU_ENABLE_ASAN)
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
  set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

enable_testing()
add_custom_target(check)

add_subdirectory(lib)
add_subdirectory(include/tau)
add_subdirectory(unittests)
add_subdirectory(test)
