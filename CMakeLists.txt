cmake_minimum_required(VERSION 3.14)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()
if(POLICY CMP0156)
  # Deduplicate libraries during linking
  cmake_policy(SET CMP0156 NEW)
endif()
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()
# immer has optional dependency on Boost and this suppresses the warning.
# We don't use it anyways.
set(CMAKE_POLICY_DEFAULT_CMP0167 OLD)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)


project(tau
  ${cmake_3_0_PROJ_VERSION}
  ${cmake_3_0_LANGUAGES}
  C
  CXX)

# Shut up warning from linking with MLIR
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON CACHE BOOL "Hide inlines")

set(CMAKE_CXX_STANDARD 20)

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
  GIT_TAG        v3.7.0
)

set(CATCH_BUILD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Catch2)

# Fetch trompeloeil
FetchContent_Declare(
  trompeloeil
  GIT_REPOSITORY https://github.com/rollbear/trompeloeil.git
  GIT_TAG        v48
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

FetchContent_Declare(
  Immer
  GIT_REPOSITORY https://github.com/arximboldi/immer.git
  GIT_TAG        v0.8.1
)
set(immer_BUILD_TESTS OFF)
set(immer_BUILD_EXAMPLES OFF)
set(immer_BUILD_DOCS OFF)
set(immer_BUILD_EXTRAS OFF)

set(FPHSA_NAME_MISMATCHED on) # see: https://github.com/arximboldi/immer/issues/204
FetchContent_MakeAvailable(immer)
unset(FPHSA_NAME_MISMATCHED)
FetchContent_GetProperties(Immer)

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

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${CLANG_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

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

option(TAU_ENABLE_ANALYSIS_TRACING "Enable analysis tracing")
if(TAU_ENABLE_ANALYSIS_TRACING)
  add_compile_definitions(ANALYSIS_TRACER_ENABLED)

  find_program(BUN_EXECUTABLE bun)
  if(NOT BUN_EXECUTABLE)
    message(FATAL_ERROR "bun executable not found. Please install bun to build trace viewer.")
  endif()

  add_subdirectory(ui/trace-viewer)
endif()

enable_testing()
add_custom_target(check)

add_subdirectory(utils/FileCheck)

add_subdirectory(lib)
add_subdirectory(include/tau)
add_subdirectory(unittests)
add_subdirectory(test)
