add_llvm_executable(tau-cc taucc.cpp)
set_property(TARGET tau-cc PROPERTY CXX_STANDARD 20)
target_link_libraries(tau-cc PRIVATE clangTooling)
