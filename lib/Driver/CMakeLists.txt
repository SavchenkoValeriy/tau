add_llvm_executable(taucc taucc.cpp)
set_property(TARGET taucc PROPERTY CXX_STANDARD 20)
target_link_libraries(taucc PRIVATE clangTooling)
