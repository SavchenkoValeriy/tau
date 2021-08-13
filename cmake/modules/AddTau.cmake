function(tau_set_compile_flags name)
  set_property(TARGET ${name} APPEND_STRING PROPERTY
    COMPILE_FLAGS " -fno-rtti")
  set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)
endfunction()

function(add_tau_executable name)
  add_llvm_executable(${name} ${ARGN})
  tau_set_compile_flags(${name})
endfunction()

function(add_tau_library name)
  add_llvm_library(${name} ${ARGN})
  tau_set_compile_flags(${name})
endfunction()
