function(tau_set_compile_flags name)
  # Ambiguous reversed operator warning is too vigilant with
  # tau's dependencies.
  set_property(TARGET ${name} APPEND_STRING PROPERTY
    COMPILE_FLAGS " -fno-rtti -Wno-ambiguous-reversed-operator")
endfunction()

function(add_tau_executable name)
  add_llvm_executable(${name} ${ARGN})
  tau_set_compile_flags(${name})
endfunction()

function(add_tau_library name)
  add_llvm_library(${name} ${ARGN})
  tau_set_compile_flags(${name})
endfunction()

function(add_tau_unittest name)
  add_tau_executable(${name} ${ARGN})
  target_link_libraries(${name}
    PRIVATE
    LLVMSupport
    Catch2::Catch2
    trompeloeil::trompeloeil
    Catch2WithMain
    )
  # trompeloeil cannot be build without exceptions
  set_property(TARGET ${name}
    APPEND_STRING PROPERTY
    COMPILE_FLAGS " -fexceptions"
    )
  set_target_properties(${name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ""
    )
  catch_discover_tests(${name})
  add_dependencies(unittests ${name})
endfunction()

function(tau_target_link_system_libraries target scope)
  set(libs ${ARGN})
  foreach(lib ${libs})
    get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${target} SYSTEM ${scope} ${lib_include_dirs})
    target_link_libraries(${target} ${scope} ${lib})
  endforeach(lib)
endfunction(tau_target_link_system_libraries)
