add_tau_library(tauCheckers
  UseOfUninit.cpp
  UseAfterFree.cpp
)

target_link_libraries(
  tauCheckers
  PRIVATE
  MLIRPass
)

add_dependencies(tauCheckers tauAir)
