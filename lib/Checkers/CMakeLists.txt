add_tau_library(tauCheckers
  Registry.cpp
  UseOfUninit.cpp
)

target_link_libraries(
  tauCheckers
  PRIVATE
  MLIRPass
)
