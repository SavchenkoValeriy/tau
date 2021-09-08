add_tau_library(tauCheckers
  UseOfUninit.cpp
)

target_link_libraries(
  tauCheckers
  PRIVATE
  MLIRPass
)
