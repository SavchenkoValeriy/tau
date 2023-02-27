add_tau_library(tauCore
  AddressTakenAnalysis.cpp
  Analysis.cpp
  CheckerPass.cpp
  CheckerRegistry.cpp
  FlowSensitive.cpp
  PostOrderEnumerator.cpp
  TopoOrderEnumerator.cpp
  State.cpp
)

tau_target_link_system_libraries(
  tauCore
  PRIVATE
  immer
)

add_dependencies(tauCore tauAir)
