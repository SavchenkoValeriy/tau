add_tau_library(tauCore
  Analysis.cpp
  CheckerPass.cpp
  CheckerRegistry.cpp
  FlowSensitive.cpp
  PostOrderEnumerator.cpp
  TopoOrderEnumerator.cpp
  State.cpp
)

target_link_libraries(
  tauCore
  PRIVATE
  immer
  )
