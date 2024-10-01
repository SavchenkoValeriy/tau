add_tau_library(tauCore
  AddressTakenAnalysis.cpp
  AliasAnalysis.cpp
  Analysis.cpp
  CheckerPass.cpp
  CheckerRegistry.cpp
  EscapeAnalysis.cpp
  Events.cpp
  FlowSensitive.cpp
  MemoryStore.cpp
  MutualExclusionAnalysis.cpp
  PointsToAnalysis.cpp
  PostOrderEnumerator.cpp
  ReachingDefs.cpp
  State.cpp
  TopoOrderEnumerator.cpp
)

tau_target_link_system_libraries(
  tauCore
  PRIVATE
  immer
)

add_dependencies(tauCore tauAir)
