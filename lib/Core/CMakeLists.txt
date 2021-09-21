add_tau_library(tauCore
  Analysis.cpp
  CheckerRegistry.cpp
  PostOrderEnumerator.cpp
  State.cpp
)

target_link_libraries(
  tauCore
  PRIVATE
  immer
  )
