add_tau_library(tauCore
  Analysis.cpp
  PostOrderEnumerator.cpp
  State.cpp
)

target_link_libraries(
  tauCore
  PRIVATE
  immer
  )
