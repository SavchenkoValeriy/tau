add_tau_library(tauCore
  Analysis.cpp
  PostOrderEnumerator.cpp
)

target_link_libraries(
  tauCore
  PRIVATE
  immer
  )
