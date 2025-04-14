add_executable(all-reduce-sync-with-host all-reduce-sync-with-host.cpp)

include_directories(${CMAKE_SOURCE_DIR}/common)

target_link_libraries(all-reduce-sync-with-host commonlib)

target_link_libraries(all-reduce-sync-with-host ze_loader OpenCL)