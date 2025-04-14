add_executable(all-reduce-sync-with-event all-reduce-sync-with-event.cpp)

include_directories(${CMAKE_SOURCE_DIR}/common)

target_link_libraries(all-reduce-sync-with-event commonlib)

target_link_libraries(all-reduce-sync-with-event ze_loader OpenCL)