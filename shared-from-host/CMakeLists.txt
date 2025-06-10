add_executable(shared-from-host shared-from-host.cpp)

include_directories(${CMAKE_SOURCE_DIR}/common)

target_link_libraries(shared-from-host commonlib)

target_link_libraries(shared-from-host OpenCL)