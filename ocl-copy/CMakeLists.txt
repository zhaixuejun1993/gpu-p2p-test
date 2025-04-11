add_executable(ocl-copy ocl-copy.cpp)

include_directories(${CMAKE_SOURCE_DIR}/common)

target_link_libraries(ocl-copy commonlib)

target_link_libraries(ocl-copy ze_loader OpenCL)