add_executable(ocl-buffer-location-reproducer ocl-buffer-location-reproducer.cpp)

include_directories(${CMAKE_SOURCE_DIR}/common)

target_link_libraries(ocl-buffer-location-reproducer commonlib)

target_link_libraries(ocl-buffer-location-reproducer ze_loader OpenCL)