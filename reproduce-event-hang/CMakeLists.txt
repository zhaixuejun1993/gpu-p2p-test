add_executable(reproduce-event-hang reproduce-event-hang.cpp)

include_directories(${CMAKE_SOURCE_DIR}/common)

target_link_libraries(reproduce-event-hang commonlib)

target_link_libraries(reproduce-event-hang ze_loader OpenCL)