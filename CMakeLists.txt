cmake_minimum_required(VERSION 3.28)
string(REGEX REPLACE ".*/(.*)" "\\1" CURRENT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})
project(${CURRENT_FOLDER} LANGUAGES CXX CUDA)

set(CUDA_TOOLKIT_ROOT_DIR "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\bin")
set(CMAKE_CUDA_ARCHITECTURES 75)
# add_subdirectory(0_hello_world)
include_directories(include)
add_subdirectory(1_check_dimension)
add_subdirectory(2_grid_block)
add_subdirectory(5_thread_index)
add_subdirectory(10_reduceInteger)
add_subdirectory(11_globel_memory)
add_subdirectory(12_atomic)



