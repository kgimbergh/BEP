cmake_minimum_required(VERSION 3.16)
project(bep_project)

if (MSVC)
    add_compile_options(/bigobj)
endif() 
add_subdirectory(kokkos)
add_subdirectory(kokkos-kernels)
add_subdirectory(matplotplusplus)

add_subdirectory(src)
