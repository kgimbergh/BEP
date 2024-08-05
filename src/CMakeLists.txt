set(SOURCES)
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

find_package(LAPACK REQUIRED)

foreach(source ${SOURCES})
    # Set name of executable
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})

    # Set profiling flags
    # target_compile_options(${name} PRIVATE -pg)
    # target_link_options(${name} PRIVATE -pg)
    
    # Set include directories
    if (INCLUDES)
        target_include_directories(${name} PRIVATE ${INCLUDES})
    endif()

    # Set compile options, enable warnings
    #target_compile_options(${name} PRIVATE -Wall -Wextra -pedantic)

    target_link_libraries(${name} Kokkos::kokkos Kokkos::kokkoskernels)
    target_link_libraries(${name} ${LAPACK_LIBRARIES})

endforeach()
