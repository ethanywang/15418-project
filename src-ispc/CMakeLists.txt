# check ISPC executable
find_program(ISPC_EXECUTABLE ispc)
if (NOT ISPC_EXECUTABLE)
    message(FATAL_ERROR "Failed to find ispc")
endif ()

# set ISPC flags
set(ISPC_FLAGS "-O2")
set(ISPC_TARGETS "avx1-i32x8")
set(ISPC_ARCH "x86-64")

# set ISPC source name
set(ALL_ISPC_ROOT_NAMES "matrix" "sigmoid" "tanh")
set(ALL_ISPC_SRC_NAMES)
set(ALL_ISPC_BUILD_OUTPUTS)

foreach (ISPC_SRC_NAME ${ALL_ISPC_ROOT_NAMES})
    set(ISPC_HEADER_NAME "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}_ispc.h")
    set(ISPC_OBJ_NAME "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}_ispc${CMAKE_CXX_OUTPUT_EXTENSION}")
    set(ISPC_BUILD_OUTPUT ${ISPC_HEADER_NAME} ${ISPC_OBJ_NAME})

    set(ALL_ISPC_SRC_NAMES ${ALL_ISPC_SRC_NAMES} "${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc")
    set(ALL_ISPC_BUILD_OUTPUTS ${ALL_ISPC_BUILD_OUTPUTS} ${ISPC_BUILD_OUTPUT})

    # ISPC command
    add_custom_command(
            OUTPUT ${ISPC_BUILD_OUTPUT}
            COMMAND ${ISPC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc" ${ISPC_FLAGS}
            --target=${ISPC_TARGETS} --arch=${ISPC_ARCH} --math-lib=default
            -h ${ISPC_HEADER_NAME} -o ${ISPC_OBJ_NAME}
            VERBATIM
            DEPENDS ${ISPC_EXECUTABLE}
            DEPENDS "${ISPC_SRC_NAME}.ispc")
endforeach ()

# set extra common CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

# set common target source
set(CXX_SOURCES
    gru.cpp
    lstm.cpp
    matrix.cpp
    sigmoid.cpp
    tanh.cpp
    tasksys.cpp
)

# add the executable
add_executable(rnn-ispc main.cpp ${ALL_ISPC_BUILD_OUTPUTS} ${ALL_ISPC_SRC_NAMES})
target_sources(rnn-ispc PRIVATE ${CXX_SOURCES})
target_include_directories(rnn-ispc PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(rnn-ispc pthread)

# export to library
add_library(libispc STATIC
    ${ALL_ISPC_BUILD_OUTPUTS}
    ${CXX_SOURCES}
)
target_include_directories(libispc PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
