project(yaml-cpp)

set(YAML_CPP_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(YAML_CPP_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE YAML_CPP_HEADER_FILES "${YAML_CPP_HEADER_DIR}/*.h")
file(GLOB_RECURSE YAML_CPP_PRIVATE_HEADER_FILES "${YAML_CPP_SRC_DIR}/*.h")
file(GLOB_RECURSE YAML_CPP_SRC_FILES "${YAML_CPP_SRC_DIR}/*.cpp")

LIST(REMOVE_ITEM YAML_CPP_SRC_FILES "${YAML_CPP_SRC_DIR}/contrib/graphbuilder.cpp")
LIST(REMOVE_ITEM YAML_CPP_SRC_FILES "${YAML_CPP_SRC_DIR}/contrib/graphbuilderadapter.cpp")


# Create yaml-cpp static lib
add_library(yaml-cpp STATIC ${YAML_CPP_HEADER_FILES} ${YAML_CPP_PRIVATE_HEADER_FILES} ${YAML_CPP_SRC_FILES})



# Include dirs
target_include_directories(yaml-cpp PUBLIC ${YAML_CPP_HEADER_DIR} ${YAML_CPP_SRC_DIR})

# Include boost
include_directories(${Boost_INCLUDE_DIR})

if(NOT BITS)
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(BITS "64")
    else()
        set(BITS "32")
    endif()
endif()


# CPP flags on debug / release mode
add_definitions("-DYAML_CPP_NO_CONTRIB")
if(MSVC)
    add_definitions("-DWINDOWS -DNOMINMAX")
endif()

set_target_properties(yaml-cpp PROPERTIES LINKER_LANGUAGE CXX)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    add_definitions("-Wno-unused-parameter -Wno-unused-local-typedef  -Wno-missing-declarations -pedantic -Wno-long-long")
    set(COMMON_CXX_FLAGS "-fPIC -m${BITS}")
    set_target_properties(yaml-cpp PROPERTIES COMPILE_FLAGS_RELEASE "${COMMON_CXX_FLAGS} -O3 -DNDEBUG" LINK_FLAGS "-m${BITS}")
    set_target_properties(yaml-cpp PROPERTIES COMPILE_FLAGS_DEBUG "${COMMON_CXX_FLAGS} -g -DDEBUG" LINK_FLAGS "-m${BITS}")
endif()


# Install yaml-cpp headers and library
#install(
#    FILES ${YAML_CPP_HEADER_FILES}
#    DESTINATION "include"
#)
#install(TARGETS yaml-cpp DESTINATION "lib" OPTIONAL)
