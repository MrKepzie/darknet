project(autocam-serialization)


set(SERIALIZATION_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SERIALIZATION_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB SERIALIZATION_HEADER_FILES "${SERIALIZATION_HEADER_DIR}/*.h")
file(GLOB SERIALIZATION_SRC_FILES "${SERIALIZATION_SRC_DIR}/*.cpp")
LIST(REMOVE_ITEM SERIALIZATION_SRC_FILES "${SERIALIZATION_SRC_DIR}/convertGroundTruth.cpp")

# Create serialization static lib
add_library(autocam-serialization STATIC ${SERIALIZATION_HEADER_FILES} ${SERIALIZATION_SRC_FILES})

add_executable(convertGroundTruth "${SERIALIZATION_SRC_DIR}/convertGroundTruth.cpp")

add_definitions("-DYAML=YAML_AUTOCAM")


# Find boost
find_package (Boost REQUIRED)
if(NOT Boost_FOUND)
    message(FATAL_ERROR "boost installation could not be found.")
endif()


# Build yaml-cpp
add_subdirectory(yaml-cpp)


# Include dirs
include_directories(${SERIALIZATION_HEADER_DIR})
include_directories(${YAML_CPP_HEADER_DIR})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../SupportExt" )

# Include boost
include_directories(${Boost_INCLUDE_DIR})

set_target_properties(autocam-serialization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(autocam-serialization yaml-cpp)

target_link_libraries(convertGroundTruth autocam-serialization)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set_target_properties(autocam-serialization PROPERTIES COMPILE_FLAGS_RELEASE "${COMMON_CXX_FLAGS} -O3 -DNDEBUG" LINK_FLAGS "-m${BITS}")
    set_target_properties(autocam-serialization PROPERTIES COMPILE_FLAGS_DEBUG "${COMMON_CXX_FLAGS} -g -DDEBUG" LINK_FLAGS "-m${BITS}")
endif()


# Install autocam-serialization headers and library
#install(
#    FILES ${SERIALIZATION_HEADER_FILES}
#    DESTINATION "include"
#)
#install(TARGETS autocam-serialization DESTINATION "lib" OPTIONAL)
