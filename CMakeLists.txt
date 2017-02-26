cmake_minimum_required(VERSION 2.8)
project(darknet C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# Check for C++ version
# https://cmake.org/cmake/help/v3.0/module/CheckCXXCompilerFlag.html

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

## Compiler settings
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--export-dynamic -Wall -Wno-sign-compare -fPIC")
endif()

#thread
FIND_PACKAGE(Threads)
    list(APPEND LIBRARIES ${CMAKE_THREAD_LIBS_INIT})


################################################################################################
# Short command for cuDNN detection. Believe it soon will be a part of CUDA toolkit distribution.
# That's why not FindcuDNN.cmake file, but just the macro
# Usage:
#   detect_cuDNN()
function(detect_cuDNN)
    set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

    find_path(CUDNN_INCLUDE cudnn.h
    PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
    DOC "Path to cuDNN include directory." )

    # dynamic libs have different suffix in mac and linux
    if(APPLE)
        set(CUDNN_LIB_NAME "libcudnn.dylib")
    else()
        set(CUDNN_LIB_NAME "libcudnn.so")
    endif()

    get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
    find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIB_NAME}
    PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
    DOC "Path to cuDNN library.")

    if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
        set(HAVE_CUDNN  TRUE PARENT_SCOPE)
        set(CUDNN_FOUND TRUE PARENT_SCOPE)

        file(READ ${CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)

        # cuDNN v3 and beyond
        string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
        CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
        CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
        string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
        CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
        CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
        string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
        CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
        CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

        if(NOT CUDNN_VERSION_MAJOR)
            set(CUDNN_VERSION "???")
        else()
            set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
        endif()

        message(STATUS "Found cuDNN: ver. ${CUDNN_VERSION} found (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")

        string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 3 cuDNNVersionIncompatible)
        if(cuDNNVersionIncompatible)
            message(FATAL_ERROR "cuDNN version >3 is required.")
        endif()

        set(CUDNN_VERSION "${CUDNN_VERSION}" PARENT_SCOPE)
        mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)

    endif()
endfunction()

# CUDA
if (USE_GPU)
    FIND_PACKAGE(CUDA)
    if (CUDA_FOUND)

        include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
        list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_cudnn_LIBRARY} )
        list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-O2;-Xcompiler;-fPIC;")
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        add_definitions(-DGPU)
        cuda_include_directories(src)

        # cudnn detection
        if(USE_CUDNN)
            detect_cuDNN()
            if(HAVE_CUDNN)
                add_definitions(-DCUDNN)
                include_directories(SYSTEM ${CUDNN_INCLUDE})
                list(APPEND LIBRARIES ${CUDNN_LIBRARY})
            endif()
        endif()
    else()
        list(APPEND LIBRARIES "m")
    endif()
endif()



# OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
add_definitions(-DOPENCV)

include_directories(SYSTEM ${OpenCV_INCLUDE_DIR})

# Add sources

include_directories(src)
include_directories(SequenceParsing)
include_directories(SequenceParsing/tinydir)
include_directories(Serialization)
include_directories(Serialization/yaml-cpp/include)

set(SRC_DIR src)

FILE(GLOB HEADERS ${PROJECT_SOURCE_DIR}/${SRC_DIR}/*.h)
FILE(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/${SRC_DIR}/*.c)
LIST(REMOVE_ITEM SRC_FILES "${PROJECT_SOURCE_DIR}/src/server.c")
LIST(REMOVE_ITEM SRC_FILES "${PROJECT_SOURCE_DIR}/src/old.c")
#LIST(REMOVE_ITEM SRC_FILES "${PROJECT_SOURCE_DIR}/src/cpu_gemm.c")
LIST(REMOVE_ITEM SRC_FILES "${PROJECT_SOURCE_DIR}/src/darknet.c")
LIST(REMOVE_ITEM SRC_FILES "${PROJECT_SOURCE_DIR}/src/demo.c")
LIST(REMOVE_ITEM SRC_FILES "${PROJECT_SOURCE_DIR}/src/yolo.c")

set(SRC_FILES "${SRC_FILES};${PROJECT_SOURCE_DIR}/SequenceParsing/SequenceParsing.cpp;${PROJECT_SOURCE_DIR}/SequenceParsing/SequenceParsing.h;${PROJECT_SOURCE_DIR}/SequenceParsing/tinydir/tinydir.h")

set_source_files_properties(${SRC_FILES} PROPERTIES LANGUAGE CXX )

if (CUDA_FOUND)


    FILE(GLOB CU_SRC_FILES ${PROJECT_SOURCE_DIR}/${SRC_DIR}/*.cu)

    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)

    set(CUDA_ARCH_BIN "20 30 32 35 37 50 52" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
    set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")

    #include(CudaComputeTargetFlags)
    #APPEND_TARGET_ARCH_FLAGS()

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")

    # Compile kernels
    CUDA_COMPILE(cuda_objs ${CU_SRC_FILES})

endif()

# build Serialization lib
add_subdirectory(Serialization)


add_library(libdarknet STATIC ${SRC_FILES} ${HEADERS} ${CU_SRC_FILES} ${cuda_objs})
target_link_libraries(libdarknet ${LIBRARIES} ${OpenCV_LIBS})

add_executable(detector ${SRC_DIR}/detectSequence.c)

target_link_libraries(detector ${LIBRARIES} libdarknet)
target_link_libraries(detector ${LIBRARIES} autocam-serialization)

add_executable(compareDetections ${SRC_DIR}/compareDetections.c)

target_link_libraries(compareDetections ${LIBRARIES} libdarknet)
target_link_libraries(compareDetections ${LIBRARIES} autocam-serialization)

add_definitions("-DYAML=YAML_AUTOCAM")




