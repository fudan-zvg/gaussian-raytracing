cmake_minimum_required(VERSION 3.18)

project(GaussianTracer
	VERSION 1.0
	DESCRIPTION "Gaussian ray-tracer"
	LANGUAGES C CXX CUDA
)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

###############################################################################
# Dependencies
###############################################################################

set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

list(APPEND INCLUDE_DIRECTORIES "include/optix" "include/glm")

###############################################################################
# Program
###############################################################################

list(APPEND INCLUDE_DIRECTORIES "include")
list(APPEND GSTRACER_LIBRARIES ${CMAKE_DL_LIBS})
list(APPEND GSTRACER_SOURCES
	src/bvh.cu
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_DIR "${CUDA_COMPILER_BIN}" DIRECTORY)
set(CUDA_INCLUDE "${CUDA_DIR}/include")


###############################################################################
# Optix
###############################################################################

add_library(optix_program OBJECT
	src/optix/gaussiantrace_forward.cu
	src/optix/gaussiantrace_backward.cu
)

set_target_properties(optix_program PROPERTIES CUDA_PTX_COMPILATION ON CUDA_ARCHITECTURES OFF)
target_compile_definitions(optix_program PUBLIC ${GSTRACER_DEFINITIONS} -DTCNN_MIN_GPU_ARCH=0)
target_compile_options(optix_program PUBLIC "--expt-relaxed-constexpr")

target_include_directories(optix_program PUBLIC ${INCLUDE_DIRECTORIES})

# OptiX programs will be compiled as PTX and packaged
# as headers to be included from the binary dir.
list(APPEND INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}")
set(OPTIX_PTX_HEADER ${CMAKE_CURRENT_BINARY_DIR}/optix_ptx.h)

find_program(bin_to_c NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
if (NOT bin_to_c)
	message(FATAL_ERROR
		"bin2c not found:\n"
		"  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
		"  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
	)
endif()

add_custom_command(
	OUTPUT "${OPTIX_PTX_HEADER}"
	COMMAND ${CMAKE_COMMAND}
	"-DBIN_TO_C_COMMAND=${bin_to_c}"
	"-DOBJECTS=$<TARGET_OBJECTS:optix_program>"
	"-DOUTPUT=${OPTIX_PTX_HEADER}"
	-P ${PROJECT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
	VERBATIM
	DEPENDS optix_program
	COMMENT "Converting PTX files to a C header"
)

list(APPEND GSTRACER_SOURCES ${OPTIX_PTX_HEADER})

add_library(gstracer STATIC ${GSTRACER_SOURCES})
set_target_properties(gstracer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gstracer PUBLIC ${GSTRACER_DEFINITIONS})
target_compile_options(gstracer PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(gstracer PUBLIC ${INCLUDE_DIRECTORIES})
target_link_libraries(gstracer PUBLIC ${GSTRACER_LIBRARIES})
