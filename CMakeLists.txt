
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)


set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_TOOLCHAIN_FILE "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_PREFIX_PATH  "${CMAKE_PREFIX_PATH} C:/dev/vcpkg/installed/x64-windows")

set (CMAKE_CXX_STANDARD 20)
project(HessianAffine LANGUAGES CXX)

find_package(OpenCV REQUIRED)

include (GenerateExportHeader)

add_library(
	HessianAffine 
SHARED
	src/Utils.cpp
	src/HessianPyramid.cpp
	src/HessianAffineDetector.cpp
	src/HessianDetector.cpp
	src/AffineDeformer.cpp 
	)

add_dependencies(
	HessianAffine
	${OpenCV_LIBS}
)
GENERATE_EXPORT_HEADER(HessianAffine
    EXPORT_MACRO_NAME HA_API
    EXPORT_FILE_NAME ha_export.hpp
)
target_link_libraries(HessianAffine 
PUBLIC
	${OpenCV_LIBS}
	)

target_include_directories(HessianAffine
PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
PRIVATE
)

target_compile_definitions(HessianAffine
PRIVATE
	HessianAffine_EXPORTS
)
#add_executable(HATest 
	#src/main.cpp
#)

#target_link_libraries(HATest
	#HessianAffine
	#${OpenCV_LIBS}
#)

install(TARGETS HessianAffine
	DESTINATION ${CMAKE_INSTALL_PREFIX}
	RUNTIME_DEPENDENCIES 
	PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
	POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
)