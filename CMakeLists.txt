cmake_minimum_required(VERSION 2.8)
project(ddcrp-gibbs)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(Eigen3 REQUIRED)
find_package( Boost 1.40 COMPONENTS program_options REQUIRED )

include_directories(include)

add_library(${PROJECT_NAME} 
	src/CustomerAssignment.cpp
	src/ddCRP.cpp 
	src/DontcareLikelihood.cpp
	src/LikelihoodFcn.cpp
	src/MultivariateNormal.cpp
	)


add_executable(${PROJECT_NAME}-example src/main.cpp)
target_link_libraries(${PROJECT_NAME}-example ${PROJECT_NAME} ${Boost_LIBRARIES})
