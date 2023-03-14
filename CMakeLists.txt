project(Color)

cmake_minimum_required (VERSION 2.6)
set(CMAKE_BUILD_TYPE Release)
find_package(CGAL QUIET COMPONENTS Core )
include( ${CGAL_USE_FILE} )
include_directories(${CGAL_INCLUDE_DIR})

find_package(OpenCV REQUIRED)

add_executable(Color color.cpp)
target_link_libraries(Color ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})        
target_link_libraries(Color ${OpenCV_LIBS} )                   
