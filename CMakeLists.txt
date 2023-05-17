# Project info ------------------------------------------------------------
cmake_minimum_required (VERSION 3.0)

project(Color)
# -------------------------------------------------------------------------

# Set (remember to change the path for what you need)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_PREFIX_PATH "E:/dev") #can be removed if install under directory locally
list(APPEND CMAKE_PREFIX_PATH "E:/dev/OpenCV/build") #can be removed if install under directory locally
# -------------------------------------------------------------------------

# include CGAL and OpenCV library -----------------------------------------
find_package(CGAL QUIET COMPONENTS Core )
include( ${CGAL_USE_FILE} )
include_directories(${CGAL_INCLUDE_DIR})
find_package(OpenCV REQUIRED)
# -------------------------------------------------------------------------

add_executable(Color color.cpp)
target_link_libraries(Color ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})        
target_link_libraries(Color ${OpenCV_LIBS} )                   
