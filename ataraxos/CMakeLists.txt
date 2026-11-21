cmake_minimum_required(VERSION 3.21)
project(ataraxos LANGUAGES CXX)

# libpkmn (Gen 1, Showdown mode, with the protocol log): oak's "client"
# build (dev/libpkmn), and oak's headers for its C++ views.
set(OAK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. CACHE PATH "The oak checkout")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(LIBPKMN_MODE Debug)
else()
  set(LIBPKMN_MODE ReleaseFast)
endif()
add_library(libpkmn_client STATIC IMPORTED)
set_target_properties(libpkmn_client PROPERTIES
  IMPORTED_LOCATION ${OAK_DIR}/cpp/libpkmn/${LIBPKMN_MODE}/client/libpkmn-showdown.a)

add_library(ataraxos INTERFACE)
target_include_directories(ataraxos INTERFACE src ${OAK_DIR}/cpp/include)
target_compile_definitions(ataraxos INTERFACE LOG CHANCE CALC)
target_link_libraries(ataraxos INTERFACE libpkmn_client)
target_compile_options(ataraxos INTERFACE -Wall -Wextra)

# The perfect-info env (src/pi_env.hpp): oak's libpkmn_chance build (no
# log), as oak's search links.
add_library(libpkmn_chance STATIC IMPORTED)
set_target_properties(libpkmn_chance PROPERTIES
  IMPORTED_LOCATION ${OAK_DIR}/cpp/libpkmn/${LIBPKMN_MODE}/chance/libpkmn-showdown.a)
add_library(ataraxos_pi_lib INTERFACE)
target_include_directories(ataraxos_pi_lib INTERFACE src ${OAK_DIR}/cpp/include)
target_compile_definitions(ataraxos_pi_lib INTERFACE CHANCE CALC)
target_link_libraries(ataraxos_pi_lib INTERFACE libpkmn_chance)
target_compile_options(ataraxos_pi_lib INTERFACE -Wall -Wextra)

find_package(Threads REQUIRED)
find_package(OpenMP)

# The checkers (tests/cpp/): one standalone program per file.
file(GLOB ATARAXOS_PROGRAMS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/*.cpp)
foreach(src ${ATARAXOS_PROGRAMS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE tests/cpp)
  target_link_libraries(${name} PRIVATE ataraxos Threads::Threads)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endforeach()

# The Python bindings, into ataraxos/_ext/ in the source tree. pybind11 is
# oak's submodule.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
add_subdirectory(${OAK_DIR}/cpp/extern/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11 EXCLUDE_FROM_ALL)
foreach(name ataraxos_env ataraxos_client)
  pybind11_add_module(${name} ataraxos/_ext/${name}.cpp)
  target_link_libraries(${name} PRIVATE ataraxos)
  set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ataraxos/_ext)
endforeach()
pybind11_add_module(ataraxos_pi ataraxos/_ext/ataraxos_pi.cpp)
target_link_libraries(ataraxos_pi PRIVATE ataraxos_pi_lib)
set_target_properties(ataraxos_pi PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ataraxos/_ext)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ataraxos_env PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(ataraxos_pi PRIVATE OpenMP::OpenMP_CXX)
endif()

# The torch-free net (src/net.hpp) for pkmn.ai's bundle (ataraxos/arena/):
# no libpkmn; Accelerate's GEMMs on Apple, Eigen's (oak's submodule) elsewhere.
pybind11_add_module(ataraxos_net ataraxos/_ext/ataraxos_net.cpp)
target_include_directories(ataraxos_net PRIVATE src)
target_compile_options(ataraxos_net PRIVATE -Wall -Wextra)
if(APPLE)
  target_link_libraries(ataraxos_net PRIVATE "-framework Accelerate")
else()
  target_include_directories(ataraxos_net SYSTEM PRIVATE ${OAK_DIR}/cpp/extern/eigen)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(ataraxos_net PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()
set_target_properties(ataraxos_net PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ataraxos/_ext)
