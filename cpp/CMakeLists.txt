cmake_minimum_required(VERSION 3.10)
project(oak VERSION 1.1.2)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(extern/lrsnash EXCLUDE_FROM_ALL)
add_subdirectory(extern/eigen EXCLUDE_FROM_ALL)
add_subdirectory(extern/argparse EXCLUDE_FROM_ALL)

# Libpkmn

function(create_library target mode dir)

  add_library(${target} STATIC IMPORTED)

  if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LIB_EXT lib)
    set(LIB_PREFIX "")
  else()
    set(LIB_EXT a)
    set(LIB_PREFIX lib)
  endif()

  set_target_properties(${target} PROPERTIES
    IMPORTED_LOCATION
      ${CMAKE_CURRENT_SOURCE_DIR}/libpkmn/${mode}/${dir}/${LIB_PREFIX}pkmn-showdown.${LIB_EXT}
  )

  target_compile_options(${target} INTERFACE -mavx2)

endfunction()

create_library(libpkmn_client_release ReleaseFast client)
create_library(libpkmn_client_debug Debug client)
create_library(libpkmn_chance_release ReleaseFast chance)
create_library(libpkmn_chance_debug Debug chance)

add_library(libpkmn_client INTERFACE)
target_link_libraries(libpkmn_client INTERFACE
  $<$<CONFIG:Debug>:libpkmn_client_debug>
  $<$<CONFIG:Release>:libpkmn_client_release>
)
add_library(libpkmn_chance INTERFACE)
target_link_libraries(libpkmn_chance INTERFACE
  $<$<CONFIG:Debug>:libpkmn_chance_debug>
  $<$<CONFIG:Release>:libpkmn_chance_release>
)

target_compile_definitions(libpkmn_client INTERFACE LOG CHANCE CALC)
target_compile_definitions(libpkmn_chance INTERFACE CHANCE CALC)

## Pybind

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Targets

add_library(search_lib STATIC src/search.cc)
target_include_directories(search_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(search_lib PUBLIC libpkmn_chance lrslibgmp eigen)
set_target_properties(search_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(benchmark src/benchmark.cc)
target_link_libraries(benchmark PRIVATE search_lib argparse)

add_executable(search-test src/search-test.cc)
target_link_libraries(search-test PRIVATE search_lib argparse)

add_executable(chall src/chall.cc)
target_link_libraries(chall PRIVATE search_lib argparse)

add_executable(generate src/generate.cc)
target_link_libraries(generate PRIVATE search_lib argparse)

add_executable(vs src/vs.cc)
target_link_libraries(vs PRIVATE search_lib argparse)

add_library(pyoak SHARED src/pyoak.cc)
target_include_directories(pyoak PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pyoak PRIVATE libpkmn_chance pybind11::module eigen lrslibgmp)
set_target_properties(pyoak PROPERTIES PREFIX "" SUFFIX ".so")
set_target_properties(pyoak PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pyoaktrain SHARED src/pyoaktrain.cc)
target_include_directories(pyoaktrain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pyoaktrain PRIVATE libpkmn_chance pybind11::module eigen)
set_target_properties(pyoaktrain PROPERTIES PREFIX "" SUFFIX ".so")
set_target_properties(pyoaktrain PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pyoaksearch SHARED src/pyoaksearch.cc)
target_link_libraries(pyoaksearch PRIVATE search_lib pybind11::module)
set_target_properties(pyoaksearch PROPERTIES PREFIX "" SUFFIX ".so")

add_library(pyoaklog SHARED src/pyoaklog.cc)
target_include_directories(pyoaklog PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pyoaklog PRIVATE libpkmn_client pybind11::module)
set_target_properties(pyoaklog PROPERTIES PREFIX "" SUFFIX ".so")
set_target_properties(pyoaklog PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(log src/log.cc)
target_include_directories(log PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(log PRIVATE libpkmn_client)

add_executable(dur src/dur.cc)
target_include_directories(dur PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dur PRIVATE libpkmn_client lrslibgmp)

# add_executable(randbat src/randbat.cc)
# target_link_libraries(randbat PRIVATE libpkmn_chance)