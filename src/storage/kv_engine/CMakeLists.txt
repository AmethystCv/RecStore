find_package(PkgConfig QUIET)

# KV engine factory
add_library(kv_engine INTERFACE)
target_link_libraries(kv_engine INTERFACE base)
if(USE_PMKV MATCHES "ON")
  target_link_libraries(kv_engine INTERFACE ${LIBPMEMKV_LIBRARIES})
endif()

target_link_libraries(
  kv_engine
  INTERFACE
  spdk_wrapper
  ${FOLLY_LIBRARIES}
)

target_sources(
  kv_engine INTERFACE kv_engine_register.cc)
# KV engine factory done
