set(Torch_DIR "/opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/")
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

add_subdirectory(proto)

set(common 
    flatc.h
    parameters.h
    parameters.cpp
)


add_library(grpc_ps_client 
    grpc_ps_client.cpp grpc_ps_client.h ${common} 
)

target_link_libraries(
    grpc_ps_client PUBLIC
    base
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    # ${TORCH_LIBRARIES}
)

add_executable(grpc_ps_server_async
    grpc_ps_server_async.cpp 
    grpc_ps_server_async.h 
    cache_ps_impl.h
    ${common}
)

target_link_libraries(grpc_ps_server_async PUBLIC
    base
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    stdc++fs
    kv_engine
)

target_include_directories(grpc_ps_server_async PRIVATE ${PROJECT_SOURCE_DIR}/third_party/atomic_queue/include)

#add_library(grpc_ps_client_python SHARED
add_library(grpc_ps_client_python MODULE
    grpc_ps_client_python.cpp
)

target_include_directories(grpc_ps_client_python PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(grpc_ps_client_python PUBLIC
    grpc_ps_client
    ${TORCH_LIBRARIES}
)

target_link_libraries(grpc_ps_client_python PRIVATE Python3::Python)

add_library(python_cpp_test MODULE
    python_cpp_test.cpp
)

target_include_directories(python_cpp_test PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(python_cpp_test PUBLIC
    ${TORCH_LIBRARIES}
)
target_link_libraries(python_cpp_test PRIVATE Python3::Python)
