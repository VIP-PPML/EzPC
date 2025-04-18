macro (add_test_OT _name)
	set(cuda_src
        "${PROJECT_SOURCE_DIR}/src/Millionaire/bob_cuda.cu"
        "${PROJECT_SOURCE_DIR}/src/Millionaire/millionaire.cu")
	add_executable(${_name}-OT "test_ring_${_name}.cpp" ${cuda_src})

	set_target_properties(${_name}-OT PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON 
		CXX_STANDARD 17
	)

    target_link_libraries(${_name}-OT SCI-OT) 
endmacro()

macro (add_test_HE _name)
    add_executable(${_name}-HE "test_field_${_name}.cpp")
    target_link_libraries(${_name}-HE SCI-HE) 
endmacro()

macro (add_test_float _name)
	add_executable(${_name}-float "test_float_${_name}.cpp")
    target_link_libraries(${_name}-float SCI-FloatingPoint) 
endmacro()

add_test_OT(matmul)
add_test_OT(value_extension)
add_test_OT(truncation)
add_test_OT(relu)
add_test_OT(argmax)
add_test_OT(aux_protocols)
add_test_OT(maxpool)
add_test_OT(hadamard_product)
add_test_OT(sigmoid)
add_test_OT(exp)
add_test_OT(tanh)
add_test_OT(sqrt)
add_test_OT(millionaire)

if(FALSE) 
add_test_HE(relu)
add_test_HE(maxpool)
add_test_HE(argmax)
add_test_HE(conv)
add_test_HE(fc)
add_test_HE(elemwise_prod)

add_test_float(bench_op)
add_test_float(primitive)
add_test_float(compound)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(MPFR 2.3.0 REQUIRED)
add_executable(math-float "test_float_math.cpp")
target_include_directories(math-float PUBLIC ${MPFR_INCLUDES}) 
target_link_libraries(math-float SCI-FloatingPoint ${MPFR_LIBRARIES}) 

add_subdirectory(GC)

macro(add_test_secfloat name)
	add_executable(${name}-secfloatml "test_floatml_${name}.cpp")
	target_link_libraries(${name}-secfloatml SCI-SecfloatML)
endmacro()

macro(add_test_beacon name)
	add_executable(${name}-beacon "test_floatml_${name}.cpp")
    target_link_libraries(${name}-beacon SCI-Beacon)
endmacro()

macro(add_test_floatml name)
	add_test_secfloat(${name})
	add_test_beacon(${name})
endmacro()

if(FALSE)
add_test_floatml(vsum)
add_test_floatml(dotprod)
add_test_floatml(matmul)
add_test_floatml(sigmoid)
add_test_floatml(softmax)
endif()
