find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(PhotonicMLIRPythonModule
  bindings/PhotonicMLIRModule.cpp
)

target_link_libraries(PhotonicMLIRPythonModule PRIVATE
  MLIRPhotonicDialect
  MLIRPhotonicTransforms
  MLIRTorchToPhotonic
  MLIRPhotonicToHLS
  MLIRIR
  MLIRParser
  MLIRPass
)

target_compile_definitions(PhotonicMLIRPythonModule PRIVATE
  VERSION_INFO=${EXAMPLE_VERSION_INFO}
)