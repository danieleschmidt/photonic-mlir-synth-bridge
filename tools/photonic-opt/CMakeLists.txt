get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(photonic-opt
  photonic-opt.cpp
)

target_link_libraries(photonic-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRPhotonicDialect
  MLIRPhotonicTransforms
  MLIRTorchToPhotonic
  MLIRPhotonicToHLS
  MLIROptLib
)