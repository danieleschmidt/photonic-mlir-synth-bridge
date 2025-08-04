add_llvm_executable(photonic-translate
  photonic-translate.cpp
)

target_link_libraries(photonic-translate PRIVATE
  MLIRPhotonicDialect
  MLIRPhotonicToHLS
  MLIRTranslateLib
)