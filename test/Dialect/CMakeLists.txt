configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(PHOTONIC_TEST_DEPENDS
  FileCheck count not
  photonic-opt
  )

add_lit_testsuite(check-photonic-dialect "Running the Photonic dialect regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${PHOTONIC_TEST_DEPENDS}
  )