// RUN: photonic-opt %s | photonic-opt | FileCheck %s

// CHECK-LABEL: func.func @test_mzi
func.func @test_mzi(%arg0: complex<f32>, %arg1: complex<f32>) -> (complex<f32>, complex<f32>) {
  %theta = arith.constant 1.57 : f32  // π/2
  %phi = arith.constant 0.0 : f32
  
  // CHECK: photonic.mzi
  %0, %1 = photonic.mzi %arg0, %arg1 phase (%theta, %phi) : (complex<f32>, complex<f32>) -> (complex<f32>, complex<f32>)
  
  return %0, %1 : complex<f32>, complex<f32>
}

// CHECK-LABEL: func.func @test_directional_coupler
func.func @test_directional_coupler(%arg0: complex<f32>, %arg1: complex<f32>) -> (complex<f32>, complex<f32>) {
  %coupling_ratio = arith.constant 0.5 : f32
  
  // CHECK: photonic.directional_coupler
  %0, %1 = photonic.directional_coupler %arg0, %arg1 ratio %coupling_ratio : (complex<f32>, complex<f32>) -> (complex<f32>, complex<f32>)
  
  return %0, %1 : complex<f32>, complex<f32>
}

// CHECK-LABEL: func.func @test_phase_shift
func.func @test_phase_shift(%arg0: complex<f32>) -> complex<f32> {
  %phase = arith.constant 3.14159 : f32  // π
  
  // CHECK: photonic.phase_shift
  %0 = photonic.phase_shift %arg0 by %phase : complex<f32> -> complex<f32>
  
  return %0 : complex<f32>
}

// CHECK-LABEL: func.func @test_photodetector
func.func @test_photodetector(%arg0: complex<f32>) -> f32 {
  %responsivity = arith.constant 0.8 : f32
  
  // CHECK: photonic.photodetector
  %0 = photonic.photodetector %arg0 responsivity %responsivity : complex<f32> -> f32
  
  return %0 : f32
}

// CHECK-LABEL: func.func @test_tensor_core
func.func @test_tensor_core(%arg0: tensor<4x4xcomplex<f32>>, %arg1: tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> {
  // CHECK: photonic.tensor_core
  %0 = photonic.tensor_core %arg0, %arg1 {
    wavelength_channels = 4,
    mesh_topology = "triangular",
    activation = "photodetector"
  } : (tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
  
  return %0 : tensor<4x4xcomplex<f32>>
}