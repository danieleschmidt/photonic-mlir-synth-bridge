#ifndef PHOTONIC_OPS
#define PHOTONIC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "PhotonicDialect.td"
include "PhotonicTypes.td"

//===----------------------------------------------------------------------===//
// Photonic dialect operations.
//===----------------------------------------------------------------------===//

class Photonic_Op<string mnemonic, list<Trait> traits = []> :
        Op<Photonic_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Basic Photonic Operations
//===----------------------------------------------------------------------===//

def Photonic_MachZehnderOp : Photonic_Op<"mzi", [Pure]> {
    let summary = "Mach-Zehnder Interferometer operation";
    let description = [{
        Represents a 2x2 unitary transform using a Mach-Zehnder Interferometer.
        Takes two optical inputs and produces two optical outputs with configurable
        phase shifts for implementing arbitrary 2x2 unitary matrices.
    }];

    let arguments = (ins 
        ComplexType:$input1, 
        ComplexType:$input2,
        F32:$theta,
        F32:$phi
    );
    let results = (outs ComplexType:$output1, ComplexType:$output2);

    let assemblyFormat = [{
        $input1 `,` $input2 `phase` `(` $theta `,` $phi `)` attr-dict `:` 
        `(` type($input1) `,` type($input2) `)` `->` 
        `(` type($output1) `,` type($output2) `)`
    }];
}

def Photonic_DirectionalCouplerOp : Photonic_Op<"directional_coupler", [Pure]> {
    let summary = "Directional coupler for beam splitting";
    let description = [{
        Implements a directional coupler that splits optical signals between
        two waveguides with a configurable coupling ratio.
    }];

    let arguments = (ins 
        ComplexType:$input1, 
        ComplexType:$input2,
        F32:$coupling_ratio
    );
    let results = (outs ComplexType:$output1, ComplexType:$output2);

    let assemblyFormat = [{
        $input1 `,` $input2 `ratio` $coupling_ratio attr-dict `:` 
        `(` type($input1) `,` type($input2) `)` `->` 
        `(` type($output1) `,` type($output2) `)`
    }];
}

def Photonic_PhaseShiftOp : Photonic_Op<"phase_shift", [Pure]> {
    let summary = "Optical phase shift operation";
    let description = [{
        Applies a phase shift to an optical signal using thermal or electro-optic
        phase shifters.
    }];

    let arguments = (ins ComplexType:$input, F32:$phase);
    let results = (outs ComplexType:$output);

    let assemblyFormat = [{
        $input `by` $phase attr-dict `:` type($input) `->` type($output)
    }];
}

def Photonic_PhotodetectorOp : Photonic_Op<"photodetector", [Pure]> {
    let summary = "Optical to electrical conversion";
    let description = [{
        Converts optical signal to electrical signal through photodetection.
        Computes the square magnitude of the complex optical field.
    }];

    let arguments = (ins ComplexType:$optical_input, F32:$responsivity);
    let results = (outs F32:$electrical_output);

    let assemblyFormat = [{
        $optical_input `responsivity` $responsivity attr-dict `:` 
        type($optical_input) `->` type($electrical_output)
    }];
}

def Photonic_TensorCoreOp : Photonic_Op<"tensor_core", [Pure]> {
    let summary = "Photonic tensor core for matrix multiplication";
    let description = [{
        Implements photonic matrix multiplication using mesh of Mach-Zehnder
        interferometers with wavelength division multiplexing.
    }];

    let arguments = (ins 
        AnyTensor:$lhs,
        AnyTensor:$rhs,
        I32Attr:$wavelength_channels,
        StrAttr:$mesh_topology,
        StrAttr:$activation
    );
    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` 
        `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];
}

#endif // PHOTONIC_OPS