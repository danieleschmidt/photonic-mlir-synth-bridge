#ifndef PHOTONIC_DIALECT
#define PHOTONIC_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Photonic dialect definition.
//===----------------------------------------------------------------------===//

def Photonic_Dialect : Dialect {
    let name = "photonic";
    let summary = "A dialect for photonic computing operations";
    let description = [{
        The Photonic dialect defines operations, types, and attributes for
        silicon photonic computing. It provides first-class support for
        optical operations including interference, phase shifts, detection,
        and wavelength division multiplexing.

        Key features:
        - Complex-valued optical signals
        - Unitary transformations via Mach-Zehnder interferometers
        - Wavelength-parallel processing
        - Thermal and noise modeling
        - Integration with PyTorch computational graphs
    }];
    let cppNamespace = "::mlir::photonic";

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

#endif // PHOTONIC_DIALECT