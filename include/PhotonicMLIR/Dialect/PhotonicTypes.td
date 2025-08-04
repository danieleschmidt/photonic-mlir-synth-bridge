#ifndef PHOTONIC_TYPES
#define PHOTONIC_TYPES

include "mlir/IR/AttrTypeBase.td"
include "PhotonicDialect.td"

//===----------------------------------------------------------------------===//
// Photonic dialect types.
//===----------------------------------------------------------------------===//

class Photonic_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Photonic_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Photonic_WavelengthType : Photonic_Type<"Wavelength", "wavelength"> {
    let summary = "Optical wavelength type";
    let description = [{
        Represents an optical wavelength in nanometers. Used for wavelength
        division multiplexing and optical routing.
    }];

    let parameters = (ins "float":$wavelength_nm);
    let assemblyFormat = "`<` $wavelength_nm `>`";

    let builders = [
        TypeBuilder<(ins "float":$wavelength), [{
            return $_get($_ctxt, wavelength);
        }]>
    ];
}

def Photonic_OpticalType : Photonic_Type<"Optical", "optical"> {
    let summary = "Optical signal type";
    let description = [{
        Represents a complex-valued optical signal at a specific wavelength.
        Combines amplitude and phase information for coherent optical processing.
    }];

    let parameters = (ins "Type":$element_type, "float":$wavelength_nm);
    let assemblyFormat = "`<` $element_type `,` $wavelength_nm `>`";

    let builders = [
        TypeBuilder<(ins "Type":$elementType, "float":$wavelength), [{
            return $_get($_ctxt, elementType, wavelength);
        }]>
    ];
}

def Photonic_WDMType : Photonic_Type<"WDM", "wdm"> {
    let summary = "Wavelength Division Multiplexed signal type";
    let description = [{
        Represents multiple optical signals multiplexed onto different
        wavelengths in a single optical fiber or waveguide.
    }];

    let parameters = (ins "Type":$element_type, "ArrayRef<float>":$wavelengths);
    let assemblyFormat = "`<` $element_type `,` `[` $wavelengths `]` `>`";

    let builders = [
        TypeBuilder<(ins "Type":$elementType, "ArrayRef<float>":$wavelengths), [{
            return $_get($_ctxt, elementType, wavelengths);
        }]>
    ];
}

#endif // PHOTONIC_TYPES