#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "PhotonicMLIR/Dialect/PhotonicDialect.h"
#include "PhotonicMLIR/Conversion/PhotonicToHLS.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::photonic;

namespace py = pybind11;

PYBIND11_MODULE(PhotonicMLIRPythonModule, m) {
    m.doc() = "Photonic MLIR Python bindings";

    // HLS Configuration
    py::class_<HLSConfig>(m, "HLSConfig")
        .def(py::init<>())
        .def_readwrite("target_pdk", &HLSConfig::targetPDK)
        .def_readwrite("process_node", &HLSConfig::processNode)
        .def_readwrite("power_budget", &HLSConfig::powerBudget)
        .def_readwrite("wavelength_channels", &HLSConfig::wavelengthChannels)
        .def_readwrite("enable_thermal_optimization", &HLSConfig::enableThermalOptimization)
        .def_readwrite("enable_noise_reduction", &HLSConfig::enableNoiseReduction);

    // Core functions
    m.def("generate_hls_code", [](const std::string &mlirCode, const HLSConfig &config) {
        MLIRContext context;
        context.loadDialect<PhotonicDialect>();
        
        auto module = parseSourceString<ModuleOp>(mlirCode, &context);
        if (!module) {
            throw std::runtime_error("Failed to parse MLIR code");
        }
        
        return generateHLSCode(*module, config);
    }, "Generate HLS code from MLIR module");

    m.def("create_photonic_context", []() {
        auto context = std::make_unique<MLIRContext>();
        context->loadDialect<PhotonicDialect>();
        return context.release();
    }, "Create MLIR context with Photonic dialect loaded");
}