import CoreML
import Foundation

/// Reference helper: copy into your Xcode iOS target after adding ``FallDetectorRowRF.mlmodel``
/// (from ``python scripts/export_row_rf_coreml.py``) to the target with "Copy Bundle Resources".
enum FallDetectorCoreML {
    private static var _model: MLModel?

    static func load() throws -> MLModel {
        if let m = _model { return m }
        guard let url = Bundle.main.url(forResource: "FallDetectorRowRF", withExtension: "mlmodelc") else {
            guard let src = Bundle.main.url(forResource: "FallDetectorRowRF", withExtension: "mlmodel") else {
                throw NSError(
                    domain: "FallDetectorCoreML",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "FallDetectorRowRF.mlmodel(c) not in bundle"]
                )
            }
            let compiled = try MLModel.compileModel(at: src)
            let m = try MLModel(contentsOf: compiled)
            _model = m
            return m
        }
        let m = try MLModel(contentsOf: url)
        _model = m
        return m
    }

    static func predict(features: [String: Double]) throws -> (label: Int, pFall: Double?) {
        let model = try load()
        let input = try MLDictionaryFeatureProvider(dictionary: features)
        let out = try model.prediction(from: input)
        let fv = out.featureValue(for: "fall")
        let lab = Int(fv?.int64Value ?? Int64(fv?.intValue ?? 0))
        var pFall: Double?
        if let dict = out.featureValue(for: "classProbability")?.dictionaryValue as? [Int64: Double] {
            pFall = dict[1]
        }
        return (lab, pFall)
    }
}
