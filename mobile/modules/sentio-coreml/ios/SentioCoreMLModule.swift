import CoreML
import ExpoModulesCore

public final class SentioCoreMLModule: Module {
  private var model: MLModel?

  public func definition() -> ModuleDefinition {
    Name("SentioCoreML")

    AsyncFunction("loadModelAsync") { () throws in
      if self.model != nil {
        return
      }
      let bundle = Bundle(for: SentioCoreMLModule.self)
      var url = bundle.url(forResource: "FallDetectorRowRF", withExtension: "mlmodelc")
      if url == nil {
        url = bundle.url(forResource: "FallDetectorRowRF", withExtension: "mlmodel")
      }
      guard let modelUrl = url else {
        throw NSError(
          domain: "SentioCoreML",
          code: 1,
          userInfo: [NSLocalizedDescriptionKey: "FallDetectorRowRF.mlmodel not in module bundle (run prebuild with config plugin)"]
        )
      }
      if modelUrl.pathExtension == "mlmodel" {
        let compiled = try MLModel.compileModel(at: modelUrl)
        self.model = try MLModel(contentsOf: compiled)
      } else {
        self.model = try MLModel(contentsOf: modelUrl)
      }
    }

    AsyncFunction("isLoaded") { () -> Bool in
      self.model != nil
    }

    AsyncFunction("predict") { (raw: [String: Any]) throws -> [String: Any?] in
      guard let m = self.model else {
        throw NSError(
          domain: "SentioCoreML",
          code: 2,
          userInfo: [NSLocalizedDescriptionKey: "Call loadModelAsync() before predict"]
        )
      }

      var dict: [String: NSNumber] = [:]
      dict.reserveCapacity(raw.count)
      for (key, value) in raw {
        if let d = value as? Double {
          dict[key] = NSNumber(value: d)
        } else if let n = value as? NSNumber {
          dict[key] = n
        }
      }

      let input = try MLDictionaryFeatureProvider(dictionary: dict)
      let output = try m.prediction(from: input)

      var result: [String: Any?] = [:]
      if let fv = output.featureValue(for: "fall") {
        result["label"] = fv.int64Value
      }
      if let fv = output.featureValue(for: "classProbability") {
        let d = fv.dictionaryValue
        if let map = d as? [Int64: Double] {
          result["probability"] = map[1]
        } else if let map = d as? [NSNumber: NSNumber] {
          result["probability"] = map[NSNumber(value: 1)]?.doubleValue
        }
      }
      return result
    }
  }
}
