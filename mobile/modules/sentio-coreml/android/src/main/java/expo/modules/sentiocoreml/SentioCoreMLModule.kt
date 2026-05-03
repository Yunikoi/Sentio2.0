package expo.modules.sentiocoreml

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class SentioCoreMLModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("SentioCoreML")

    AsyncFunction("loadModelAsync") { }

    AsyncFunction("isLoaded") { false }

    AsyncFunction("predict") { _: Map<String, Any?> ->
      throw Exception("SentioCoreML is iOS-only; use a device build on iOS.")
    }
  }
}
