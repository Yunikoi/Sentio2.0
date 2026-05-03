/**
 * Expo config plugin: copy Core ML model into the local Expo module before native prebuild.
 *
 * Source (tracked in git): assets/coreml/FallDetectorRowRF.mlmodel
 * Destination (consumed by CocoaPods): modules/sentio-coreml/ios/FallDetectorRowRF.mlmodel
 */
const fs = require('fs');
const path = require('path');

const { withDangerousMod } = require('@expo/config-plugins');

function withSentioCoreML(config) {
  return withDangerousMod(config, [
    'ios',
    async (cfg) => {
      const projectRoot = cfg.modRequest.projectRoot;
      const src = path.join(projectRoot, 'assets', 'coreml', 'FallDetectorRowRF.mlmodel');
      const destDir = path.join(projectRoot, 'modules', 'sentio-coreml', 'ios');
      const dest = path.join(destDir, 'FallDetectorRowRF.mlmodel');

      if (!fs.existsSync(src)) {
        console.warn(
          '[withSentioCoreML] Skip copy: model not found at',
          src,
          '(train with --export-coreml or run scripts/export_row_rf_coreml.py)',
        );
        return cfg;
      }

      await fs.promises.mkdir(destDir, { recursive: true });
      await fs.promises.copyFile(src, dest);
      return cfg;
    },
  ]);
}

module.exports = withSentioCoreML;
