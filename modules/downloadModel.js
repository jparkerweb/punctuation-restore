// -------------
// -- imports --
// -------------
import https from 'https';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ---------------------
// --- configuration ---
// ---------------------
const MODEL_OWNER = '1-800-BAD-CODE';
const MODEL_NAME = 'punctuation_fullstop_truecase_english';
const MODEL_FILE_SOURCE_TARGET = [
  {
    "source_filename": "punct_cap_seg_en.onnx",
    "target_filename": "model.onnx"
  },
  {
    "source_filename": "spe_32k_lc_en.model",
    "target_filename": "tokenizer.model"
  }
];

// Hugging Face endpoints
const BASE_RAW_URL = `https://huggingface.co/${MODEL_OWNER}/${MODEL_NAME}/resolve/main`;
// Directory to store downloaded models
const MODELS_DIR = path.join(__dirname, `../models/${MODEL_OWNER}/${MODEL_NAME}`);


// -----------------------------------
// -- download file helper function --
// -----------------------------------
function downloadFile(url, dest, callback) {
  https
    .get(url, (res) => {
      // If we get a redirect, follow it
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return downloadFile(res.headers.location, dest, callback);
      }

      if (res.statusCode !== 200) {
        return callback(
          new Error(`Failed to download. Status: ${res.statusCode} - URL: ${url}`)
        );
      }

      // Pipe the response data into the file
      const fileStream = fs.createWriteStream(dest);
      res.pipe(fileStream);

      fileStream.on('finish', () => {
        fileStream.close(callback);
      });
    })
    .on('error', (err) => {
      callback(err);
    });
}


// -------------------------
// -- download model file --
// -------------------------
export async function downloadModel() {
  const paths = {};
  let pending = MODEL_FILE_SOURCE_TARGET.length;

  return new Promise((resolve, reject) => {
    MODEL_FILE_SOURCE_TARGET.forEach(({ source_filename, target_filename }) => {
      const sourceUrl = `${BASE_RAW_URL}/${source_filename}`;
      const localFilePath = path.join(MODELS_DIR, target_filename);
      paths[target_filename === 'model.onnx' ? 'modelPath' : 'tokenizerPath'] = localFilePath;

      // Create the models directory if it doesn't exist
      fs.mkdirSync(MODELS_DIR, { recursive: true });

      // Check if file already exists
      if (fs.existsSync(localFilePath)) {
        // console.log(`${target_filename} already exists, skipping download...`);
        pending--;
        if (pending === 0) {
          // console.log('All files present!');
          resolve(paths);
        }
        return;
      }

      // Download the file
      downloadFile(sourceUrl, localFilePath, (err) => {
        // show a log message of the file we are downloading
        console.log(`Downloading ${target_filename}...`);

        if (err) {
          reject(err);
          return;
        }

        pending--;
        if (pending === 0) {
          console.log('All downloads complete!');
          resolve(paths);
        }
      });
    });
  });
}
