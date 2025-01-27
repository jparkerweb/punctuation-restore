// -------------
// -- imports --
// -------------
import * as ort from 'onnxruntime-node';
import { parseSentences } from 'sentence-parse';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { downloadModel } from './modules/downloadModel.js';
import Tokenizer from './modules/tokenizer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// -----------------------------------
// -- The PunctuationRestorer class --
// -----------------------------------
class PunctuationRestorer {
    constructor() {
        this.session = null;
        this.tokenizer = new Tokenizer();
    }

    async initialize() {
        if (!this.session) {
            try {
                const { modelPath, tokenizerPath } = await downloadModel();
                this.session = await ort.InferenceSession.create(modelPath);
                await this.tokenizer.loadModel(tokenizerPath);
            } catch (error) {
                throw new Error(`Failed to initialize ONNX model: ${error.message}`);
            }
        }
    }

    async cleanup() {
        if (this.session) {
            await this.session.release();
            this.session = null;
        }
    }

    async restore(texts) {
        if (!Array.isArray(texts)) {
            throw new Error('Input must be an array of strings');
        }

        if (!this.session) {
            await this.initialize();
        }

        try {
            const results = [];
            for (const text of texts) {
                // Tokenize input
                const tokens = this.tokenizer.tokenize(text);
                
                // Convert to model input format
                const { inputIds, attentionMask } = this.tokenizer.toModelInput(tokens);
                
                try {
                    // Run inference
                    const feeds = {
                        'input_ids': new ort.Tensor('int64', inputIds, [1, inputIds.length]),
                        'attention_mask': new ort.Tensor('int64', attentionMask, [1, attentionMask.length])
                    };
                    
                    const outputMap = await this.session.run(feeds);
                    
                    // Post-process output
                    const result = this.tokenizer.postprocess(tokens, outputMap);
                    results.push(result);
                } catch (error) {
                    throw new Error(`Failed to process text: ${error.message}`);
                }
            }
            const sentences = [];
            for (const result of results) {
                sentences.push(...await parseSentences(result));
            }
            return sentences;
        } finally {
            await this.cleanup();
        }
    }
}


// ==========================================
// == Export the PunctuationRestorer class ==
// ==========================================
export default PunctuationRestorer;
