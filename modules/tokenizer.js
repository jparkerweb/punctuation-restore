// Basic tokenizer for the punctuation restoration model
import PostProcessor from './postProcessor.js';


// ---------------------
// -- Tokenizer class --
// ---------------------
class Tokenizer {
    constructor() {
        this.maxLength = 512;
        this.specialTokens = {
            PAD: '[PAD]',
            UNK: '[UNK]',
            CLS: '[CLS]',
            SEP: '[SEP]'
        };
        this.postProcessor = new PostProcessor();
        this.model = null;
    }

    // Basic preprocessing
    preprocess(text) {
        // Convert to lowercase as the model will handle casing
        return text.toLowerCase()
            // Remove extra whitespace
            .replace(/\s+/g, ' ')
            // Remove existing punctuation
            .replace(/[.,!?;:]/g, '')
            .trim();
    }

    // Convert text to input features
    tokenize(text) {
        const preprocessed = this.preprocess(text);
        const words = preprocessed.split(' ');
        
        // Add special tokens
        const tokens = [this.specialTokens.CLS, ...words, this.specialTokens.SEP];
        
        // Truncate if needed
        if (tokens.length > this.maxLength) {
            console.warn(`Input text truncated from ${tokens.length} to ${this.maxLength} tokens`);
            tokens.splice(this.maxLength - 1, tokens.length - this.maxLength + 1, this.specialTokens.SEP);
        }
        
        return tokens;
    }

    // Convert tokens to model input tensors
    toModelInput(tokens) {
        // Create input IDs (simplified - in production would use actual vocabulary)
        const inputIds = tokens.map(token => {
            if (token in this.specialTokens) {
                return Object.keys(this.specialTokens).indexOf(token);
            }
            return 1; // UNK token ID
        });

        // Create attention mask (1 for real tokens, 0 for padding)
        const attentionMask = new Array(this.maxLength).fill(0);
        for (let i = 0; i < tokens.length; i++) {
            attentionMask[i] = 1;
        }

        // Pad input IDs
        while (inputIds.length < this.maxLength) {
            inputIds.push(0); // PAD token ID
        }

        return {
            inputIds,
            attentionMask
        };
    }

    async loadModel(modelPath) {
        // For now just store the path, actual model loading will be implemented later
        this.model = modelPath;
    }

    // Post-process model outputs to restore punctuation and casing
    postprocess(tokens, modelOutputs) {
        return this.postProcessor.process(tokens, modelOutputs);
    }
}


// ================================
// == Export the Tokenizer class ==
// ================================
export default Tokenizer;
