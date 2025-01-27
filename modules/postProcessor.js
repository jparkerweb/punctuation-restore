const PUNCTUATION_MARKS = {
    NONE: 0,
    PERIOD: 1,
    COMMA: 2,
    QUESTION: 3,
    // EXCLAMATION: 4  // Uncomment if your model supports exclamation
  };
  
  const PUNCTUATION_MAP = {
    0: '',
    1: '.',
    2: ',',
    3: '?',
    // 4: '!' // Uncomment if your model supports exclamation
  };
  
  // Words that often precede commas
  const COMMA_TRIGGERS = new Set([
    'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
    'however', 'therefore', 'moreover', 'furthermore',
    'nevertheless', 'meanwhile', 'consequently',
    'instead', 'indeed', 'namely', 'specifically',
    'additionally', 'similarly', 'likewise',
    'hence', 'thus', 'still', 'otherwise',
    'rather', 'accordingly', 'finally',
  ]);
  
  // Words that shouldn't end a sentence
  const NON_TERMINAL_WORDS = new Set([
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'in', 'on', 'at', 'by', 'for', 'with', 'to', 'of',
    'mr', 'ms', 'mrs', 'dr', 'prof',
  ]);
  
  class PostProcessor {
    constructor(debugTokenLimit = 3) {
      /**
       * @param debugTokenLimit {Number}
       *  How many tokens to print detailed debug information for.
       */
      this.debugTokenLimit = debugTokenLimit;
    }
  
    /**
     * Convert a TypedArray (including BigInt64Array) to a standard number array.
     * @param {TypedArray|Array} data
     * @returns {number[]}
     */
    toNumberArray(data) {
      return Array.from(data, (val) => Number(val));
    }
  
    /**
     * Safely retrieve an integer class index from a tensor at a given position.
     * - For 2D data ([batch, seqLen]) we return the numeric value at `position`.
     * - For 3D data ([batch, seqLen, numClasses]) we interpret the slice as one-hot
     *   and return the index of `1`. (Or default to `0` if not found.)
     * @param {Object} tensor - ONNX-like tensor object {data, dims, type}.
     * @param {number} position - Index within the sequence.
     * @returns {number} - Predicted class index or raw value.
     */
    getTensorValue(tensor, position) {
      // For 2D tensors [batch_size, sequence_length]
      if (tensor.dims.length === 2) {
        if (position >= tensor.data.length) return 0;
        return Number(tensor.data[position]);
      }
  
      // For 3D tensors [batch_size, sequence_length, num_classes]
      if (tensor.dims.length === 3) {
        const seqLen = tensor.dims[1];
        const numClasses = tensor.dims[2];
        const start = position * numClasses;
  
        // Safety check to avoid going out of bounds
        if (start + numClasses > tensor.data.length) {
          return 0;
        }
        const values = this.toNumberArray(
          tensor.data.slice(start, start + numClasses)
        );
        const classIdx = values.indexOf(1);
        return classIdx >= 0 ? classIdx : 0;
      }
  
      return 0; // Default if shape is unrecognized.
    }
  
    /**
     * Decide which punctuation mark to place after a token, if any.
     * @param {number} punctPred - Model punctuation class index.
     * @param {number} segPred   - Model segmentation class index (sentence boundary).
     * @param {string} token
     * @param {string} nextToken
     * @param {string} prevToken
     * @returns {string} - The chosen punctuation character (or empty).
     */
    getPunctuation(punctPred, segPred, token, nextToken, prevToken) {
      const rawMark = PUNCTUATION_MAP[punctPred] || '';
  
      // Determine if the model or segmentation suggests sentence boundary
      // We treat period, question, (optionally exclamation) as end-of-sentence if not in NON_TERMINAL_WORDS.
      const couldBeEndMark = rawMark === '.' || rawMark === '?' /*|| rawMark === '!'*/;
  
      // If segPred = 1 and our context check says it might be a boundary, treat it as a potential sentence boundary.
      const segBoundary = segPred === 1 && this.isPotentialSentenceBoundary(token, nextToken, prevToken);
  
      // If it's truly an end-of-sentence, return that mark unless the token is known not to end sentences.
      if ((couldBeEndMark || segBoundary) && !this.isNonTerminal(token)) {
        // If the rawMark is question mark (or exclamation if enabled), keep it
        if (rawMark === '?' /*|| rawMark === '!'*/) {
          return rawMark;
        }
        // If rawMark was a period or we only found a boundary from segPred, unify to period.
        return '.';
      }
  
      // For commas, we refine logic:
      if (rawMark === ',') {
        // If the next word is a known conjunction/transition, allow the comma
        if (nextToken && COMMA_TRIGGERS.has(nextToken.toLowerCase())) {
          return rawMark;
        }
        // E.g., "in London," or "at Harvard," if recognized as a location/proper noun
        if (
          token &&
          /^(in|at|on|from|to)$/.test(prevToken?.toLowerCase()) &&
          token[0] === token[0].toUpperCase()
        ) {
          return rawMark;
        }
        // Otherwise default to no punctuation
        return '';
      }
  
      return rawMark;
    }
  
    /**
     * Decide if a token is a "non-terminal" word that generally shouldn't end a sentence.
     * @param {string} word
     * @returns {boolean}
     */
    isNonTerminal(word) {
      if (!word) return false;
      const lower = word.toLowerCase();
      return (
        NON_TERMINAL_WORDS.has(lower) ||
        // Pure digits are often part of a larger numeric expression and shouldn't end sentences abruptly.
        /^\d+$/.test(lower)
      );
    }
  
    /**
     * Context-based check to see if we likely have a sentence boundary.
     * @param {string} token
     * @param {string} nextToken
     * @param {string} prevToken
     * @returns {boolean}
     */
    isPotentialSentenceBoundary(token, nextToken, prevToken) {
      if (!token || !nextToken) return false;
  
      // Common boundary heuristics:
      // 1) Next token is a personal pronoun often starting a sentence
      // 2) Next token is capitalized (likely a proper noun) and not a standard conjunction/article
      // 3) The current token is a verb that typically ends quotes or sentences
      // 4) The current token is purely numeric, but next token is not a numeric suffix or measure
      //    (e.g., "3 pm", "3 weeks", "3rd", etc.)
      return (
        /^(he|she|it|they|we|i)$/i.test(nextToken) ||
        (
          nextToken[0] === nextToken[0].toUpperCase() &&
          !NON_TERMINAL_WORDS.has(nextToken.toLowerCase()) &&
          !/^(and|but|or|nor|for|yet|so)$/i.test(nextToken)
        ) ||
        /^(said|replied|asked|thought|wondered|exclaimed|continued)$/i.test(token) ||
        (
          /^\d+$/.test(token) &&
          !/^(am|pm|th|st|nd|rd|\d|dollars|cents|years|days|months|weeks)$/i.test(nextToken)
        )
      );
    }
  
    /**
     * Apply "true casing" decision: whether to capitalize or lowercase.
     * @param {string} word
     * @param {boolean} capitalize - If true, force capitalization of the token.
     * @returns {string}
     */
    applyTrueCasing(word, capitalize) {
      if (!word) return word;
  
      // Always uppercase "I" in isolation
      if (word.toLowerCase() === 'i') {
        return 'I';
      }
  
      // If told to capitalize or if it's recognized as a proper noun by pattern
      if (capitalize || this.isProperNoun(word)) {
        return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
      }
  
      return word.toLowerCase();
    }
  
    /**
     * Check if a word is likely a proper noun (by pattern).
     * You can expand this logic to handle more use cases if needed.
     * @param {string} word
     * @returns {boolean}
     */
    isProperNoun(word) {
      const lcWord = word.toLowerCase();
      // Titles, days, months, etc.
      return (
        /^(mr|ms|mrs|dr|prof)\.?$/i.test(lcWord) ||
        /^(monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)$/i.test(lcWord)
      );
    }
  
    /**
     * Main processing method.
     * Takes raw tokens and model outputs, returns a punctuated, cased string.
     * @param {string[]} tokens - Tokenized text from the model (including [CLS], [SEP]).
     * @param {Object} modelOutputs - Dictionary of named tensors (e.g. {post_preds, cap_preds, seg_preds}).
     * @returns {string} - The final text with punctuation and casing.
     */
    process(tokens, modelOutputs) {
      if (!tokens || tokens.length <= 2) {
        // Not enough tokens or no special tokens to remove
        return tokens.join(' ');
      }
  
      // Remove special tokens (often [CLS] at start, [SEP] at end).
      tokens = tokens.slice(1, -1);
  
      // Debug info for model outputs
    //   console.log('Model outputs:');
    //   for (const [key, value] of Object.entries(modelOutputs)) {
    //     const shape = value.dims;
    //     const firstFew = this.toNumberArray(value.data).slice(0, 10); // Show first 10 data points
    //     console.log(`${key}:`, { shape, sampleData: firstFew, type: value.type });
    //   }
  
      let result = [];
      let currentSentence = [];
      let forceCapNext = true; // Capitalize the very first token of the text
  
      for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i] || '';
        const nextToken = i < tokens.length - 1 ? tokens[i + 1] : null;
        const prevToken = i > 0 ? tokens[i - 1] : null;
  
        // Skip empty or whitespace-only tokens
        if (!token.trim()) continue;
  
        // Retrieve model predictions
        const punctPred = this.getTensorValue(modelOutputs.post_preds, i);
        const capPred = this.getTensorValue(modelOutputs.cap_preds, i);
        const segPred = this.getTensorValue(modelOutputs.seg_preds, i);
  
        // Decide if we should capitalize
        const shouldCap = forceCapNext || capPred === 1;
        const processedToken = this.applyTrueCasing(token, shouldCap);
  
        // Decide punctuation
        const punctuation = this.getPunctuation(punctPred, segPred, token, nextToken, prevToken);
  
        // Optional Debug Logging for first N tokens
        // if (i < this.debugTokenLimit) {
        //   console.log('\n--- Debug Info ---');
        //   console.log(`Token index: ${i}`);
        //   console.log(`Raw token: "${token}"`);
        //   console.log(`Next token: "${nextToken}"`);
        //   console.log(`Prev token: "${prevToken}"`);
        //   console.log(`punctPred: ${punctPred} -> punctuation: "${punctuation}"`);
        //   console.log(`capPred:   ${capPred} -> final token: "${processedToken}"`);
        //   console.log(`segPred:   ${segPred}`);
        // }
  
        // Add a space before the token if not at the start of the sentence
        if (currentSentence.length > 0) {
          currentSentence.push(' ');
        }
        currentSentence.push(processedToken);
  
        // Attach punctuation if predicted
        if (punctuation) {
          currentSentence.push(punctuation);
        }
  
        // If we ended the sentence, push the current sentence to result and reset
        if (punctuation === '.' || punctuation === '?' /*|| punctuation === '!'*/) {
          result.push(currentSentence.join(''));
          currentSentence = [];
          forceCapNext = true; // Start next token capitalized
        } else {
          forceCapNext = false;
        }
      }
  
      // Flush remaining tokens as a final sentence, if any
      if (currentSentence.length > 0) {
        // If the last chunk doesn't end with . or ?, we give it a final period.
        const lastChar = currentSentence[currentSentence.length - 1];
        if (!/[.?]/.test(lastChar)) {
          currentSentence.push('.');
        }
        result.push(currentSentence.join(''));
      }
  
      // Join all sentences with a space in between
      return result.join(' ');
    }
  }
  
  export default PostProcessor;
  