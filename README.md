# 🧑‍🏭 Punctuation Restore

A Node.js package that restores punctuation and casing to unpunctuated text using the `punctuation_fullstop_truecase_english` ONNX model:
https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english

![punctuation-restore](./punctuation-restore.jpg)

## Features

- Restores punctuation marks (periods, commas, question marks, etc.)
- Handles casing
- Supports batch processing of multiple texts
- Uses efficient ONNX runtime for inference
- Automatically downloads required models

### _Notes_
- Models are automatically downloaded from Hugging Face on first use and saved locally to the `./models` directory for future use.
- Punctuation correction isn't perfect, but it's good enough for most use cases (use with caution).


## Installation

```bash
npm install punctuation-restore
```

## Quick Start

```javascript
import PunctuationRestorer from 'punctuation-restore';

const restorer = new PunctuationRestorer();

const texts = [
  "this is a string without any punctuation or casing yesterday i went to disneyworld and had a great time",
  "washing your dog once a month is important nothing quite beats a walk on the beach"
];

const results = await restorer.restore(texts);
console.log(results);
```

## API Reference

### `PunctuationRestorer`

The main class for handling punctuation restoration.

#### Methods

- `async restore(texts: string[]): Promise<string[]>`
  - Takes an array of unpunctuated texts
  - Returns an array of punctuated and cased sentences
  - Automatically handles model initialization and cleanup

- `async cleanup()`
  - Manually release ONNX session resources
  - Called automatically after `restore()`, but can be called explicitly if needed

### Model Architecture

The package uses two main models:
- `model.onnx`: Main ONNX model for punctuation and casing prediction
- `tokenizer.model`: Tokenizer model for text preprocessing

Models are automatically downloaded from Hugging Face on **first use** and saved locally to the `./models` directory for future use.

## Example

Check out `example/example.js` for a complete working example:

```javascript
import PunctuationRestorer from '../punctuationRestore.js';

const testTexts = [
  "this is a string without any punctuation or casing yesterday i went to disneyworld and had a great time",
  "washing your dog once a month is important nothing quite beats a walk on the beach"
];

try {
    const restorer = new PunctuationRestorer();
    const results = await restorer.restore(testTexts);
    results.forEach(result => console.log(result));
} catch (error) {
    console.error('Test failed:', error);
}
```

## Development

### Scripts

- `npm run clean`: Clean install dependencies
- `npm run example`: Run the example script

### Project Structure

```
punctuation-restore/
├── modules/
│   ├── downloadModel.js    # Model download handling
│   ├── tokenizer.js        # Text tokenization
│   └── postProcessor.js    # Output processing
├── example/
│   └── example.js         # Usage example
└── punctuationRestore.js  # Main package entry
```

## Dependencies

- `onnxruntime-node`: ^1.16.3 - ONNX runtime for Node.js
- `sentence-parse`: ^1.3.0 - Sentence parsing utilities

## License

MIT
