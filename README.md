# Zero-Latency Vision Pipeline

A fast, client-side object detection pipeline powered by WebGPU. 

This project uses [Transformers.js](https://huggingface.co/docs/transformers.js) and the `Xenova/detr-resnet-50` model to run object detection completely locally, bypassing the need for cloud inference APIs.

## Features
- **Zero Latency:** Computations are offloaded to your local GPU via WebGPU, taking the heavy lifting off the CPU and returning results almost instantly.
- **Offline-First:** The model weights are downloaded once and cached locally. After the first run, it works entirely offline.
- **High Accuracy:** Uses the robust `detr-resnet-50` model with a strict 85% confidence threshold to filter out false positives.

## Prerequisites
- [Node.js](https://nodejs.org/) installed on your machine.

## Getting Started

1. Navigate to the project folder in your terminal:
   ```bash
   cd "Zero-Latency Vision Pipeline"
   ```

2. Install the required dependencies:
   ```bash
   npm install
   ```

3. Run the script:
   ```bash
   node index.js
   ```

## Example Output

By default, the script processes a sample image and outputs the detected objects, their confidence scores, and bounding box coordinates in JSON format:

```json
[System] Allocating WebGPU memory and caching model weights...
[System] Executing zero-latency inference on client hardware...
[
  {
    "score": 0.9990647435188293,
    "label": "cat",
    "box": {
      "xmin": 4,
      "ymin": 53,
      "xmax": 323,
      "ymax": 468
    }
  },
  {
    "score": 0.9958711266517639,
    "label": "couch",
    "box": {
      "xmin": -3,
      "ymin": 0,
      "xmax": 636,
      "ymax": 472
    }
  }
]
```

## Testing Your Own Images
Want to try it on a different image? Just open `index.js` and replace the URL in the `imageUrl` variable with any image link you'd like to test.
