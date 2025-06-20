# rembg-ts

A TypeScript/JavaScript port of the popular Python rembg library for background removal in web environments.

## Overview

This library provides the same functionality as the Python rembg library but designed to run in web browsers using ONNX Runtime Web. It supports both u2net and u2netp models for background removal.

## Features

- **Web-native**: Runs entirely in the browser using ONNX Runtime Web
- **Multiple models**: Support for u2net and u2netp models
- **Flexible input/output**: Supports Uint8Array, PIL-like images, and numpy-like arrays
- **Alpha matting**: Advanced edge refinement for high-quality cutouts
- **Async operations**: Non-blocking operations that won't freeze the UI
- **TypeScript**: Full type safety and excellent IDE support

## Installation

```bash
npm install rembg-ts
```

Note: Make sure to place your ONNX model files in the `public/models/` directory of your web application:
- `public/models/u2net.onnx`
- `public/models/u2netp.onnx`

## Quick Start

```typescript
import { remove, Image } from 'rembg-ts';

// Basic usage with image bytes
async function removeBackground() {
    // Load image from file input or fetch
    const response = await fetch('/path/to/image.jpg');
    const imageBytes = new Uint8Array(await response.arrayBuffer());
    
    // Remove background
    const resultBytes = await remove(imageBytes) as Uint8Array;
    
    // Create blob and display
    const blob = new Blob([resultBytes], { type: 'image/png' });
    const url = URL.createObjectURL(blob);
    
    const img = document.createElement('img');
    img.src = url;
    document.body.appendChild(img);
}
```

## API Reference

### Main Functions

#### `remove(data, options?)`

The main function for background removal.

**Parameters:**
- `data: Uint8Array | PILImage | NumpyArray` - Input image data
- `alpha_matting?: boolean` - Enable alpha matting (default: false)
- `alpha_matting_foreground_threshold?: number` - Foreground threshold (default: 240)
- `alpha_matting_background_threshold?: number` - Background threshold (default: 10)
- `alpha_matting_erode_size?: number` - Erosion size (default: 10)
- `session?: BaseSession` - Custom session object
- `only_mask?: boolean` - Return only the mask (default: false)
- `post_process_mask?: boolean` - Apply post-processing to mask (default: false)
- `bgcolor?: [number, number, number, number]` - Background color (RGBA)
- `force_return_bytes?: boolean` - Force return as bytes (default: false)

**Returns:** `Promise<Uint8Array | PILImage | NumpyArray>`

#### `new_session(modelName)`

Create a new session for a specific model.

**Parameters:**
- `modelName: string` - Model name ("u2net" or "u2netp")

**Returns:** `BaseSession`

#### `new_session_async(modelName)`

Create and initialize a session asynchronously.

**Parameters:**
- `modelName: string` - Model name ("u2net" or "u2netp")

**Returns:** `Promise<BaseSession>`

### Image Processing

The library includes PIL-like image processing capabilities:

```typescript
import { Image } from 'rembg-ts';

// Open image from blob
const img = Image.open(blob);

// Resize image
const resized = img.resize([640, 480]);

// Convert image mode
const rgb = img.convert('RGB');

// Save image
const bytes = new io.BytesIO();
img.save(bytes, 'PNG');
```

### Numpy-like Operations

```typescript
import { np } from 'rembg-ts';

// Create arrays
const arr = np.zeros([100, 100, 3]);
const ones = np.ones([50, 50]);

// Array operations
const result = arr.multiply(2).add(ones);
```

## Examples

### Using Different Models

```typescript
import { remove, new_session_async } from 'rembg-ts';

// Use u2netp (lighter, faster)
const session = await new_session_async('u2netp');
const result = await remove(imageData, false, 240, 10, 10, session);

// Use u2net (higher quality)
const sessionHQ = await new_session_async('u2net');
const resultHQ = await remove(imageData, false, 240, 10, 10, sessionHQ);
```

### Alpha Matting for Better Edges

```typescript
const result = await remove(
    imageData,
    true,  // Enable alpha matting
    240,   // Foreground threshold
    10,    // Background threshold
    10     // Erosion size
);
```

### Adding Background Color

```typescript
const result = await remove(
    imageData,
    false, // alpha_matting
    240,   // alpha_matting_foreground_threshold
    10,    // alpha_matting_background_threshold
    10,    // alpha_matting_erode_size
    undefined, // session
    false, // only_mask
    false, // post_process_mask
    [255, 0, 0, 255] // Red background
);
```

### Batch Processing

```typescript
// Create session once for efficiency
const session = await new_session_async('u2net');

for (const imageData of imageList) {
    const result = await remove(imageData, false, 240, 10, 10, session);
    // Process result...
}
```

## Supported Models

- **u2net**: General purpose, high quality background removal
- **u2netp**: Lightweight version of u2net, faster processing

## Requirements

- Modern web browser with WebAssembly support
- ONNX Runtime Web
- Model files served from your web server

## Differences from Python rembg

1. **Async operations**: All model operations are asynchronous to prevent UI blocking
2. **Model loading**: Models are fetched from `/public/models/` directory
3. **Web APIs**: Uses Fetch API, Blob, and other web APIs instead of file system operations
4. **TypeScript**: Full type safety and modern JavaScript features

## Performance Tips

1. **Reuse sessions**: Create sessions once and reuse them for multiple images
2. **Choose appropriate models**: Use u2netp for speed, u2net for quality
3. **Preload models**: Call `new_session_async()` early to preload models
4. **Image sizing**: Resize large images before processing for better performance

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
