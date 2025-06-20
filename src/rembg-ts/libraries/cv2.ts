/**
 * OpenCV-like image processing operations for the web
 * Implements morphological operations and filtering using canvas and image processing techniques
 */

import { NumpyArray } from './numpy';
import * as np from './numpy';

// Constants for morphological operations
export const MORPH_ELLIPSE = 2;
export const MORPH_OPEN = 2;
export const BORDER_DEFAULT = 4;

export interface StructuringElement {
    shape: number[];
    data: Uint8Array;
}

export function getStructuringElement(shape: number, ksize: [number, number]): StructuringElement {
    const [width, height] = ksize;
    const data = new Uint8Array(width * height);
    
    if (shape === MORPH_ELLIPSE) {
        // Create elliptical structuring element
        const centerX = Math.floor(width / 2);
        const centerY = Math.floor(height / 2);
        const radiusX = centerX;
        const radiusY = centerY;
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const dx = x - centerX;
                const dy = y - centerY;
                const distance = (dx * dx) / (radiusX * radiusX) + (dy * dy) / (radiusY * radiusY);
                data[y * width + x] = distance <= 1 ? 1 : 0;
            }
        }
    } else {
        // Default to rectangular
        data.fill(1);
    }
    
    return {
        shape: [height, width],
        data
    };
}

export function morphologyEx(src: NumpyArray, op: number, kernel: StructuringElement): NumpyArray {
    // Simplified morphological operation
    // In a real implementation, you'd need proper erosion/dilation algorithms
    
    if (op === MORPH_OPEN) {
        // Opening = erosion followed by dilation
        const eroded = erode(src, kernel);
        return dilate(eroded, kernel);
    }
    
    return src; // fallback
}

function erode(src: NumpyArray, kernel: StructuringElement): NumpyArray {
    // Simplified erosion operation
    const [height, width] = src.shape.slice(0, 2);
    const [kh, kw] = kernel.shape;
    const result = new Uint8Array(src.data.length);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let minVal = 255;
            
            for (let ky = 0; ky < kh; ky++) {
                for (let kx = 0; kx < kw; kx++) {
                    const sy = y + ky - Math.floor(kh / 2);
                    const sx = x + kx - Math.floor(kw / 2);
                    
                    if (sy >= 0 && sy < height && sx >= 0 && sx < width && kernel.data[ky * kw + kx]) {
                        const srcIndex = sy * width + sx;
                        minVal = Math.min(minVal, src.data[srcIndex]);
                    }
                }
            }
              result[y * width + x] = minVal;
        }
    }
    
    const resultArray = np.array(result);
    resultArray.shape = src.shape;
    return resultArray.astype(src.dtype);
}

function dilate(src: NumpyArray, kernel: StructuringElement): NumpyArray {
    // Simplified dilation operation
    const [height, width] = src.shape.slice(0, 2);
    const [kh, kw] = kernel.shape;
    const result = new Uint8Array(src.data.length);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let maxVal = 0;
            
            for (let ky = 0; ky < kh; ky++) {
                for (let kx = 0; kx < kw; kx++) {
                    const sy = y + ky - Math.floor(kh / 2);
                    const sx = x + kx - Math.floor(kw / 2);
                    
                    if (sy >= 0 && sy < height && sx >= 0 && sx < width && kernel.data[ky * kw + kx]) {
                        const srcIndex = sy * width + sx;
                        maxVal = Math.max(maxVal, src.data[srcIndex]);
                    }
                }
            }
              result[y * width + x] = maxVal;
        }
    }
    
    const resultArray = np.array(result);
    resultArray.shape = src.shape;
    return resultArray.astype(src.dtype);
}

export function GaussianBlur(
    src: NumpyArray, 
    ksize: [number, number], 
    options: { sigmaX: number; sigmaY: number; borderType: number }
): NumpyArray {
    // Simplified Gaussian blur implementation
    const [height, width] = src.shape.slice(0, 2);
    const [kw, kh] = ksize;
    const { sigmaX, sigmaY } = options;
    
    // Create Gaussian kernel
    const kernel = createGaussianKernel(kw, kh, sigmaX, sigmaY);
    
    // Apply convolution
    const result = new Uint8Array(src.data.length);
    const halfKW = Math.floor(kw / 2);
    const halfKH = Math.floor(kh / 2);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let sum = 0;
            let weightSum = 0;
            
            for (let ky = 0; ky < kh; ky++) {
                for (let kx = 0; kx < kw; kx++) {
                    const sy = y + ky - halfKH;
                    const sx = x + kx - halfKW;
                    
                    if (sy >= 0 && sy < height && sx >= 0 && sx < width) {
                        const weight = kernel[ky * kw + kx];
                        const srcIndex = sy * width + sx;
                        sum += src.data[srcIndex] * weight;
                        weightSum += weight;
                    }
                }
            }
              result[y * width + x] = Math.round(sum / weightSum);
        }
    }
    
    const resultArray = np.array(result);
    resultArray.shape = src.shape;
    return resultArray.astype(src.dtype);
}

function createGaussianKernel(width: number, height: number, sigmaX: number, sigmaY: number): Float32Array {
    const kernel = new Float32Array(width * height);
    const centerX = Math.floor(width / 2);
    const centerY = Math.floor(height / 2);
    
    let sum = 0;
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const dx = x - centerX;
            const dy = y - centerY;
            const value = Math.exp(-(dx * dx) / (2 * sigmaX * sigmaX) - (dy * dy) / (2 * sigmaY * sigmaY));
            kernel[y * width + x] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for (let i = 0; i < kernel.length; i++) {
        kernel[i] /= sum;
    }
    
    return kernel;
}
