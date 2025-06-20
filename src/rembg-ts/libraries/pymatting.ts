/**
 * PyMatting-like functions using MODNet ONNX model
 * This provides the same API as pymatting but uses a fast ONNX model instead
 */

import * as ort from 'onnxruntime-web';
import { NumpyArray, zeros } from './numpy';

// Global model session cache
let modnetSession: ort.InferenceSession | null = null;

/**
 * Get optimal execution providers for ONNX Runtime
 */
function getExecutionProviders(): string[] {
    const providers: string[] = [];
    
    // Try WebGPU first if available
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
        providers.push('webgpu');
    }
    
    // Always fall back to CPU
    providers.push('cpu');
    
    return providers;
}

/**
 * Load the MODNet ONNX model
 */
async function loadMODNetModel(): Promise<ort.InferenceSession> {
    if (modnetSession) {
        return modnetSession;
    }
    
    try {
        const modelPath = '/models/modnet_photographic_portrait_matting.onnx';
        const executionProviders = getExecutionProviders();
        
        console.log('Loading MODNet model with execution providers:', executionProviders);
        
        const sessionOptions: ort.InferenceSession.SessionOptions = {
            executionProviders: executionProviders,
            graphOptimizationLevel: 'all',
        };
        
        modnetSession = await ort.InferenceSession.create(modelPath, sessionOptions);
        return modnetSession;
    } catch (error) {
        console.error('Failed to load MODNet model:', error);
        throw error;
    }
}

/**
 * Get scale factors for resizing image while maintaining aspect ratio
 */
function getScaleFactors(imH: number, imW: number, refSize: number = 512): [number, number] {
    let imRh: number;
    let imRw: number;
    
    if (Math.max(imH, imW) < refSize || Math.min(imH, imW) > refSize) {
        if (imW >= imH) {
            imRh = refSize;
            imRw = Math.floor(imW / imH * refSize);
        } else {
            imRw = refSize;
            imRh = Math.floor(imH / imW * refSize);
        }
    } else {
        imRh = imH;
        imRw = imW;
    }
    
    // Make dimensions divisible by 32
    imRw = imRw - (imRw % 32);
    imRh = imRh - (imRh % 32);
    
    const xScaleFactor = imRw / imW;
    const yScaleFactor = imRh / imH;
    
    return [xScaleFactor, yScaleFactor];
}

/**
 * Preprocess image for MODNet inference
 */
function preprocessImage(image: NumpyArray): { tensor: ort.Tensor, originalShape: [number, number] } {    // Convert to RGB if needed and normalize shape
    const [imH, imW] = image.shape;
    
    // Normalize values to [-1, 1] (MODNet expects this range)
    const normalized = image.divide(127.5).subtract(1.0);
    
    // Get scale factors
    const [xScale, yScale] = getScaleFactors(imH, imW);
    
    // Resize image (simplified - using nearest neighbor for now)
    const newH = Math.floor(imH * yScale);
    const newW = Math.floor(imW * xScale);
    
    const resized = zeros([newH, newW, 3], 'float32');
    for (let y = 0; y < newH; y++) {
        for (let x = 0; x < newW; x++) {
            const srcY = Math.min(imH - 1, Math.floor(y / yScale));
            const srcX = Math.min(imW - 1, Math.floor(x / xScale));
            
            for (let c = 0; c < 3; c++) {
                resized.set([y, x, c], normalized.get([srcY, srcX, c]));
            }
        }
    }
    
    // Convert from HWC to CHW format for ONNX
    const tensorData = new Float32Array(3 * newH * newW);
    let idx = 0;
    
    for (let c = 0; c < 3; c++) {
        for (let y = 0; y < newH; y++) {
            for (let x = 0; x < newW; x++) {
                tensorData[idx++] = resized.get([y, x, c]);
            }
        }
    }
    
    const tensor = new ort.Tensor('float32', tensorData, [1, 3, newH, newW]);
    return { tensor, originalShape: [imH, imW] };
}

/**
 * Run MODNet inference to get alpha matte
 */
async function runMODNetInference(image: NumpyArray): Promise<NumpyArray> {
    const session = await loadMODNetModel();
    const { tensor, originalShape } = preprocessImage(image);
    
    // Run inference
    const feeds = { [session.inputNames[0]]: tensor };
    const results = await session.run(feeds);
    
    // Get output tensor
    const outputTensor = results[session.outputNames[0]];
    const outputData = outputTensor.data as Float32Array;
    const [, , outH, outW] = outputTensor.dims as number[];
    
    // Convert back to original size
    const [origH, origW] = originalShape;
    const alpha = zeros([origH, origW], 'float32');
    
    for (let y = 0; y < origH; y++) {
        for (let x = 0; x < origW; x++) {
            const srcY = Math.min(outH - 1, Math.floor(y * outH / origH));
            const srcX = Math.min(outW - 1, Math.floor(x * outW / origW));
            const srcIdx = srcY * outW + srcX;
            
            alpha.set([y, x], Math.max(0, Math.min(1, outputData[srcIdx])));
        }
    }
    
    return alpha;
}

/**
 * Estimate alpha from an input image using MODNet
 * This replaces the closed-form alpha matting with a neural network approach
 * 
 * @param image - Input image with shape h × w × 3 (values 0-255 or 0-1)
 * @param trimap - Trimap (ignored - MODNet doesn't need trimaps)
 * @returns Alpha matte with shape h × w
 */
export async function estimate_alpha_cf(
    image: NumpyArray,
    trimap?: NumpyArray  // Made optional since MODNet doesn't need trimaps
): Promise<NumpyArray> {
    console.log('estimate_alpha_cf called - using MODNet model');
    
    // Validate input
    if (image.shape.length !== 3 || image.shape[2] !== 3) {
        throw new Error('Input image must have shape [height, width, 3]');
    }
    
    // Ensure image is in 0-255 range for preprocessing
    const imageNormalized = image.max() <= 1.0 ? image.multiply(255) : image;
    return await runMODNetInference(imageNormalized);
}

/**
 * Estimate foreground using simple alpha compositing
 * For portraits, this is often sufficient since MODNet provides good alpha
 * 
 * @param image - Input image with shape h × w × 3
 * @param alpha - Alpha matte with shape h × w  
 * @returns Foreground image with shape h × w × 3
 */
export function estimate_foreground_ml(
    image: NumpyArray,
    alpha: NumpyArray
): NumpyArray {
    // Validate inputs
    if (image.shape.length !== 3 || image.shape[2] !== 3) {
        throw new Error('Input image must have shape [height, width, 3]');
    }
    if (alpha.shape.length !== 2) {
        throw new Error('Alpha matte must have shape [height, width]');
    }
    if (image.shape[0] !== alpha.shape[0] || image.shape[1] !== alpha.shape[1]) {
        throw new Error('Image and alpha must have same height and width');
    }
    
    const [h, w, c] = image.shape;
    const foreground = zeros([h, w, c], 'float32');
    
    // Simple foreground estimation: F = I / alpha (where alpha > threshold)
    const threshold = 0.01;
    
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const alphaValue = alpha.get([y, x]);
            
            for (let ch = 0; ch < c; ch++) {
                const imageValue = image.get([y, x, ch]);
                
                if (alphaValue > threshold) {
                    // Foreground = Image / alpha
                    const foregroundValue = imageValue / alphaValue;
                    foreground.set([y, x, ch], Math.min(1.0, foregroundValue));
                } else {
                    // For very small alpha, use original image value
                    foreground.set([y, x, ch], imageValue);
                }
            }
        }
    }
    
    return foreground;
}

/**
 * Stack images along the color channel dimension
 * Combines multiple images into a single multi-channel image
 * 
 * @param images - Images to stack
 * @returns Combined image
 */
export function stack_images(...images: NumpyArray[]): NumpyArray {
    if (images.length === 0) {
        throw new Error('At least one image must be provided');
    }
    
    // Handle the common case of foreground + alpha -> RGBA
    if (images.length === 2) {
        const [foreground, alpha] = images;
        const [h, w] = foreground.shape;
        
        if (foreground.shape.length === 3 && foreground.shape[2] === 3 && 
            alpha.shape.length === 2) {
            // Create RGBA image
            const result = zeros([h, w, 4], 'float32');
            
            // Copy RGB channels
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    for (let c = 0; c < 3; c++) {
                        result.set([y, x, c], foreground.get([y, x, c]));
                    }
                    result.set([y, x, 3], alpha.get([y, x]));
                }
            }
            
            return result;
        }
    }
    
    // General case: concatenate along last dimension
    const [h, w] = images[0].shape;
    let totalChannels = 0;
    
    // Calculate total channels needed
    for (const img of images) {
        if (img.shape.length === 2) {
            totalChannels += 1;
        } else if (img.shape.length === 3) {
            totalChannels += img.shape[2];
        }
    }
    
    const result = zeros([h, w, totalChannels], 'float32');
    let channelOffset = 0;
    
    for (const img of images) {
        if (img.shape.length === 2) {
            // Single channel image
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    result.set([y, x, channelOffset], img.get([y, x]));
                }
            }
            channelOffset += 1;
        } else if (img.shape.length === 3) {
            // Multi-channel image
            const channels = img.shape[2];
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    for (let c = 0; c < channels; c++) {
                        result.set([y, x, channelOffset + c], img.get([y, x, c]));
                    }
                }
            }
            channelOffset += channels;
        }
    }
    
    return result;
}
