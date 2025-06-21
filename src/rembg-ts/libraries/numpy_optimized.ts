/**
 * Memory-optimized NumPy-like operations using ONNX Tensors
 * This module provides a more memory-efficient alternative to the standard numpy module
 * by working directly with ort.Tensor objects when possible, avoiding unnecessary data copies.
 */

import * as ort from 'onnxruntime-web';
import type { NumpyArray } from './numpy';
import * as np from './numpy';

/**
 * Extended numpy array interface that can work with ONNX tensors for memory efficiency
 */
export interface TensorNumpyArray extends NumpyArray {
    // Additional properties for tensor integration
    _tensor?: ort.Tensor;
    _isFromTensor: boolean;
    
    // Enhanced methods for tensor operations
    toTensor(): ort.Tensor;
}

class TensorWebNumpyArray implements TensorNumpyArray {
    public shape: number[];
    public dtype: string;
    public data: Float32Array | Uint8Array | Int32Array;
    public _tensor?: ort.Tensor;
    public _isFromTensor: boolean;

    constructor(
        data: Float32Array | Uint8Array | Int32Array, 
        shape: number[], 
        dtype: string,
        tensor?: ort.Tensor
    ) {
        this.data = data;
        this.shape = shape;
        this.dtype = dtype;
        this._tensor = tensor;
        this._isFromTensor = !!tensor;
    }

    static fromTensor(tensor: ort.Tensor): TensorNumpyArray {
        /**
         * Create a TensorNumpyArray from an ONNX tensor without copying data when possible
         */
        const data = tensor.data as Float32Array | Uint8Array | Int32Array;
        let dtype: string;
        
        switch (tensor.type) {
            case 'float32':
                dtype = 'float32';
                break;
            case 'uint8':
                dtype = 'uint8';
                break;
            case 'int32':
                dtype = 'int32';
                break;
            default:
                dtype = 'float32';
        }
        
        // Create without copying the data - we keep a reference to the original tensor
        return new TensorWebNumpyArray(data, [...tensor.dims], dtype, tensor);
    }

    toTensor(): ort.Tensor {
        /**
         * Convert this array to an ONNX tensor
         */
        if (this._tensor && this._isFromTensor) {
            // If this was created from a tensor and hasn't been modified, return the original
            return this._tensor;
        }
        
        // Create a new tensor from current data
        let tensorType: 'float32' | 'uint8' | 'int32';
        switch (this.dtype) {
            case 'float32':
                tensorType = 'float32';
                break;
            case 'uint8':
                tensorType = 'uint8';
                break;
            case 'int32':
                tensorType = 'int32';
                break;
            default:
                tensorType = 'float32';
        }
        
        return new ort.Tensor(tensorType, this.data, this.shape);
    }    // Implement all required NumpyArray methods by delegating to standard numpy array
    private _getStandardArray(): NumpyArray {
        // Create a new WebNumpyArray using the internal constructor
        // We need to replicate the WebNumpyArray functionality here
        const WebNumpyArray = np.array(this.data).constructor as any;
        return new WebNumpyArray(this.data, this.shape, this.dtype);
    }

    astype(dtype: string): NumpyArray {
        return this._getStandardArray().astype(dtype);
    }

    divide(value: number | NumpyArray): NumpyArray {
        // Mark that we're no longer using the original tensor data
        this._isFromTensor = false;
        return this._getStandardArray().divide(value);
    }

    multiply(value: number): NumpyArray {
        this._isFromTensor = false;
        return this._getStandardArray().multiply(value);
    }

    subtract(value: number | NumpyArray): NumpyArray {
        this._isFromTensor = false;
        return this._getStandardArray().subtract(value);
    }

    gt(value: number): NumpyArray {
        return this._getStandardArray().gt(value);
    }

    lt(value: number): NumpyArray {
        return this._getStandardArray().lt(value);
    }

    ge(value: number): NumpyArray {
        return this._getStandardArray().ge(value);
    }

    le(value: number): NumpyArray {
        return this._getStandardArray().le(value);
    }

    min(): number {
        return this._getStandardArray().min();
    }

    max(): number {
        return this._getStandardArray().max();
    }

    sum(): number {
        return this._getStandardArray().sum();
    }

    flatten(): NumpyArray {
        return this._getStandardArray().flatten();
    }

    reshape(shape: number[]): NumpyArray {
        // For reshape, we can potentially keep using tensor data
        if (this._isFromTensor && this._tensor) {
            return new TensorWebNumpyArray(this.data, shape, this.dtype, this._tensor);
        }
        return this._getStandardArray().reshape(shape);
    }

    clip(min: number, max: number): NumpyArray {
        this._isFromTensor = false;
        return this._getStandardArray().clip(min, max);
    }

    transpose(axes: number[]): NumpyArray {
        this._isFromTensor = false;
        return this._getStandardArray().transpose(axes);
    }

    and(other: NumpyArray): NumpyArray {
        return this._getStandardArray().and(other);
    }

    or(other: NumpyArray): NumpyArray {
        return this._getStandardArray().or(other);
    }

    not(): NumpyArray {
        return this._getStandardArray().not();
    }

    setValues(indices: NumpyArray, value: number): void {
        this._isFromTensor = false;
        this._getStandardArray().setValues(indices, value);
    }

    getChannel(channel: number): NumpyArray {
        return this._getStandardArray().getChannel(channel);
    }

    setChannel(channel: number, values: NumpyArray): void {
        this._isFromTensor = false;
        this._getStandardArray().setChannel(channel, values);
    }

    get(indices: number[]): number {
        return this._getStandardArray().get(indices);
    }

    set(indices: number[], value: number): void {
        this._isFromTensor = false;
        this._getStandardArray().set(indices, value);
    }
}

/**
 * Create a numpy array from ONNX tensor output for memory efficiency
 */
export function fromTensorOutput(tensorOutput: ort.Tensor): TensorNumpyArray {
    return TensorWebNumpyArray.fromTensor(tensorOutput);
}

/**
 * Memory-efficient mask processing that works directly with tensor data
 */
export function processU2NetOutput(output: ort.Tensor, targetHeight: number, targetWidth: number): NumpyArray {
    /**
     * Process U2Net model output tensor directly without unnecessary copies
     * 
     * Parameters:
     *     output: The ONNX tensor output from U2Net model
     *     targetHeight: Target height for the output mask
     *     targetWidth: Target width for the output mask
     * 
     * Returns:
     *     NumpyArray: Processed mask ready for PIL Image conversion
     */
      // Work directly with tensor data
    const predData = output.data as Float32Array;
    const [, , height, width] = output.dims;
    
    // Calculate channel size
    const channelSize = height * width;
    
    // Extract first channel data (avoid copying entire array)
    const slicedData = new Float32Array(channelSize);
    for (let i = 0; i < channelSize; i++) {
        slicedData[i] = predData[i]; // First channel data
    }

    // Find min and max for normalization (could be optimized further with SIMD)
    let ma = -Infinity;
    let mi = Infinity;
    for (let i = 0; i < slicedData.length; i++) {
        const val = slicedData[i];
        if (val > ma) ma = val;
        if (val < mi) mi = val;
    }

    // Normalize and convert to uint8 in one pass
    const clippedData = new Uint8Array(channelSize);
    const range = ma - mi;
    if (range > 0) {
        for (let i = 0; i < slicedData.length; i++) {
            const normalized = (slicedData[i] - mi) / range;
            const clipped = Math.max(0, Math.min(1, normalized));
            clippedData[i] = Math.round(clipped * 255);
        }
    } else {
        // If all values are the same, fill with 255 or 0
        clippedData.fill(ma > 0.5 ? 255 : 0);
    }    // Create optimized numpy array
    const result = np.array(clippedData);
    result.shape = [height, width];
    return result;
}

/**
 * Convert a tensor to a standard numpy array when tensor optimization isn't beneficial
 */
export function tensorToNumpyArray(tensor: ort.Tensor): NumpyArray {
    const data = tensor.data as Float32Array | Uint8Array | Int32Array;
    
    const result = np.array(data);
    result.shape = [...tensor.dims];
    return result;
}

/**
 * Create a tensor from numpy array data
 */
export function numpyArrayToTensor(array: NumpyArray): ort.Tensor {
    let tensorType: 'float32' | 'uint8' | 'int32';
    
    switch (array.dtype) {
        case 'float32':
            tensorType = 'float32';
            break;
        case 'uint8':
            tensorType = 'uint8';
            break;
        case 'int32':
            tensorType = 'int32';
            break;
        default:
            tensorType = 'float32';
    }
    
    return new ort.Tensor(tensorType, array.data, array.shape);
}

// Export the tensor-optimized functions
export {
    TensorWebNumpyArray
};
