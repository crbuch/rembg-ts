/**
 * NumPy-like operations using ONNX tensors and Web APIs
 * Implements a subset of NumPy functionality needed for background removal
 */

export interface NumpyArray {
    shape: number[];
    dtype: string;
    data: Float32Array | Uint8Array | Int32Array;
    astype(dtype: string): NumpyArray;
    
    // Arithmetic operations
    divide(value: number | NumpyArray): NumpyArray;
    multiply(value: number): NumpyArray;
    subtract(value: number | NumpyArray): NumpyArray;
    gt(value: number): NumpyArray; // greater than
    lt(value: number): NumpyArray; // less than
    ge(value: number): NumpyArray; // greater than or equal
    le(value: number): NumpyArray; // less than or equal
    
    // Array statistics
    min(): number;
    max(): number;
    sum(): number;
    
    // Array manipulation
    flatten(): NumpyArray;
    reshape(shape: number[]): NumpyArray;
    clip(min: number, max: number): NumpyArray;
    transpose(axes: number[]): NumpyArray;
    
    // Boolean operations
    and(other: NumpyArray): NumpyArray;
    or(other: NumpyArray): NumpyArray;
    not(): NumpyArray;
    
    // Array indexing and assignment operations
    setValues(indices: NumpyArray, value: number): void;
    getChannel(channel: number): NumpyArray;
    setChannel(channel: number, values: NumpyArray): void;
    get(indices: number[]): number;
    set(indices: number[], value: number): void;
}

export interface DataType {
    uint8: string;
    float32: string;
    int32: string;
}

export const uint8: DataType['uint8'] = 'uint8';
export const float32: DataType['float32'] = 'float32';
export const int32: DataType['int32'] = 'int32';

class WebNumpyArray implements NumpyArray {
    public shape: number[];
    public dtype: string;
    public data: Float32Array | Uint8Array | Int32Array;

    constructor(data: Float32Array | Uint8Array | Int32Array, shape: number[], dtype: string) {
        this.data = data;
        this.shape = shape;
        this.dtype = dtype;
    }    astype(dtype: string): NumpyArray {
        let newData: Float32Array | Uint8Array | Int32Array;
        
        switch (dtype) {
            case 'uint8':
                newData = new Uint8Array(this.data.length);
                for (let i = 0; i < this.data.length; i++) {
                    newData[i] = Math.max(0, Math.min(255, Math.round(this.data[i])));
                }
                break;
            case 'float32':
                newData = new Float32Array(this.data);
                break;
            case 'int32':
                newData = new Int32Array(this.data.length);
                for (let i = 0; i < this.data.length; i++) {
                    newData[i] = Math.round(this.data[i]);
                }
                break;
            default:
                newData = this.data;
        }
        
        return new WebNumpyArray(newData, [...this.shape], dtype);
    }    divide(value: number | NumpyArray): NumpyArray {
        if (typeof value === 'number') {
            const newData = new Float32Array(this.data.length);
            for (let i = 0; i < this.data.length; i++) {
                newData[i] = this.data[i] / value;
            }
            return new WebNumpyArray(newData, [...this.shape], 'float32');
        } else {
            // Element-wise division
            const newData = new Float32Array(this.data.length);
            for (let i = 0; i < this.data.length; i++) {
                newData[i] = this.data[i] / value.data[i];
            }
            return new WebNumpyArray(newData, [...this.shape], 'float32');
        }
    }

    subtract(value: number | NumpyArray): NumpyArray {
        if (typeof value === 'number') {
            const newData = new Float32Array(this.data.length);
            for (let i = 0; i < this.data.length; i++) {
                newData[i] = this.data[i] - value;
            }
            return new WebNumpyArray(newData, [...this.shape], 'float32');
        } else {
            // Element-wise subtraction
            const newData = new Float32Array(this.data.length);
            for (let i = 0; i < this.data.length; i++) {
                newData[i] = this.data[i] - value.data[i];
            }
            return new WebNumpyArray(newData, [...this.shape], 'float32');
        }
    }

    multiply(value: number): NumpyArray {
        const newData = new Float32Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) {
            newData[i] = this.data[i] * value;
        }
        return new WebNumpyArray(newData, [...this.shape], 'float32');
    }

    gt(value: number): NumpyArray {
        const newData = new Uint8Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) {
            newData[i] = this.data[i] > value ? 1 : 0;
        }
        return new WebNumpyArray(newData, [...this.shape], 'uint8');
    }    lt(value: number): NumpyArray {
        const newData = new Uint8Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) {
            newData[i] = this.data[i] < value ? 1 : 0;
        }
        return new WebNumpyArray(newData, [...this.shape], 'uint8');
    }

    ge(value: number): NumpyArray {
        const newData = new Uint8Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) {
            newData[i] = this.data[i] >= value ? 1 : 0;
        }
        return new WebNumpyArray(newData, [...this.shape], 'uint8');
    }

    le(value: number): NumpyArray {
        const newData = new Uint8Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) {
            newData[i] = this.data[i] <= value ? 1 : 0;
        }
        return new WebNumpyArray(newData, [...this.shape], 'uint8');
    }

    min(): number {
        let minVal = Infinity;
        for (let i = 0; i < this.data.length; i++) {
            if (this.data[i] < minVal) {
                minVal = this.data[i];
            }
        }
        return minVal;
    }

    max(): number {
        let maxVal = -Infinity;
        for (let i = 0; i < this.data.length; i++) {
            if (this.data[i] > maxVal) {
                maxVal = this.data[i];
            }
        }
        return maxVal;
    }

    sum(): number {
        let total = 0;
        for (let i = 0; i < this.data.length; i++) {
            total += this.data[i];
        }
        return total;
    }

    flatten(): NumpyArray {
        const newShape = [this.data.length];
        const newData = new Float32Array(this.data);
        return new WebNumpyArray(newData, newShape, this.dtype);
    }

    reshape(shape: number[]): NumpyArray {
        const totalSize = shape.reduce((acc, dim) => acc * dim, 1);
        if (totalSize !== this.data.length) {
            throw new Error(`Cannot reshape array of size ${this.data.length} into shape ${shape}`);
        }
        return new WebNumpyArray(this.data, shape, this.dtype);
    }

    and(other: NumpyArray): NumpyArray {
        const newData = new Uint8Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) {
            newData[i] = (this.data[i] > 0 && other.data[i] > 0) ? 1 : 0;
        }
        return new WebNumpyArray(newData, [...this.shape], 'uint8');
    }

    or(other: NumpyArray): NumpyArray {
        const newData = new Uint8Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) {
            newData[i] = (this.data[i] > 0 || other.data[i] > 0) ? 1 : 0;
        }
        return new WebNumpyArray(newData, [...this.shape], 'uint8');
    }

    not(): NumpyArray {
        const newData = new Uint8Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) {
            newData[i] = this.data[i] > 0 ? 0 : 1;
        }
        return new WebNumpyArray(newData, [...this.shape], 'uint8');
    }

    get(indices: number[]): number {
        if (indices.length !== this.shape.length) {
            throw new Error('Number of indices must match array dimensions');
        }
        
        let flatIndex = 0;
        let multiplier = 1;
        
        for (let i = this.shape.length - 1; i >= 0; i--) {
            flatIndex += indices[i] * multiplier;
            multiplier *= this.shape[i];
        }
        
        return this.data[flatIndex];
    }

    set(indices: number[], value: number): void {
        if (indices.length !== this.shape.length) {
            throw new Error('Number of indices must match array dimensions');
        }
        
        let flatIndex = 0;
        let multiplier = 1;
        
        for (let i = this.shape.length - 1; i >= 0; i--) {
            flatIndex += indices[i] * multiplier;
            multiplier *= this.shape[i];
        }
        
        this.data[flatIndex] = value;
    }setValues(indices: NumpyArray, value: number): void {
        // Set values at boolean indices to a specific value
        for (let i = 0; i < this.data.length; i++) {
            if (indices.data[i] > 0) { // Boolean true
                this.data[i] = value;
            }
        }
    }

    getChannel(channel: number): NumpyArray {
        // Extract a single channel from a multi-channel array
        // Assumes shape is [height, width, channels]
        if (this.shape.length !== 3) {
            throw new Error('getChannel requires a 3D array');
        }
        
        const [height, width, channels] = this.shape;
        const channelData = new Float32Array(height * width);
        
        for (let i = 0; i < height * width; i++) {
            channelData[i] = this.data[i * channels + channel];
        }
        
        return new WebNumpyArray(channelData, [height, width], this.dtype);
    }

    setChannel(channel: number, values: NumpyArray): void {
        // Set a single channel in a multi-channel array
        // Assumes shape is [height, width, channels]
        if (this.shape.length !== 3) {
            throw new Error('setChannel requires a 3D array');
        }
        
        const [height, width, channels] = this.shape;
        
        for (let i = 0; i < height * width; i++) {
            this.data[i * channels + channel] = values.data[i];
        }
    }

    transpose(axes: number[]): NumpyArray {
        // Transpose array according to given axes
        if (axes.length !== this.shape.length) {
            throw new Error('axes length must match array dimensions');
        }
        
        // For the common case of (2, 0, 1) transpose on 3D array (HWC to CHW)
        if (this.shape.length === 3 && axes[0] === 2 && axes[1] === 0 && axes[2] === 1) {
            const [height, width, channels] = this.shape;
            const newShape = [channels, height, width];
            const newData = new Float32Array(this.data.length);
            
            for (let c = 0; c < channels; c++) {
                for (let h = 0; h < height; h++) {
                    for (let w = 0; w < width; w++) {
                        const oldIndex = h * width * channels + w * channels + c;
                        const newIndex = c * height * width + h * width + w;
                        newData[newIndex] = this.data[oldIndex];
                    }
                }
            }
            
            return new WebNumpyArray(newData, newShape, this.dtype);
        }
          // For other transpose operations, implement as needed
        throw new Error('transpose: only (2,0,1) axes currently supported');
    }

    clip(min: number, max: number): NumpyArray {
        const newData = new Float32Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) {
            newData[i] = Math.max(min, Math.min(max, this.data[i]));
        }
        return new WebNumpyArray(newData, [...this.shape], this.dtype);
    }
}

export function asarray(input: unknown): NumpyArray {
    /**
     * Convert input to NumpyArray.
     * Supports PILImage and existing NumpyArrays.
     */
    
    // If it's already a NumpyArray, return as is
    if (input && typeof input === 'object' && 'shape' in input && 'data' in input && 'dtype' in input) {
        return input as NumpyArray;
    }
    
    // If it's a PIL Image, convert from canvas
    if (input && typeof input === 'object' && 'getCanvas' in input && typeof (input as { getCanvas: unknown }).getCanvas === 'function') {
        const canvas = (input as { getCanvas(): HTMLCanvasElement }).getCanvas();
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Could not get canvas context');
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = new Uint8Array(imageData.data);
        
        return new WebNumpyArray(data, [canvas.height, canvas.width, 4], 'uint8');
    }
    
    throw new Error(`Cannot convert ${typeof input} to NumpyArray`);
}

export function ones(shape: number[], options: { dtype: string }): NumpyArray {
    const totalSize = shape.reduce((acc, dim) => acc * dim, 1);
    
    let data: Float32Array | Uint8Array | Int32Array;
    switch (options.dtype) {
        case 'uint8':
            data = new Uint8Array(totalSize).fill(1);
            break;
        case 'float32':
            data = new Float32Array(totalSize).fill(1.0);
            break;
        case 'int32':
            data = new Int32Array(totalSize).fill(1);
            break;
        default:
            data = new Float32Array(totalSize).fill(1.0);
    }
    
    return new WebNumpyArray(data, shape, options.dtype);
}

export function full(shape: number[], options: { dtype: string; fill_value: number }): NumpyArray {
    const totalSize = shape.reduce((acc, dim) => acc * dim, 1);
    
    let data: Float32Array | Uint8Array | Int32Array;
    switch (options.dtype) {
        case 'uint8':
            data = new Uint8Array(totalSize).fill(options.fill_value);
            break;
        case 'float32':
            data = new Float32Array(totalSize).fill(options.fill_value);
            break;
        case 'int32':
            data = new Int32Array(totalSize).fill(options.fill_value);
            break;
        default:
            data = new Float32Array(totalSize).fill(options.fill_value);
    }
    
    return new WebNumpyArray(data, shape, options.dtype);
}

export function where(condition: NumpyArray, valueIfTrue: number, valueIfFalse: number): NumpyArray {
    const result = new Uint8Array(condition.data.length);
    
    for (let i = 0; i < condition.data.length; i++) {
        result[i] = condition.data[i] > 0 ? valueIfTrue : valueIfFalse;
    }
    
    return new WebNumpyArray(result, condition.shape, 'uint8');
}

export function clip(array: NumpyArray, min: number, max: number): WebNumpyArray {
    const result = new Float32Array(array.data.length);
    
    for (let i = 0; i < array.data.length; i++) {
        result[i] = Math.max(min, Math.min(max, array.data[i]));
    }
    
    return new WebNumpyArray(result, array.shape, array.dtype);
}

export function array(input: number[] | number[][] | Uint8Array | Float32Array | unknown): NumpyArray {
    // Handle typed arrays directly
    if (input instanceof Uint8Array) {
        return new WebNumpyArray(input, [input.length], 'uint8');
    }
    
    if (input instanceof Float32Array) {
        return new WebNumpyArray(input, [input.length], 'float32');
    }
    
    // Handle regular arrays
    if (Array.isArray(input)) {
        // Handle 1D arrays
        if (typeof input[0] === 'number') {
            const data = new Float32Array(input as number[]);
            return new WebNumpyArray(data, [input.length], 'float32');
        }
        
        // Handle 2D arrays
        if (Array.isArray(input[0])) {
            const input2d = input as number[][];
            const rows = input2d.length;
            const cols = input2d[0].length;
            const data = new Float32Array(rows * cols);
            
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    data[i * cols + j] = input2d[i][j];
                }
            }
            
            return new WebNumpyArray(data, [rows, cols], 'float32');
        }
    }
    
    // Fallback - convert anything else to array via asarray
    return asarray(input);
}

export function zeros(shape: number[], dtype: string = 'float32'): NumpyArray {
    const totalSize = shape.reduce((acc, dim) => acc * dim, 1);
    
    let data: Float32Array | Uint8Array | Int32Array;
    switch (dtype) {
        case 'uint8':
            data = new Uint8Array(totalSize).fill(0);
            break;
        case 'float32':
            data = new Float32Array(totalSize).fill(0);
            break;
        case 'int32':
            data = new Int32Array(totalSize).fill(0);
            break;
        default:
            data = new Float32Array(totalSize).fill(0);
    }
    
    return new WebNumpyArray(data, shape, dtype);
}

export function max(array: NumpyArray): number {
    let maxVal = -Infinity;
    for (let i = 0; i < array.data.length; i++) {
        if (array.data[i] > maxVal) {
            maxVal = array.data[i];
        }
    }
    return maxVal;
}

export function min(array: NumpyArray): number {
    let minVal = Infinity;
    for (let i = 0; i < array.data.length; i++) {
        if (array.data[i] < minVal) {
            minVal = array.data[i];
        }
    }
    return minVal;
}

export function expand_dims(array: NumpyArray, axis: number): NumpyArray {
    const newShape = [...array.shape];
    newShape.splice(axis, 0, 1);
    return new WebNumpyArray(array.data, newShape, array.dtype);
}

export function squeeze(array: NumpyArray, axis?: number): NumpyArray {
    let newShape: number[];
    
    if (axis !== undefined) {
        newShape = [...array.shape];
        if (newShape[axis] === 1) {
            newShape.splice(axis, 1);
        }
    } else {
        newShape = array.shape.filter(dim => dim !== 1);
    }
    
    return new WebNumpyArray(array.data, newShape, array.dtype);
}

// Export the main numpy-like object
const numpy = {
    asarray,
    ones,
    full,
    where,
    clip,
    array,
    zeros,
    max,
    min,
    expand_dims,
    squeeze,
    uint8,
    float32,
    int32
};

export default numpy;
