/**
 * PIL-like image processing utilities using Web APIs
 * Implements a subset of PIL functionality using HTMLImageElement, HTMLCanvasElement, and Blob URLs
 */

export interface PILImage {
    width: number;
    height: number;
    mode: string;
    size: [number, number];
    
    convert(mode: string): PILImage;
    resize(size: [number, number], resampleMode?: number): PILImage;
    putalpha(alpha: PILImage): void;
    paste(img: PILImage, position: [number, number]): void;    save(output: unknown, format: string): void;
    saveAsync?(output: unknown, format: string): Promise<void>;
    getCanvas?(): HTMLCanvasElement;
    toBlob?(format?: string): Promise<Blob>;
    toUint8Array?(format?: string): Promise<Uint8Array>;
}

export class WebPILImage implements PILImage {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    public mode: string;    constructor(canvas: HTMLCanvasElement, mode = "RGBA") {
        this.canvas = canvas;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Could not get 2D context');
        this.ctx = ctx;
        this.mode = mode;
    }

    get width(): number {
        return this.canvas.width;
    }

    get height(): number {
        return this.canvas.height;
    }

    get size(): [number, number] {
        return [this.width, this.height];
    }    convert(mode: string): PILImage {
        // For web implementation, we'll just change the mode flag
        // In a real implementation, you'd convert the pixel data
        const newCanvas = document.createElement('canvas');
        newCanvas.width = this.width;
        newCanvas.height = this.height;
        const newCtx = newCanvas.getContext('2d');
        if (!newCtx) throw new Error('Could not get 2D context');
        newCtx.drawImage(this.canvas, 0, 0);
        
        return new WebPILImage(newCanvas, mode);
    }

    putalpha(alpha: PILImage): void {
        if (alpha instanceof WebPILImage) {
            const imageData = this.ctx.getImageData(0, 0, this.width, this.height);
            const alphaData = alpha.ctx.getImageData(0, 0, alpha.width, alpha.height);
            
            // Apply alpha channel from the mask
            for (let i = 0; i < imageData.data.length; i += 4) {
                const alphaIndex = (i / 4) * 4; // Get corresponding pixel in alpha image
                imageData.data[i + 3] = alphaData.data[alphaIndex]; // Use red channel as alpha
            }
            
            this.ctx.putImageData(imageData, 0, 0);
        }
    }

    paste(img: PILImage, position: [number, number]): void {
        if (img instanceof WebPILImage) {
            this.ctx.drawImage(img.canvas, position[0], position[1]);
        }
    }    save(output: unknown, format: string): void {
        // For web implementation, this would typically convert to blob
        this.canvas.toBlob((blob) => {
            if (output && typeof output === 'object' && 'write' in output && typeof (output as { write: unknown }).write === 'function') {
                (output as { write: (blob: Blob | null) => void }).write(blob);
            }
        }, `image/${format.toLowerCase()}`);
    }

    async saveAsync(output: unknown, format: string): Promise<void> {
        return new Promise((resolve, reject) => {
            this.canvas.toBlob(async (blob) => {
                try {
                    if (blob && output && typeof output === 'object' && 'writeBlob' in output && typeof (output as { writeBlob: unknown }).writeBlob === 'function') {
                        await (output as { writeBlob: (blob: Blob) => Promise<number> }).writeBlob(blob);
                    }
                    resolve();
                } catch (error) {
                    reject(error);
                }
            }, `image/${format.toLowerCase()}`);
        });
    }

    getCanvas(): HTMLCanvasElement {
        return this.canvas;
    }

    getImageData(): ImageData {
        return this.ctx.getImageData(0, 0, this.width, this.height);
    }

    async toBlob(format = 'image/png'): Promise<Blob> {
        return new Promise((resolve, reject) => {
            this.canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error('Failed to convert canvas to blob'));
                }
            }, format);
        });
    }

    async toUint8Array(format = 'image/png'): Promise<Uint8Array> {
        const blob = await this.toBlob(format);
        const arrayBuffer = await blob.arrayBuffer();
        return new Uint8Array(arrayBuffer);
    }

    resize(size: [number, number], resampleMode?: number): PILImage {
        const [newWidth, newHeight] = size;
        const newCanvas = document.createElement('canvas');
        newCanvas.width = newWidth;
        newCanvas.height = newHeight;
        const newCtx = newCanvas.getContext('2d');
        if (!newCtx) throw new Error('Could not get 2D context');
        
        // Set resampling mode if provided (though browser has limited control)
        if (resampleMode === Image.Resampling.LANCZOS) {
            newCtx.imageSmoothingQuality = 'high';
        }
        newCtx.imageSmoothingEnabled = true;
        
        // Draw the current canvas onto the new canvas with new dimensions
        newCtx.drawImage(this.canvas, 0, 0, this.width, this.height, 0, 0, newWidth, newHeight);
        
        return new WebPILImage(newCanvas, this.mode);
    }
}

export class Image {
    // Resampling constants to match PIL
    static Resampling = {
        NEAREST: 0,
        LANCZOS: 1,
        BILINEAR: 2,
        BICUBIC: 3
    };

    static new(mode: string, size: [number, number] | number, color?: number | [number, number, number, number]): PILImage {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Could not get 2D context');
        
        if (Array.isArray(size)) {
            canvas.width = size[0];
            canvas.height = size[1];
        } else {
            // Handle case where size is a single number (should be array)
            canvas.width = canvas.height = size;
        }

        if (color !== undefined) {
            if (typeof color === 'number') {
                ctx.fillStyle = `rgba(${color}, ${color}, ${color}, 1)`;
            } else if (Array.isArray(color)) {
                const [r, g, b, a = 255] = color;
                ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${a / 255})`;
            }
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        return new WebPILImage(canvas, mode);
    }static open(source: HTMLImageElement | HTMLCanvasElement | ImageData | File | Blob | string): PILImage {
        // Handle HTMLImageElement
        if (source instanceof HTMLImageElement) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Could not get 2D context');
            
            canvas.width = source.naturalWidth || source.width;
            canvas.height = source.naturalHeight || source.height;
            ctx.drawImage(source, 0, 0);
            return new WebPILImage(canvas);
        }
        
        // Handle HTMLCanvasElement
        if (source instanceof HTMLCanvasElement) {
            return new WebPILImage(source);
        }
        
        // Handle ImageData
        if (source instanceof ImageData) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Could not get 2D context');
            
            canvas.width = source.width;
            canvas.height = source.height;
            ctx.putImageData(source, 0, 0);
            return new WebPILImage(canvas);
        }
        
        // Handle File, Blob or string (URL) - these need async loading
        if (source instanceof File || source instanceof Blob || typeof source === 'string') {
            throw new Error('Use Image.openAsync() for File, Blob, or URL sources');
        }
        
        // Fallback
        throw new Error(`Unsupported source type for Image.open: ${typeof source}`);
    }    static async openAsync(source: File | Blob | string): Promise<PILImage> {
        return new Promise((resolve, reject) => {
            const img = document.createElement('img');
            
            img.onload = () => {
                try {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    if (!ctx) throw new Error('Could not get 2D context');
                    
                    canvas.width = img.naturalWidth || img.width;
                    canvas.height = img.naturalHeight || img.height;
                    ctx.drawImage(img, 0, 0);
                    
                    // Clean up object URL if we created one
                    if (source instanceof File || source instanceof Blob) {
                        URL.revokeObjectURL(img.src);
                    }
                    
                    resolve(new WebPILImage(canvas));
                } catch (error) {
                    reject(error);
                }
            };
            
            img.onerror = () => {
                reject(new Error('Failed to load image'));
            };
            
            // Set the source
            if (source instanceof File || source instanceof Blob) {
                img.src = URL.createObjectURL(source);
            } else {
                img.src = source;
            }
        });
    }static fromarray(input: ImageData | HTMLCanvasElement | { data: Uint8Array | Float32Array | Int32Array; shape: number[] } | unknown[]): PILImage {
        // Handle ImageData directly
        if (input instanceof ImageData) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Could not get 2D context');
            
            canvas.width = input.width;
            canvas.height = input.height;
            ctx.putImageData(input, 0, 0);
            return new WebPILImage(canvas);
        }
        
        // Handle HTMLCanvasElement
        if (input instanceof HTMLCanvasElement) {
            return new WebPILImage(input);
        }
        
        // Handle tensor-like data with shape
        if (input && typeof input === 'object' && 'data' in input && 'shape' in input) {
            const tensorData = input as { data: Uint8Array | Float32Array | Int32Array; shape: number[] };
            const [height, width, channels = 1] = tensorData.shape;
            
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Could not get 2D context');
            
            canvas.width = width;
            canvas.height = height;
            
            const imageData = ctx.createImageData(width, height);
            const pixels = imageData.data;
            
            // Convert tensor data to RGBA
            for (let i = 0; i < height * width; i++) {
                if (channels === 1) {
                    // Grayscale
                    const value = Math.round(Math.max(0, Math.min(255, tensorData.data[i])));
                    pixels[i * 4] = value;     // R
                    pixels[i * 4 + 1] = value; // G
                    pixels[i * 4 + 2] = value; // B
                    pixels[i * 4 + 3] = 255;   // A
                } else if (channels >= 3) {
                    // RGB or RGBA
                    pixels[i * 4] = Math.round(Math.max(0, Math.min(255, tensorData.data[i * channels])));         // R
                    pixels[i * 4 + 1] = Math.round(Math.max(0, Math.min(255, tensorData.data[i * channels + 1]))); // G
                    pixels[i * 4 + 2] = Math.round(Math.max(0, Math.min(255, tensorData.data[i * channels + 2]))); // B
                    pixels[i * 4 + 3] = channels >= 4 ? 
                        Math.round(Math.max(0, Math.min(255, tensorData.data[i * channels + 3]))) : 255; // A
                }
            }
            
            ctx.putImageData(imageData, 0, 0);
            return new WebPILImage(canvas);
        }
        
        // Handle unknown arrays (fallback)
        if (Array.isArray(input)) {
            // Create a small placeholder canvas
            const canvas = document.createElement('canvas');
            canvas.width = 100;
            canvas.height = 100;
            return new WebPILImage(canvas);
        }
        
        throw new Error(`Unsupported input type for fromarray: ${typeof input}`);
    }    static composite(img1: PILImage, img2: PILImage, mask: PILImage): PILImage {
        if (img1 instanceof WebPILImage && img2 instanceof WebPILImage && mask instanceof WebPILImage) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Could not get 2D context');
            
            canvas.width = img1.width;
            canvas.height = img1.height;

            // Get image data
            const img1Data = img1.getImageData();
            const img2Data = img2.getImageData();
            const maskData = mask.getImageData();
            const resultData = ctx.createImageData(canvas.width, canvas.height);

            // Composite based on mask
            for (let i = 0; i < resultData.data.length; i += 4) {
                const maskAlpha = maskData.data[i] / 255; // Use red channel as mask
                
                resultData.data[i] = img1Data.data[i] * maskAlpha + img2Data.data[i] * (1 - maskAlpha);     // R
                resultData.data[i + 1] = img1Data.data[i + 1] * maskAlpha + img2Data.data[i + 1] * (1 - maskAlpha); // G
                resultData.data[i + 2] = img1Data.data[i + 2] * maskAlpha + img2Data.data[i + 2] * (1 - maskAlpha); // B
                resultData.data[i + 3] = Math.max(img1Data.data[i + 3], img2Data.data[i + 3]); // A
            }

            ctx.putImageData(resultData, 0, 0);
            return new WebPILImage(canvas);
        }
        
        return img1; // fallback
    }

    static alpha_composite(background: PILImage, foreground: PILImage): PILImage {
        if (background instanceof WebPILImage && foreground instanceof WebPILImage) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Could not get 2D context');
            
            canvas.width = background.width;
            canvas.height = background.height;

            // Draw background first
            ctx.drawImage(background.getCanvas(), 0, 0);
            
            // Draw foreground with alpha blending
            ctx.globalCompositeOperation = 'source-over';
            ctx.drawImage(foreground.getCanvas(), 0, 0);

            return new WebPILImage(canvas);
        }
        
        return background; // fallback
    }    static fromCanvas(canvas: HTMLCanvasElement): PILImage {
        return new WebPILImage(canvas);
    }
}

export class ImageOps {
    static exif_transpose(img: PILImage): PILImage {
        // For web implementation, EXIF rotation would need to be handled
        // This is a simplified version that just returns the image as-is
        return img;
    }
}

// Utility functions for web-native image handling
export function createImageFromCanvas(canvas: HTMLCanvasElement): HTMLImageElement {
    /**
     * Create an HTMLImageElement from a canvas element.
     * Useful for displaying processed images in the DOM.
     */
    const img = new HTMLImageElement();
    img.src = canvas.toDataURL();
    return img;
}

export async function createImageFromPIL(pilImage: PILImage): Promise<HTMLImageElement> {
    /**
     * Create an HTMLImageElement from a PIL image for display.
     */
    if (pilImage instanceof WebPILImage) {
        return createImageFromCanvas(pilImage.getCanvas());
    }
    
    // Fallback for other PIL implementations
    const img = new HTMLImageElement();
    if (pilImage.getCanvas) {
        img.src = pilImage.getCanvas().toDataURL();
    }
    return img;
}

export function isWebNativeImageType(input: unknown): input is HTMLImageElement | HTMLCanvasElement | ImageData | File | Blob {
    /**
     * Type guard to check if input is a web-native image type.
     */
    return input instanceof HTMLImageElement || 
           input instanceof HTMLCanvasElement || 
           input instanceof ImageData ||
           input instanceof File ||
           input instanceof Blob;
}

export function isPILImage(input: unknown): input is PILImage {
    /**
     * Type guard to check if input is a PIL image.
     */
    return input !== null && 
           typeof input === 'object' && 
           'width' in input && 
           'height' in input && 
           'mode' in input &&
           typeof (input as PILImage).width === 'number' &&
           typeof (input as PILImage).height === 'number' &&
           typeof (input as PILImage).mode === 'string';
}
