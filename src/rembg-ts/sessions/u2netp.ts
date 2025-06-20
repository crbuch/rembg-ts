import * as ort from 'onnxruntime-web';
import * as np from '../libraries/numpy';
import { Image } from '../libraries/PIL';
import type { PILImage } from '../libraries/PIL';

import { BaseSession } from './base';

export class U2netpSession extends BaseSession {
    /**
     * This class represents a session for using the U2netp model.
     */    async predict(img: PILImage, ..._args: unknown[]): Promise<PILImage[]> {
        /**
         * Predicts the mask for the given image using the U2netp model.
         *
         * Parameters:
         *     img (PILImage): The input image.
         *     ..._args: Additional arguments.
         *
         * Returns:
         *     PILImage[]: The predicted mask.
         */
        await this.ensureInitialized();
        
        if (!this.inner_session) {
            throw new Error("Session not initialized");
        }

        const inputs = this.normalize(
            img, 
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225], 
            [320, 320]
        );

        try {
            // Convert NumpyArray to ONNX tensor format
            const inputTensors: Record<string, ort.Tensor> = {};
            for (const [name, npArray] of Object.entries(inputs)) {
                inputTensors[name] = new ort.Tensor('float32', npArray.data as Float32Array, npArray.shape);
            }

            // Run inference
            const ort_outs = await this.inner_session.run(inputTensors);
            
            // Get the output tensor (first output)
            const outputName = Object.keys(ort_outs)[0];
            const pred = ort_outs[outputName];
            
            // Convert tensor data to numpy-like array: pred[:, 0, :, :]
            const predData = pred.data as Float32Array;
            const [, , height, width] = pred.dims;
            
            // Extract first channel: pred[:, 0, :, :]
            const channelSize = height * width;
            const slicedData = new Float32Array(channelSize);
            for (let i = 0; i < channelSize; i++) {
                slicedData[i] = predData[i]; // First channel data
            }
            
            // Find min and max for normalization
            let ma = -Infinity;
            let mi = Infinity;
            for (let i = 0; i < slicedData.length; i++) {
                ma = Math.max(ma, slicedData[i]);
                mi = Math.min(mi, slicedData[i]);            }
              // Normalize: (pred - mi) / (ma - mi)  
            const normalizedData = new Float32Array(channelSize);
            for (let i = 0; i < slicedData.length; i++) {
                normalizedData[i] = (slicedData[i] - mi) / (ma - mi);
            }
            
            // Note: u2netp doesn't use clip() like u2net, it directly scales: pred * 255
            const scaledData = new Uint8Array(channelSize);
            for (let i = 0; i < normalizedData.length; i++) {
                scaledData[i] = Math.round(normalizedData[i] * 255);
            }
            
            // Create numpy array for the mask
            const maskArray = np.array(scaledData);
            maskArray.shape = [height, width];
            
            // Convert to PIL Image (mode "L" for grayscale)
            const mask = Image.fromarray(maskArray);
            const resized_mask = mask.resize(img.size, Image.Resampling.LANCZOS);

            return [resized_mask];
        } catch (error) {
            console.error("U2netp prediction failed:", error);
            throw error;
        }
    }    static async download_models(..._args: unknown[]): Promise<string> {
        /**
         * Downloads the U2netp model.
         *
         * Parameters:
         *     ..._args: Additional arguments.
         *
         * Returns:
         *     string: The path to the downloaded model.
         */
        const fname = `${this.getModelName(..._args)}.onnx`;
        const model_path = `/models/${fname}`;
        
        try {
            console.log(`Checking U2netp model availability: ${model_path}`);
            const response = await fetch(model_path, { method: 'HEAD' });
            if (!response.ok) {
                throw new Error(`U2netp model not found: ${response.statusText}`);
            }
            console.log(`U2netp model is available at: ${model_path}`);
            return model_path;
        } catch (error) {
            console.error(`Failed to verify U2netp model:`, error);
            throw error;
        }
    }    static getModelName(..._args: unknown[]): string {
        /**
         * Returns the name of the U2netp model.
         *
         * Parameters:
         *     ..._args: Additional arguments.
         *
         * Returns:
         *     string: The name of the model.
         */        return "u2netp";
    }
}
