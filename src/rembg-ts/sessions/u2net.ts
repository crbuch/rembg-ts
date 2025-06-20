import * as ort from 'onnxruntime-web';
import * as np from '../libraries/numpy';
import { Image } from '../libraries/PIL';
import type { PILImage } from '../libraries/PIL';

import { BaseSession } from './base';

export class U2netSession extends BaseSession {
    /**
     * This class represents a U2net session, which is a subclass of BaseSession.
     */    async predict(img: PILImage, ..._args: unknown[]): Promise<PILImage[]> {
        /**
         * Predicts the output masks for the input image using the inner session.
         *
         * Parameters:
         *     img (PILImage): The input image.
         *     ..._args: Additional arguments.
         *
         * Returns:
         *     PILImage[]: The list of output masks.
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
                mi = Math.min(mi, slicedData[i]);
            }
              // Normalize: (pred - mi) / (ma - mi)
            const normalizedData = new Float32Array(channelSize);
            for (let i = 0; i < slicedData.length; i++) {
                normalizedData[i] = (slicedData[i] - mi) / (ma - mi);
            }
            
            // Clip to [0, 1] range and scale to [0, 255]: pred.clip(0, 1) * 255
            const clippedData = new Uint8Array(channelSize);
            for (let i = 0; i < normalizedData.length; i++) {
                const clipped = Math.max(0, Math.min(1, normalizedData[i]));
                clippedData[i] = Math.round(clipped * 255);
            }
            
            // Create numpy array for the mask
            const maskArray = np.array(clippedData);
            maskArray.shape = [height, width];
            
            // Convert to PIL Image (mode "L" for grayscale)
            const mask = Image.fromarray(maskArray);
            const resized_mask = mask.resize(img.size, Image.Resampling.LANCZOS);

            return [resized_mask];
        } catch (error) {
            console.error("Prediction failed:", error);
            throw error;
        }
    }    static async download_models(...args: unknown[]): Promise<string> {
        /**
         * Downloads the U2net model file from a specific URL and saves it.
         *
         * Parameters:
         *     ...args: Additional arguments.
         *
         * Returns:
         *     string: The path to the downloaded model file.
         */
        const fname = `${this.getModelName(...args)}.onnx`;
        const model_path = `/models/${fname}`;
        
        try {
            console.log(`Checking U2net model availability: ${model_path}`);
            const response = await fetch(model_path, { method: 'HEAD' });
            if (!response.ok) {
                throw new Error(`U2net model not found: ${response.statusText}`);
            }
            console.log(`U2net model is available at: ${model_path}`);
            return model_path;
        } catch (error) {
            console.error(`Failed to verify U2net model:`, error);
            throw error;
        }
    }

    static getModelName(..._args: unknown[]): string {
        /**
         * Returns the name of the U2net session.
         *
         * Parameters:
         *     ..._args: Additional arguments.
         *
         * Returns:
         *     string: The name of the session.
         */
        return "u2net";
    }
}
