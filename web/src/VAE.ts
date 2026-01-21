import * as ort from 'onnxruntime-web';
import { ModelManager, ModelConfig } from './ModelManager';

export class VAE {
  private manager: ModelManager;

  constructor(manager: ModelManager) {
    this.manager = manager;
  }

  get config(): ModelConfig {
    if (!this.manager.config) {
      throw new Error('Model config not loaded');
    }
    return this.manager.config;
  }

  /**
   * Encode an image to normalized latent space.
   * Input: ImageData or Float32Array in [-1, 1] range, shape (1, 3, H, W)
   * Output: Float32Array of normalized latent, shape (1, 4, 18, 18)
   */
  async encode(imageData: ImageData | Float32Array): Promise<Float32Array> {
    if (!this.manager.encoder) {
      throw new Error('Encoder not loaded');
    }

    const { imageSize, latentMean, latentStd } = this.config;

    // Convert ImageData to tensor if needed
    let inputData: Float32Array;
    if (imageData instanceof ImageData) {
      inputData = this.imageDataToTensor(imageData);
    } else {
      inputData = imageData;
    }

    // Create input tensor
    const inputTensor = new ort.Tensor('float32', inputData, [1, 3, imageSize, imageSize]);

    // Run encoder
    const results = await this.manager.encoder.run({ image: inputTensor });
    const mean = results.mean.data as Float32Array;

    // Normalize: z = (mean - latent_mean) / latent_std
    const normalized = new Float32Array(mean.length);
    for (let i = 0; i < mean.length; i++) {
      normalized[i] = (mean[i] - latentMean) / latentStd;
    }

    return normalized;
  }

  /**
   * Decode a normalized latent to image.
   * Input: Float32Array of normalized latent, shape (1, 4, 18, 18)
   * Output: ImageData (144x144)
   */
  async decode(latent: Float32Array): Promise<ImageData> {
    if (!this.manager.decoder) {
      throw new Error('Decoder not loaded');
    }

    const { latentChannels, latentSize, latentMean, latentStd, imageSize } = this.config;

    // Denormalize: z_denorm = z * latent_std + latent_mean
    const denormalized = new Float32Array(latent.length);
    for (let i = 0; i < latent.length; i++) {
      denormalized[i] = latent[i] * latentStd + latentMean;
    }

    // Create input tensor
    const inputTensor = new ort.Tensor('float32', denormalized, [1, latentChannels, latentSize, latentSize]);

    // Run decoder
    const results = await this.manager.decoder.run({ latent: inputTensor });
    const output = results.image.data as Float32Array;

    // Convert tensor to ImageData
    return this.tensorToImageData(output, imageSize, imageSize);
  }

  /**
   * Convert ImageData (RGBA) to tensor (CHW, RGB, range [-1, 1]).
   */
  private imageDataToTensor(imageData: ImageData): Float32Array {
    const { width, height, data } = imageData;
    const tensor = new Float32Array(3 * width * height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcIdx = (y * width + x) * 4;
        const dstIdxR = 0 * width * height + y * width + x;
        const dstIdxG = 1 * width * height + y * width + x;
        const dstIdxB = 2 * width * height + y * width + x;

        // Convert [0, 255] to [-1, 1]
        tensor[dstIdxR] = (data[srcIdx] / 255) * 2 - 1;
        tensor[dstIdxG] = (data[srcIdx + 1] / 255) * 2 - 1;
        tensor[dstIdxB] = (data[srcIdx + 2] / 255) * 2 - 1;
      }
    }

    return tensor;
  }

  /**
   * Convert tensor (CHW, RGB, range [-1, 1]) to ImageData (RGBA).
   */
  private tensorToImageData(tensor: Float32Array, width: number, height: number): ImageData {
    const data = new Uint8ClampedArray(width * height * 4);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcIdxR = 0 * width * height + y * width + x;
        const srcIdxG = 1 * width * height + y * width + x;
        const srcIdxB = 2 * width * height + y * width + x;
        const dstIdx = (y * width + x) * 4;

        // Convert [-1, 1] to [0, 255], clamping
        data[dstIdx] = Math.min(255, Math.max(0, Math.round(((tensor[srcIdxR] + 1) / 2) * 255)));
        data[dstIdx + 1] = Math.min(255, Math.max(0, Math.round(((tensor[srcIdxG] + 1) / 2) * 255)));
        data[dstIdx + 2] = Math.min(255, Math.max(0, Math.round(((tensor[srcIdxB] + 1) / 2) * 255)));
        data[dstIdx + 3] = 255; // Alpha
      }
    }

    return new ImageData(data, width, height);
  }
}
