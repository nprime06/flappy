import * as ort from 'onnxruntime-web';

export interface ModelConfig {
  latentMean: number;
  latentStd: number;
  contextFrames: number;
  latentChannels: number;
  numAugBins: number;
  numClasses: number;
  imageSize: number;
  latentSize: number;
  useDoneHead: boolean;
}

export type ProgressCallback = (stage: string, progress: number) => void;

export class ModelManager {
  encoder: ort.InferenceSession | null = null;
  decoder: ort.InferenceSession | null = null;
  resunet: ort.InferenceSession | null = null;
  config: ModelConfig | null = null;

  private useWebGPU = false;

  async init(modelPath: string, onProgress?: ProgressCallback): Promise<void> {
    // Check WebGPU availability
    this.useWebGPU = await this.checkWebGPU();
    const provider = this.useWebGPU ? 'webgpu' : 'wasm';

    onProgress?.('provider', 0);
    console.log(`Using execution provider: ${provider}`);

    // Configure session options
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: [provider],
      graphOptimizationLevel: 'all',
    };

    // Load config
    onProgress?.('config', 5);
    const configResponse = await fetch(`${modelPath}/config.json`);
    const configJson = await configResponse.json();
    this.config = {
      latentMean: configJson.latent_mean,
      latentStd: configJson.latent_std,
      contextFrames: configJson.context_frames,
      latentChannels: configJson.latent_channels,
      numAugBins: configJson.num_aug_bins,
      numClasses: configJson.num_classes,
      imageSize: configJson.image_size,
      latentSize: configJson.latent_size,
      useDoneHead: configJson.use_done_head ?? false,
    };

    // Load encoder (smallest, load first for quick feedback)
    onProgress?.('encoder', 10);
    const encoderPath = `${modelPath}/encoder.onnx`;
    this.encoder = await ort.InferenceSession.create(encoderPath, sessionOptions);
    console.log('Encoder loaded');

    // Load decoder
    onProgress?.('decoder', 30);
    const decoderPath = `${modelPath}/decoder.onnx`;
    this.decoder = await ort.InferenceSession.create(decoderPath, sessionOptions);
    console.log('Decoder loaded');

    // Load ResUNet (largest)
    onProgress?.('resunet', 50);
    const resunetPath = `${modelPath}/resunet.onnx`;
    this.resunet = await ort.InferenceSession.create(resunetPath, sessionOptions);
    console.log('ResUNet loaded');

    // Warmup inference to trigger shader compilation
    onProgress?.('warmup', 90);
    await this.warmup();

    onProgress?.('done', 100);
    console.log('All models loaded and warmed up');
  }

  private async checkWebGPU(): Promise<boolean> {
    if (!navigator.gpu) {
      console.log('WebGPU not supported in this browser');
      return false;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.log('No WebGPU adapter found');
        return false;
      }
      const device = await adapter.requestDevice();
      if (!device) {
        console.log('Could not get WebGPU device');
        return false;
      }
      console.log('WebGPU available');
      return true;
    } catch (e) {
      console.log('WebGPU check failed:', e);
      return false;
    }
  }

  private async warmup(): Promise<void> {
    if (!this.encoder || !this.decoder || !this.resunet || !this.config) {
      throw new Error('Models not loaded');
    }

    const { latentChannels, latentSize, contextFrames, imageSize } = this.config;

    // Warmup encoder
    const dummyImage = new ort.Tensor('float32', new Float32Array(1 * 3 * imageSize * imageSize), [1, 3, imageSize, imageSize]);
    await this.encoder.run({ image: dummyImage });

    // Warmup decoder
    const dummyLatent = new ort.Tensor('float32', new Float32Array(1 * latentChannels * latentSize * latentSize), [1, latentChannels, latentSize, latentSize]);
    await this.decoder.run({ latent: dummyLatent });

    // Warmup ResUNet
    const dummyX = new ort.Tensor('float32', new Float32Array(1 * latentChannels * latentSize * latentSize), [1, latentChannels, latentSize, latentSize]);
    const dummyT = new ort.Tensor('float32', new Float32Array([0.5]), [1]);
    const dummyAction = new ort.Tensor('int64', BigInt64Array.from([0n]), [1]);
    const dummyZCond = new ort.Tensor('float32', new Float32Array(1 * contextFrames * latentChannels * latentSize * latentSize), [1, contextFrames * latentChannels, latentSize, latentSize]);
    const dummyAug = new ort.Tensor('int64', BigInt64Array.from([0n]), [1]);
    await this.resunet.run({
      x: dummyX,
      t: dummyT,
      action: dummyAction,
      z_cond: dummyZCond,
      aug_level: dummyAug,
    });

    console.log('Warmup complete');
  }

  isWebGPU(): boolean {
    return this.useWebGPU;
  }
}
