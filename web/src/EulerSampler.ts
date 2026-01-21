import * as ort from 'onnxruntime-web';
import { ModelManager, ModelConfig } from './ModelManager';

export class EulerSampler {
  private manager: ModelManager;
  numSteps: number;

  constructor(manager: ModelManager, numSteps = 20) {
    this.manager = manager;
    this.numSteps = numSteps;
  }

  get config(): ModelConfig {
    if (!this.manager.config) {
      throw new Error('Model config not loaded');
    }
    return this.manager.config;
  }

  /**
   * Sample next frame latent using Euler ODE integration.
   *
   * Integrates: dz/dt = v(z, t) from t=0 to t=1
   *
   * @param z0 Starting noise (1, 4, 18, 18)
   * @param zCond Conditioning latents (1, 32, 18, 18) - 8 past frames concatenated
   * @param action Action index (0=no-flap, 1=flap)
   * @param augLevel Augmentation bin index (0-15)
   * @returns Final latent at t=1 (1, 4, 18, 18)
   */
  async sample(
    z0: Float32Array,
    zCond: Float32Array,
    action: number,
    augLevel: number
  ): Promise<Float32Array> {
    if (!this.manager.resunet) {
      throw new Error('ResUNet not loaded');
    }

    const { latentChannels, latentSize, contextFrames } = this.config;
    const dt = 1.0 / this.numSteps;
    const latentElements = latentChannels * latentSize * latentSize;

    // Clone z0 to work with
    let z = new Float32Array(z0);

    // Pre-create tensors that don't change
    const zCondTensor = new ort.Tensor(
      'float32',
      zCond,
      [1, contextFrames * latentChannels, latentSize, latentSize]
    );
    const actionTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(action)]), [1]);
    const augTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(augLevel)]), [1]);

    // Euler integration
    for (let i = 0; i < this.numSteps; i++) {
      const t = (i + 0.5) * dt; // Midpoint rule

      // Create tensors for this step
      const zTensor = new ort.Tensor('float32', z, [1, latentChannels, latentSize, latentSize]);
      const tTensor = new ort.Tensor('float32', new Float32Array([t]), [1]);

      // Run model to get velocity prediction
      const results = await this.manager.resunet.run({
        x: zTensor,
        t: tTensor,
        action: actionTensor,
        z_cond: zCondTensor,
        aug_level: augTensor,
      });

      const vPred = results.v_pred.data as Float32Array;

      // Euler step: z = z + v * dt
      // Also clamp to [-3, 3] for stability
      for (let j = 0; j < latentElements; j++) {
        z[j] = Math.max(-3, Math.min(3, z[j] + vPred[j] * dt));
      }
    }

    return z;
  }

  /**
   * Generate Gaussian noise using Box-Muller transform.
   */
  generateNoise(size: number): Float32Array {
    const noise = new Float32Array(size);

    for (let i = 0; i < size; i += 2) {
      const u1 = Math.random();
      const u2 = Math.random();

      const r = Math.sqrt(-2 * Math.log(u1));
      const theta = 2 * Math.PI * u2;

      noise[i] = r * Math.cos(theta);
      if (i + 1 < size) {
        noise[i + 1] = r * Math.sin(theta);
      }
    }

    return noise;
  }

  /**
   * Generate noise as perturbation from current frame.
   */
  generatePerturbedNoise(current: Float32Array, scale: number): Float32Array {
    const noise = this.generateNoise(current.length);
    const result = new Float32Array(current.length);

    for (let i = 0; i < current.length; i++) {
      result[i] = current[i] + noise[i] * scale;
    }

    return result;
  }

  /**
   * Check if episode is done using the done head.
   * Should be called after sampling with the final latent.
   *
   * @param z Final latent at t=1
   * @param zCond Conditioning latents
   * @param action Action taken
   * @param augLevel Augmentation level
   * @returns Done probability (0-1), or null if done head not available
   */
  async checkDone(
    z: Float32Array,
    zCond: Float32Array,
    action: number,
    augLevel: number
  ): Promise<number | null> {
    if (!this.manager.resunet || !this.config.useDoneHead) {
      return null;
    }

    const { latentChannels, latentSize, contextFrames } = this.config;

    // Run model at t=1 to get done prediction
    const zTensor = new ort.Tensor('float32', z, [1, latentChannels, latentSize, latentSize]);
    const tTensor = new ort.Tensor('float32', new Float32Array([1.0]), [1]);
    const actionTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(action)]), [1]);
    const zCondTensor = new ort.Tensor(
      'float32',
      zCond,
      [1, contextFrames * latentChannels, latentSize, latentSize]
    );
    const augTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(augLevel)]), [1]);

    const results = await this.manager.resunet.run({
      x: zTensor,
      t: tTensor,
      action: actionTensor,
      z_cond: zCondTensor,
      aug_level: augTensor,
    });

    // Check if done_logit output exists
    if (!results.done_logit) {
      return null;
    }

    const doneLogit = (results.done_logit.data as Float32Array)[0];
    // Apply sigmoid to get probability
    const doneProb = 1 / (1 + Math.exp(-doneLogit));

    return doneProb;
  }
}
