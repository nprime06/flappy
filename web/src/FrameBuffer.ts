/**
 * Circular buffer for maintaining past frame latents.
 * Used for conditioning the flow model on temporal context.
 */
export class FrameBuffer {
  private buffer: Float32Array[];
  private capacity: number;
  private latentElements: number;

  /**
   * Create a frame buffer.
   * @param capacity Number of frames to store (e.g., 8)
   * @param latentChannels Channels per latent (e.g., 4)
   * @param latentSize Spatial size of latent (e.g., 18)
   */
  constructor(capacity: number, latentChannels: number, latentSize: number) {
    this.capacity = capacity;
    this.latentElements = latentChannels * latentSize * latentSize;
    this.buffer = [];

    // Initialize with zeros
    for (let i = 0; i < capacity; i++) {
      this.buffer.push(new Float32Array(this.latentElements));
    }
  }

  /**
   * Push a new frame, removing the oldest.
   * @param latent Frame latent (1, C, H, W) flattened
   */
  push(latent: Float32Array): void {
    // Remove oldest
    this.buffer.shift();
    // Add newest (clone to avoid reference issues)
    this.buffer.push(new Float32Array(latent));
  }

  /**
   * Get the concatenated conditioning tensor.
   * @returns Concatenated latents (1, k*C, H, W) as flattened Float32Array
   */
  getConcatenated(): Float32Array {
    const result = new Float32Array(this.capacity * this.latentElements);

    for (let i = 0; i < this.capacity; i++) {
      result.set(this.buffer[i], i * this.latentElements);
    }

    return result;
  }

  /**
   * Get a specific frame from the buffer.
   * @param index Frame index (0 = oldest, capacity-1 = newest)
   */
  get(index: number): Float32Array {
    if (index < 0 || index >= this.capacity) {
      throw new Error(`Index ${index} out of bounds [0, ${this.capacity})`);
    }
    return this.buffer[index];
  }

  /**
   * Get the most recent frame.
   */
  getLatest(): Float32Array {
    return this.buffer[this.capacity - 1];
  }

  /**
   * Reset buffer to zeros.
   */
  reset(): void {
    for (let i = 0; i < this.capacity; i++) {
      this.buffer[i].fill(0);
    }
  }

  /**
   * Initialize buffer with the same frame repeated.
   * Useful for starting from a single initial frame.
   */
  fill(latent: Float32Array): void {
    for (let i = 0; i < this.capacity; i++) {
      this.buffer[i] = new Float32Array(latent);
    }
  }
}
