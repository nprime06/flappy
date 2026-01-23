import { ModelManager } from './ModelManager';
import { VAE } from './VAE';
import { EulerSampler } from './EulerSampler';
import { FrameBuffer } from './FrameBuffer';

export interface AppConfig {
  modelPath: string;
  canvasId: string;
  displayScale: number;
  numSteps: number;
  augLevel: number;
  noiseScale: number;
  doneThreshold: number;
  targetFps: number;
}

export class App {
  private config: AppConfig;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  private manager: ModelManager;
  private vae: VAE;
  private sampler: EulerSampler;
  private frameBuffer: FrameBuffer | null = null;

  private currentLatent: Float32Array | null = null;
  private action = 0; // 0 = no-flap, 1 = flap
  private running = false;
  private frameCount = 0;
  private startTime = 0;
  private lastFpsUpdate = 0;
  private lastFrameTime = 0;
  private fps = 0;
  private isGameOver = false;
  private gameOverTimer = 0;

  // UI elements
  private loadingEl: HTMLElement;
  private loadingBarEl: HTMLElement;
  private loadingTextEl: HTMLElement;
  private containerEl: HTMLElement;
  private statusEl: HTMLElement;
  private fpsEl: HTMLElement;

  constructor(config: AppConfig) {
    this.config = config;

    // Get canvas
    this.canvas = document.getElementById(config.canvasId) as HTMLCanvasElement;
    this.ctx = this.canvas.getContext('2d')!;

    // Get UI elements
    this.loadingEl = document.getElementById('loading')!;
    this.loadingBarEl = document.getElementById('loading-bar')!;
    this.loadingTextEl = document.getElementById('loading-text')!;
    this.containerEl = document.getElementById('container')!;
    this.statusEl = document.getElementById('status')!;
    this.fpsEl = document.getElementById('fps')!;

    // Initialize models
    this.manager = new ModelManager();
    this.vae = new VAE(this.manager);
    this.sampler = new EulerSampler(this.manager, config.numSteps);

    // Setup input handlers
    this.setupInputHandlers();
  }

  async init(): Promise<void> {
    // Load models with progress
    await this.manager.init(this.config.modelPath, (stage, progress) => {
      this.loadingBarEl.style.width = `${progress}%`;
      this.loadingTextEl.textContent = this.getLoadingMessage(stage);
    });

    // Initialize frame buffer
    const { latentChannels, latentSize, contextFrames } = this.manager.config!;
    this.frameBuffer = new FrameBuffer(contextFrames, latentChannels, latentSize);

    // Initialize with noise (will be replaced when we have initial frames)
    const latentElements = latentChannels * latentSize * latentSize;
    this.currentLatent = this.sampler.generateNoise(latentElements);
    this.frameBuffer.fill(this.currentLatent);

    // Show game UI
    this.loadingEl.classList.add('hidden');
    this.containerEl.classList.remove('hidden');

    // Update status with provider info
    const provider = this.manager.isWebGPU() ? 'WebGPU' : 'WASM';
    this.statusEl.textContent = `Press SPACE to flap (${provider})`;
  }

  private getLoadingMessage(stage: string): string {
    const messages: Record<string, string> = {
      provider: 'Checking WebGPU support...',
      config: 'Loading config...',
      encoder: 'Loading encoder...',
      decoder: 'Loading decoder...',
      resunet: 'Loading flow model...',
      warmup: 'Warming up (compiling shaders)...',
      done: 'Ready!',
    };
    return messages[stage] || 'Loading...';
  }

  private setupInputHandlers(): void {
    // Keyboard input
    document.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        this.action = 1;
      } else if (e.code === 'KeyR') {
        this.reset();
      } else if (e.code === 'Equal' || e.code === 'NumpadAdd') {
        // Increase steps
        this.sampler.numSteps = Math.min(50, this.sampler.numSteps + 5);
        this.statusEl.textContent = `Steps: ${this.sampler.numSteps}`;
      } else if (e.code === 'Minus' || e.code === 'NumpadSubtract') {
        // Decrease steps
        this.sampler.numSteps = Math.max(5, this.sampler.numSteps - 5);
        this.statusEl.textContent = `Steps: ${this.sampler.numSteps}`;
      }
    });

    document.addEventListener('keyup', (e) => {
      if (e.code === 'Space') {
        this.action = 0;
      }
    });

    // Touch input for mobile
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.action = 1;
    });

    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.action = 0;
    });

    // Mouse input
    this.canvas.addEventListener('mousedown', () => {
      this.action = 1;
    });

    this.canvas.addEventListener('mouseup', () => {
      this.action = 0;
    });
  }

  async start(): Promise<void> {
    if (this.running) return;

    this.running = true;
    this.startTime = performance.now();
    this.lastFpsUpdate = this.startTime;
    this.lastFrameTime = 0; // Allow first frame immediately
    this.frameCount = 0;
    this.isGameOver = false;

    // Show initial frame
    await this.renderCurrentFrame();

    // Start game loop using requestAnimationFrame
    requestAnimationFrame(() => this.gameLoop());
  }

  stop(): void {
    this.running = false;
  }

  reset(): void {
    if (!this.frameBuffer || !this.manager.config) return;

    const { latentChannels, latentSize } = this.manager.config;
    const latentElements = latentChannels * latentSize * latentSize;

    // Reset to noise
    this.currentLatent = this.sampler.generateNoise(latentElements);
    this.frameBuffer.fill(this.currentLatent);

    this.statusEl.textContent = 'Reset! Press SPACE to flap';
  }

  private async gameLoop(): Promise<void> {
    if (!this.running) return;

    const now = performance.now();
    const targetFrameTime = 1000 / this.config.targetFps;

    // Strict FPS cap: skip if not enough time has passed
    if (now - this.lastFrameTime < targetFrameTime) {
      requestAnimationFrame(() => this.gameLoop());
      return;
    }

    this.lastFrameTime = now;

    // Handle game over state
    if (this.isGameOver) {
      this.gameOverTimer -= targetFrameTime;
      if (this.gameOverTimer <= 0) {
        this.reset();
        this.isGameOver = false;
      }
      requestAnimationFrame(() => this.gameLoop());
      return;
    }

    try {
      await this.step();
    } catch (e) {
      console.error('Error in game loop:', e);
      this.statusEl.textContent = 'Error: ' + (e as Error).message;
      this.running = false;
      return;
    }

    // Update FPS counter
    this.frameCount++;
    if (now - this.lastFpsUpdate > 1000) {
      this.fps = this.frameCount / ((now - this.lastFpsUpdate) / 1000);
      this.fpsEl.textContent = `${this.fps.toFixed(1)} FPS | ${this.sampler.numSteps} steps`;
      this.frameCount = 0;
      this.lastFpsUpdate = now;
    }

    requestAnimationFrame(() => this.gameLoop());
  }

  private async step(): Promise<void> {
    if (!this.currentLatent || !this.frameBuffer || !this.manager.config) {
      throw new Error('Not initialized');
    }

    // Update frame buffer FIRST to include current frame in context
    // This is critical: we predict the NEXT frame, so context must include current frame
    this.frameBuffer.push(this.currentLatent);

    // Generate starting noise (perturbation from current frame)
    const z0 = this.sampler.generatePerturbedNoise(
      this.currentLatent,
      this.config.noiseScale
    );

    // Get conditioning from frame buffer (now includes currentLatent)
    const zCond = this.frameBuffer.getConcatenated();

    // Sample next frame
    const zNext = await this.sampler.sample(
      z0,
      zCond,
      this.action,
      this.config.augLevel
    );

    // Check if episode is done (if done head is available)
    const doneProb = await this.sampler.checkDone(
      zNext,
      zCond,
      this.action,
      this.config.augLevel
    );

    if (doneProb !== null && doneProb > this.config.doneThreshold) {
      this.handleGameOver(doneProb);
    }

    // Update current latent
    this.currentLatent = zNext;

    // Render
    await this.renderCurrentFrame();

    // Show action indicator (if not game over)
    if (!this.isGameOver) {
      if (this.action === 1) {
        this.statusEl.textContent = 'FLAP!';
      } else {
        this.statusEl.textContent = 'Press SPACE to flap';
      }
    }
  }

  private handleGameOver(doneProb: number): void {
    this.isGameOver = true;
    this.gameOverTimer = 2000; // 2 seconds before auto-reset
    this.statusEl.textContent = `GAME OVER! (${(doneProb * 100).toFixed(0)}% confidence) - Restarting...`;
    console.log(`Game over detected with probability ${doneProb.toFixed(3)}`);
  }

  private async renderCurrentFrame(): Promise<void> {
    if (!this.currentLatent) return;

    // Decode latent to image
    const imageData = await this.vae.decode(this.currentLatent);

    // Scale up and draw
    const { imageSize } = this.manager.config!;
    const scale = this.config.displayScale;

    // Create temporary canvas for scaling
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imageSize;
    tempCanvas.height = imageSize;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.putImageData(imageData, 0, 0);

    // Draw scaled (pixelated scaling is handled by CSS)
    this.ctx.drawImage(
      tempCanvas,
      0,
      0,
      imageSize * scale,
      imageSize * scale
    );
  }

  /**
   * Load initial frames from an image URL.
   * The image should be a single 144x144 frame.
   */
  async loadInitialFrame(imageUrl: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = async () => {
        try {
          // Draw to temp canvas to get ImageData
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = img.width;
          tempCanvas.height = img.height;
          const tempCtx = tempCanvas.getContext('2d')!;
          tempCtx.drawImage(img, 0, 0);
          const imageData = tempCtx.getImageData(0, 0, img.width, img.height);

          // Encode to latent
          const latent = await this.vae.encode(imageData);

          // Fill frame buffer with this frame
          this.currentLatent = latent;
          this.frameBuffer?.fill(latent);

          // Render
          await this.renderCurrentFrame();

          resolve();
        } catch (e) {
          reject(e);
        }
      };
      img.onerror = () => reject(new Error(`Failed to load image: ${imageUrl}`));
      img.src = imageUrl;
    });
  }
}
