import { App } from './App';

async function main() {
  // Configuration
  const config = {
    modelPath: '/models',
    canvasId: 'game-canvas',
    displayScale: 3, // 144 * 3 = 432px
    numSteps: 20, // Euler integration steps (adjust for performance)
    augLevel: 8, // Middle of 0-15 range for stability
    noiseScale: 0.1, // Perturbation scale for z0
    doneThreshold: 0.5, // Probability threshold for episode termination
    targetFps: 30, // Maximum frames per second
  };

  try {
    // Create and initialize app
    const app = new App(config);
    await app.init();

    // Try to load initial frame if available
    try {
      await app.loadInitialFrame('/models/initial_frame.png');
      console.log('Loaded initial frame');
    } catch {
      console.log('No initial frame found, starting with noise');
    }

    // Start game loop
    await app.start();
  } catch (error) {
    console.error('Failed to initialize:', error);

    // Show error to user
    const loadingText = document.getElementById('loading-text');
    if (loadingText) {
      loadingText.textContent = `Error: ${(error as Error).message}`;
      loadingText.style.color = '#ff6b6b';
    }
  }
}

// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', main);
} else {
  main();
}
