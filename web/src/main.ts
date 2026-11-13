/**
 * Main application entry point for LearnSign.
 * Mounts the root container and initialises the user interface.
 */

/**
 * Initializes and mounts the LearnSign application to the DOM.
 * Finds the `#app` element and sets up the root title and container.
 */
export function mountApp(): void {
  const rootElement = document.getElementById('app');
  if (!rootElement) {
    // Why: Defensively handle the edge case where the DOM has not rendered #app yet.
    return;
  }

  rootElement.innerHTML = `
    <main class="learnsign-container">
      <h1>LearnSign</h1>
      <p class="subtitle">Zero-cost, browser-only, real-time American Sign Language tutor</p>
    </main>
  `;
}

// Automatically mount when loaded in the browser document context
if (typeof document !== 'undefined') {
  mountApp();
}
