import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config for AEGISNET C2 console.
 *
 * Tests require the Python bridge server to be running on port 8765 and the
 * Vite dev server to be running on port 5173.  Run both with:
 *
 *   python -m sim.bridge.server &
 *   npm run dev &
 *   npm run test
 */
export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  retries: 1,
  use: {
    baseURL: "http://localhost:5173",
    headless: true,
    viewport: { width: 1400, height: 900 },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  // Start the bridge + vite when running `npm test`
  webServer: [
    {
      command: "python -m sim.bridge.server --port 8765",
      url: "http://localhost:8765",
      reuseExistingServer: !process.env["CI"],
      timeout: 15_000,
      cwd: "..",
    },
    {
      command: "npm run dev",
      url: "http://localhost:5173",
      reuseExistingServer: !process.env["CI"],
      timeout: 15_000,
    },
  ],
});
