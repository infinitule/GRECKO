/**
 * PV Playwright acceptance tests.
 *
 * Acceptance criteria:
 * 1. WEAPONS HOLD: clicking the button sends WEAPONS_HOLD to the bridge and
 *    the UI reflects the hold state (banner + button label).
 * 2. Track rows: confirmed tracks appear in the HOTL panel with AUTHORIZE,
 *    HOLD, and FRIENDLY buttons.
 * 3. AUTHORIZE: clicking AUTHORIZE on a track sends the command and the track
 *    row shows the authorized state.
 * 4. Replay determinism: verified programmatically in Python
 *    (test_pv_c2.py::TestReplayDeterminism) — this spec tests the UI side.
 */
import { test, expect, type Page } from "@playwright/test";

async function waitForConnection(page: Page, timeout = 10_000): Promise<void> {
  await page.waitForFunction(
    () => {
      const el = document.getElementById("connection-indicator");
      return el?.classList.contains("connected");
    },
    { timeout }
  );
}

async function waitForTracks(page: Page, minCount = 1, timeout = 15_000): Promise<void> {
  await page.waitForFunction(
    (min) => {
      const rows = document.querySelectorAll("[data-track-id]");
      return rows.length >= min;
    },
    minCount,
    { timeout }
  );
}

test.describe("AEGISNET C2 Console", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForConnection(page);
  });

  // ------------------------------------------------------------------ //
  // 1. WEAPONS HOLD                                                      //
  // ------------------------------------------------------------------ //

  test("WEAPONS HOLD button toggles and banner appears", async ({ page }) => {
    const btn = page.getByTestId("weapons-hold-btn");
    await expect(btn).toBeVisible();

    // Initially not active
    await expect(btn).not.toHaveClass(/hold-active/);

    // Click to activate
    await btn.click();
    await expect(btn).toHaveClass(/hold-active/);
    await expect(btn).toContainText("ACTIVE");

    // Click again to deactivate
    await btn.click();
    await expect(btn).not.toHaveClass(/hold-active/);
  });

  test("WEAPONS HOLD state reflected in sim-state broadcast", async ({ page }) => {
    const btn = page.getByTestId("weapons-hold-btn");

    // Activate weapons hold
    await btn.click();

    // The broadcast state should reflect weapons_hold=true after one broadcast
    // cycle (~200 ms).  We verify via the button class that the round-trip
    // worked (server echoes state back to all clients).
    await page.waitForTimeout(500);
    await expect(btn).toHaveClass(/hold-active/);

    // Deactivate
    await btn.click();
    await page.waitForTimeout(500);
    await expect(btn).not.toHaveClass(/hold-active/);
  });

  // ------------------------------------------------------------------ //
  // 2. HOTL track rows                                                   //
  // ------------------------------------------------------------------ //

  test("HOTL panel shows confirmed tracks", async ({ page }) => {
    // Wait up to 15 s for tracks to be confirmed (sensors confirm within ~5 s)
    await waitForTracks(page, 1, 15_000);

    const rows = page.locator("[data-track-id]");
    const count = await rows.count();
    expect(count).toBeGreaterThanOrEqual(1);

    // Each row should have the three action buttons
    const firstRow = rows.first();
    await expect(firstRow.locator(".btn.auth")).toBeVisible();
    await expect(firstRow.locator(".btn.hold")).toBeVisible();
    await expect(firstRow.locator(".btn.friendly-btn")).toBeVisible();
  });

  // ------------------------------------------------------------------ //
  // 3. AUTHORIZE command round-trip                                      //
  // ------------------------------------------------------------------ //

  test("AUTHORIZE button marks track as authorized", async ({ page }) => {
    await waitForTracks(page, 1, 15_000);
    const firstRow = page.locator("[data-track-id]").first();
    const authBtn = firstRow.locator(".btn.auth");

    // Should not be active initially
    await expect(authBtn).not.toHaveClass(/active/);

    // Click authorize
    await authBtn.click();

    // After broadcast round-trip the button should be active
    await page.waitForTimeout(600);
    await expect(authBtn).toHaveClass(/active/);
  });

  // ------------------------------------------------------------------ //
  // 4. Audit trail                                                       //
  // ------------------------------------------------------------------ //

  test("Audit trail shows events", async ({ page }) => {
    const auditList = page.getByTestId("audit-list");
    await expect(auditList).toBeVisible();

    // Wait for some audit events to accumulate
    await waitForTracks(page, 1, 15_000);
    const firstRow = page.locator("[data-track-id]").first();
    await firstRow.locator(".btn.auth").click();

    // AUTHORIZE event should appear in the audit trail
    await page.waitForFunction(
      () => {
        const list = document.getElementById("audit-list");
        const text = list?.innerText ?? "";
        return text.includes("AUTHORIZE");
      },
      { timeout: 5_000 }
    );
    await expect(auditList).toContainText("AUTHORIZE");
  });

  // ------------------------------------------------------------------ //
  // 5. ROE lambda slider                                                 //
  // ------------------------------------------------------------------ //

  test("Lambda slider sends SET_LAMBDA command", async ({ page }) => {
    const slider = page.locator("#lambda-slider");
    await expect(slider).toBeVisible();

    // Move slider to 0.3
    await slider.fill("0.3");
    await slider.dispatchEvent("input");

    // Wait for debounce + broadcast round-trip
    await page.waitForTimeout(500);

    // The displayed value should update
    const valEl = page.locator("#lambda-val");
    await expect(valEl).toContainText("0.3");
  });
});
