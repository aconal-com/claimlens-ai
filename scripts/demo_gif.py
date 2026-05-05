#!/usr/bin/env python3
"""
ClaimLens AI — Browser demo script
Uses Playwright to walk through the web UI, submit a test form,
capture screenshots, and compile them into a GIF.
"""

import asyncio
import os
import time
import shutil
from pathlib import Path

from playwright.async_api import async_playwright

SCREENSHOT_DIR = Path("/tmp/claimlens_screenshots")
FRONTEND_URL = "http://localhost:8000/"
TEST_IMAGE = "/tmp/test_claim.jpg"
GIF_PATH = str(SCREENSHOT_DIR / "claimlens_demo.gif")
GIF_FPS = 1.5  # seconds per frame

SCREENSHOT_DIR.mkdir(exist_ok=True)

# Cleanup old screenshots
for f in SCREENSHOT_DIR.glob("step_*.png"):
    f.unlink()


async def main():
    screenshots = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1200, "height": 900},
        )
        page = await context.new_page()

        # ── Step 1: Load the page ──────────────────────────────────────
        print("Step 1: Loading page...")
        await page.goto(FRONTEND_URL, wait_until="networkidle", timeout=15000)
        await page.screenshot(path=str(SCREENSHOT_DIR / "step_1_loaded.png"))
        screenshots.append(str(SCREENSHOT_DIR / "step_1_loaded.png"))

        # Check API status
        api_status = await page.text_content("#statusText")
        print(f"  API status text: {api_status}")

        # ── Step 2: Fill optional form fields ──────────────────────────
        print("Step 2: Filling optional form fields...")
        await page.fill('input[name="name"]', "Jan de Vries")
        await page.fill('input[name="phone"]', "+31 6 1234 5678")
        await page.fill('input[name="policy_number"]', "POL-A-4589021")
        await page.fill('input[name="accident_location"]', "Kruisdrift / Parkkant, Utrecht")
        await page.screenshot(path=str(SCREENSHOT_DIR / "step_2_form_filled.png"))
        screenshots.append(str(SCREENSHOT_DIR / "step_2_form_filled.png"))

        # ── Step 3: Upload test image ──────────────────────────────────
        print("Step 3: Uploading test image...")
        # Directly set the file on the hidden input
        file_input = page.locator("#file-input")
        await file_input.set_input_files(TEST_IMAGE)
        await asyncio.sleep(0.5)

        # Check if filename shows
        filename_el = page.locator("#uploadFilename")
        filename_text = await filename_el.text_content() if await filename_el.is_visible() else ""
        print(f"  Uploaded file: {filename_text}")
        await page.screenshot(path=str(SCREENSHOT_DIR / "step_3_file_uploaded.png"))
        screenshots.append(str(SCREENSHOT_DIR / "step_3_file_uploaded.png"))

        # ── Step 4: Verify OCR pipeline is selected ────────────────────
        print("Step 4: Pipeline mode check...")
        ocr_btn = page.locator('[data-mode="ocr"]')
        is_ocr_active = await ocr_btn.get_attribute("class")
        print(f"  OCR button class: {is_ocr_active}")

        # ── Step 5: Click Analyze Claim ─────────────────────────────────
        print("Step 5: Submitting claim...")
        await page.click("#submitBtn")
        await page.screenshot(path=str(SCREENSHOT_DIR / "step_5_submitted.png"))
        screenshots.append(str(SCREENSHOT_DIR / "step_5_submitted.png"))

        # Wait for results card to appear (max 60s)
        print("  Waiting for analysis to complete...")
        try:
            await page.wait_for_selector("#resultCard.visible", timeout=90000)
            print("  Results appeared!")
        except Exception as e:
            print(f"  Results wait timeout: {e}")

        # ── Step 6: Results loaded ─────────────────────────────────────
        print("Step 6: Capturing results...")
        await asyncio.sleep(2)  # Let JS render fully
        await page.screenshot(path=str(SCREENSHOT_DIR / "step_6_results.png"))
        screenshots.append(str(SCREENSHOT_DIR / "step_6_results.png"))

        # ── Step 7: Expand JSON ────────────────────────────────────────
        print("Step 7: Expanding JSON output...")
        await page.click(".json-toggle")
        await asyncio.sleep(0.5)
        await page.screenshot(path=str(SCREENSHOT_DIR / "step_7_json_open.png"))
        screenshots.append(str(SCREENSHOT_DIR / "step_7_json_open.png"))

        await browser.close()

    print(f"\nScreenshots captured: {len(screenshots)}")
    for s in screenshots:
        print(f"  {s}")

    # ── Build GIF ────────────────────────────────────────────────────────
    print("\nBuilding GIF...")
    import imageio

    images = []
    for path in screenshots:
        img = imageio.imread(path)
        # Resize to 900px wide for reasonable GIF file size
        from PIL import Image
        pil_img = Image.fromarray(img)
        ratio = 900 / pil_img.width
        new_h = int(pil_img.height * ratio)
        pil_img = pil_img.resize((900, new_h), Image.LANCZOS)
        import numpy as np
        images.append(np.array(pil_img))

    imageio.mimsave(GIF_PATH, images, fps=GIF_FPS, loop=0)
    size_mb = os.path.getsize(GIF_PATH) / 1024 / 1024
    print(f"\nGIF saved: {GIF_PATH}")
    print(f"Size: {size_mb:.1f} MB, {len(images)} frames @ {GIF_FPS}s/frame")

    # Copy to desktop for easy sharing
    desktop = Path.home() / "Desktop" / "claimlens_demo.gif"
    shutil.copy(GIF_PATH, desktop)
    print(f"Also copied to: {desktop}")


if __name__ == "__main__":
    asyncio.run(main())
