"""Validate intro timing, accessibility, replay, and responsive layout in Chromium."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={'width':1440,'height':1000})
        page = context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto('http://127.0.0.1:7860',wait_until='domcontentloaded')
        expect(page.locator('#intro')).to_be_visible()
        expect(page.locator('#skip-intro')).to_be_focused()
        assert page.locator('.intro-helix').count()==4
        assert page.locator('.intro-residue').count()==16
        assert page.locator('.intro-residue.is-inside').count()==9
        page.wait_for_function("getComputedStyle(document.querySelector('.intro-tagline')).opacity === '1'",timeout=6000)
        expect(page.locator('#intro')).to_be_visible()
        page.screenshot(path=str(ROOT/'reports/workspace_intro.png'))
        expect(page.locator('#intro')).not_to_be_visible(timeout=3000)
        expect(page.locator('#scan-summary')).to_be_visible(timeout=60000)
        checks.append('Helices and 16 residues animate; nine align with final window; tagline appears before automatic exit')

        page.locator('#replay-intro').click()
        expect(page.locator('#intro')).to_be_visible()
        page.locator('#skip-intro').click()
        expect(page.locator('#intro')).not_to_be_visible()
        expect(page.locator('#replay-intro')).to_be_focused()
        page.locator('#replay-intro').click()
        page.keyboard.press('Escape')
        expect(page.locator('#intro')).not_to_be_visible()
        checks.append('Replay restarts; Skip and Escape dismiss immediately and return keyboard focus')

        page.set_viewport_size({'width':390,'height':844})
        page.locator('#replay-intro').click()
        page.wait_for_function("getComputedStyle(document.querySelector('.intro-tagline')).opacity === '1'")
        page.screenshot(path=str(ROOT/'reports/workspace_intro_mobile.png'))
        assert page.locator('#intro').evaluate('(el) => el.scrollWidth <= el.clientWidth')
        skip=page.locator('#skip-intro').bounding_box()
        assert skip['y']>=0 and skip['y']+skip['height']<=844
        page.locator('#skip-intro').click()
        page.set_viewport_size({'width':844,'height':390})
        page.locator('#replay-intro').click()
        skip=page.locator('#skip-intro').bounding_box()
        assert skip['y']>=0 and skip['y']+skip['height']<=390
        page.keyboard.press('Escape')
        checks.append('390px portrait and 390px-high landscape preserve visible dismissal controls without horizontal overflow')

        reduced = browser.new_context(viewport={'width':390,'height':844},reduced_motion='reduce')
        static = reduced.new_page()
        static.on('pageerror',lambda e:errors.append(str(e)))
        static.goto('http://127.0.0.1:7860',wait_until='domcontentloaded')
        expect(static.locator('#intro')).not_to_be_visible()
        expect(static.locator('#scan-summary')).to_be_visible(timeout=60000)
        static.locator('#replay-intro').click()
        expect(static.locator('#intro')).to_be_visible()
        assert static.locator('#intro').evaluate("el => el.getAnimations({subtree:true}).length") == 0
        static.locator('#skip-intro').click()
        checks.append('Reduced motion bypasses automatic intro; explicit replay is static; workspace still loads')
        assert not errors,errors
        browser.close()
    record=dict(passed=True,checks=checks,browser_errors=errors)
    (ROOT/'results/benchmark/intro_checks.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
