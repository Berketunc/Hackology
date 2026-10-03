"""End-to-end browser checks against the running workspace.

Requires the optional requirements-dev.txt and `python -m playwright install chromium`.
Runs in an isolated browser context; never reads or modifies a user's browser profile.
"""
import csv
import io
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = []
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={'width':1440,'height':1100}, device_scale_factor=1)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:7860', wait_until='networkidle')
        expect(page.locator('#intro')).not_to_be_visible(timeout=10000)
        expect(page.locator('#scan-summary')).to_be_visible(timeout=90000)
        expect(page.locator('#ranked-rows tr')).to_have_count(12)
        expect(page.locator('.heat-cell')).to_have_count(372)
        expect(page.locator('#selected-content')).to_contain_text('BLOSUM MLP')
        page.screenshot(path=str(ROOT/'reports/workspace_desktop.png'), full_page=True)
        checks.append('Real default scan: 62 windows × 6 alleles; ranked table and heatmap')

        page.locator('#ranked-rows [data-shortlist]').first.click()
        expect(page.locator('#short-count')).to_have_text('· 1')
        page.locator('[data-tier="high"]').click()
        for text in page.locator('#ranked-rows .tag').all_text_contents():
            assert text == '≥6 h'
        page.locator('[data-tier="all"]').click()
        page.locator('.heat-cell').first.click()
        expect(page.locator('#selected-title')).to_contain_text('MKTAYIAKQ')
        with page.expect_download() as download:
            page.locator('#export-scan').click()
        exported=list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text())))
        assert len(exported)==62 and exported[0]['model']=='BLOSUM MLP (reference)'
        checks.append('Tier filters, pair inspection, shortlist and CSV export')

        page.locator('#save-scan').click()
        page.locator('#run-name').fill('QA protein scan')
        page.locator('#save-dialog').get_by_role('button',name='Save run',exact=True).click()
        page.get_by_role('link',name='Saved runs',exact=True).click()
        expect(page.locator('#saved-runs')).to_contain_text('QA protein scan')
        page.reload(wait_until='networkidle')
        expect(page.locator('#intro')).not_to_be_visible(timeout=10000)
        expect(page.locator('#saved-runs')).to_contain_text('QA protein scan')
        page.locator('[data-open-run]').first.click()
        expect(page.locator('#short-count')).to_have_text('· 1')
        expect(page.locator('#protein')).to_have_value('MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV')
        checks.append('Save, reload persistence and full scan restoration')

        page.get_by_role('link',name='Single pair',exact=True).click()
        page.locator('[data-example="0"]').click()
        expect(page.locator('#single-result tbody tr')).to_have_count(5, timeout=90000)
        expect(page.locator('#single-result')).to_contain_text('1,070')
        expect(page.locator('#single-result')).to_contain_text('Training exposure:')
        page.screenshot(path=str(ROOT/'reports/workspace_single.png'), full_page=True)
        page.locator('#single-peptide').fill('AAAAAAAAX')
        page.locator('#single-submit').click()
        expect(page.locator('#notice')).to_contain_text('standard amino-acid')
        checks.append('All five real model predictions, dataset exposure and invalid input feedback')

        page.get_by_role('link',name='Batch',exact=True).click()
        page.locator('#batch-file').set_input_files({'name':'pairs.csv','mimeType':'text/csv',
            'buffer':b'peptide,allele\nFSVQRNLPF,HLA-B*15:01\nAEMGANLCV,HLA-B*13:02\n'})
        page.locator('#batch-submit').click()
        expect(page.locator('#batch-rows tr')).to_have_count(2, timeout=30000)
        page.locator('#batch-rows [data-shortlist]').first.click()
        page.locator('#batch-rows [data-shortlist]').nth(1).click()
        checks.append('CSV file upload, batch scoring and shortlisting')

        page.get_by_role('link',name='Compare models',exact=True).click()
        page.locator('#compare-plm').uncheck()
        page.locator('#run-comparison').click()
        expect(page.locator('#comparison-results tbody tr')).to_have_count(3, timeout=30000)
        expect(page.locator('#comparison-results thead th')).to_have_count(4)
        checks.append('Shortlist comparison across the three sequence baselines')

        page.get_by_role('link',name='Methods',exact=True).click()
        expect(page.locator('#dataset-facts')).to_contain_text('27,031')
        page.set_viewport_size({'width':390,'height':844})
        page.get_by_role('link',name='Scan protein',exact=True).click()
        page.screenshot(path=str(ROOT/'reports/workspace_mobile.png'), full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile viewport overflows horizontally'
        expect(page.locator('#scan-submit')).to_be_visible()
        checks.append('Methods content and 390px mobile layout without page overflow')
        assert not errors, errors
        browser.close()
    result=dict(checks=checks, passed=True, browser_errors=errors, browser='isolated headless Chromium')
    (ROOT/'results/benchmark/workspace_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
