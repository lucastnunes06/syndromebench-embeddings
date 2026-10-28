"""Capture the actual aggregate dashboard in a sandboxed browser.

Optional dependency: pip install playwright; playwright install chromium.
Reads the local HTML and local figures, without opening an external URL.
"""
import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(target=None, chromium_path=None):
    from playwright.sync_api import sync_playwright
    html=(ROOT/'dashboard'/'index.html').read_text(encoding='utf-8')
    def inline(match):
        path = ROOT/'outputs'/match.group(1)
        data=base64.b64encode(path.read_bytes()).decode('ascii')
        return 'src="data:image/png;base64,'+data+'"'
    html=re.sub(r'src="\.\./outputs/([a-z_]+\.png)"',inline,html)
    path=Path(target) if target else ROOT/'docs'/'dashboard.png'
    path.parent.mkdir(exist_ok=True)
    with sync_playwright() as p:
        options={'headless':True,'args':['--no-sandbox','--disable-dev-shm-usage']}
        if chromium_path: options['executable_path']=chromium_path
        browser=p.chromium.launch(**options)
        page=browser.new_page(viewport={'width':1440,'height':960},device_scale_factor=1)
        errors=[]
        page.on('pageerror', lambda error:errors.append(str(error)))
        page.set_content(html,wait_until='load',timeout=30000)
        # The JS sets the confusion src at startup; explicitly inline it for the sandbox capture.
        confusion=(ROOT/'outputs'/'confusion_matrix_cosine.png').read_bytes()
        encoded='data:image/png;base64,'+base64.b64encode(confusion).decode('ascii')
        page.locator('#confusion').evaluate('(img,uri)=>{img.src=uri}',encoded)
        assert page.locator('#metricbars .barrow').count()==4
        assert page.locator('#baselines tr').count()>=3
        if errors: raise RuntimeError('Browser JS errors: '+str(errors))
        page.screenshot(path=str(path),full_page=True)
        browser.close()
    print('Dashboard screenshot:',path)
    return path


if __name__=='__main__':
    render(chromium_path='/usr/bin/chromium')
