from playwright.sync_api import sync_playwright

def test_playwright():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto('https://finance.yahoo.com/quote/AAPL')

        title = page.title()
        content = page.content()

        print(f"Page title: {title}")
        print(f"Page content (length): {len(content)}")

        browser.close()

if __name__ == "__main__":
    test_playwright()