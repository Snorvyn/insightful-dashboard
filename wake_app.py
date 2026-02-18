import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def wake_my_app():
    url = "https://insightful-dashboard-sa-ai-data-engineering.streamlit.app/"
    options = Options()
    options.add_argument("--headless") # Runs without a visible window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=options)
    print(f"Visiting {url} to keep it awake...")
    driver.get(url)
    
    # Wait 15 seconds to ensure the WebSocket connects
    time.sleep(15) 
    
    print("Success! App is awake.")
    driver.quit()

if __name__ == "__main__":
    wake_my_app()
