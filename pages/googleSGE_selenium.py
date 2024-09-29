from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def login_to_google(email, password):
    # Set Chrome options to disable personalization
    options = Options()
    prefs = {"profile.default_content_setting_values.notifications" : 2}
    options.add_experimental_option("prefs",prefs)

    # Create a new instance of the Chrome driver
    driver = webdriver.Chrome(executable_path='./chromedriver', options=options)

    # Go to Google's sign in page
    driver.get('https://accounts.google.com/ServiceLogin')

    # Wait until the email field is available
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'identifierId')))

    # Fill the email field
    email_field = driver.find_element_by_id('identifierId')
    email_field.send_keys(email)

    # Click on the Next button
    driver.find_element_by_id('identifierNext').click()

    # Wait until the password field is available
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'password')))

    # Fill the password field
    password_field = driver.find_element_by_name('password')
    password_field.send_keys(password)

    # Click on the Next button
    driver.find_element_by_id('passwordNext').click()

    # Add delay to wait for login
    time.sleep(5)

    # Close the browser
    driver.quit()

# Call the function
login_to_google('rvsge93@gmail.com', 'RedVentures246')