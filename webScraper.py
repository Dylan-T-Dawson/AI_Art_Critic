# Web Scraper for pulling sprites from pkmn database
from bs4 import BeautifulSoup
import requests

# Request HTML data and pass it as text
request = requests.get('https://www.shutterstock.com/search/black-and-white').text
# load the requested HTML into BeautifulSoup
soup = BeautifulSoup(request, 'html.parser') 
# storing image objects as all HTML elements of type div w/ img
images = soup.select("div img")
# get 200 images
for i in range(1, 50):
    # store the image url (identified by src element)
    images_url = images[i]['src']
    # getting the content of the image url (i.e. the image itself)
    img_data = requests.get(images_url).content 
    # creating files in the pkmnSprites folder based on the name of the img src
    with open('C:/Users/Ryan/Desktop/AI_Art_Critic/data/data/Color/black_and_white' + str(i) + '.png', 'wb') as handler: 
        # creating the file from the extracted image data
        handler.write(img_data) 