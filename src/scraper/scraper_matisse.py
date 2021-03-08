import scrapy
from scrapy import signals
from scrapy.crawler import CrawlerProcess
import pandas as pd
import scrapy
from src.constants import *
import re
#from scrapy.xlib.pydispatch import dispatcher
class Paintings(scrapy.Item):
    # ... other item fields ...
    image_urls = scrapy.Field()
    images = scrapy.Field()


class PaintingsSpider(scrapy.Spider):
    name = "paintings"

    years = []
    titles = []
    custom_settings = {
        "DOWNLOAD_DELAY": 3,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 10
    }

    def start_requests(self):

        urls = [
            'https://www.henrimatisse.org/paintings.jsp',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # Get all albums paintings list:
        paintings = response.css('table a')
        for t in paintings:
            if len(t.css('a').xpath('@href')) == 1:
                #   Get title and URL of the image
                paintingUrl = 'https://www.henrimatisse.org{}'.format(t.css('a').xpath('@href').extract()[0])
                #   Extact the image and save a file with the data inside
                yield scrapy.Request(url=paintingUrl, callback=self.extractData)

    def extractData(self, response):
        print(response)
        image = 'https://www.henrimatisse.org' + response.css('table img').xpath('@src').extract()[0]
        #   Get the title and the year
        if image and len(response.css('h1 ::text').extract())>=1:
            title_full = response.css('h1 ::text').extract()[1].strip().lower()
            if title_full:
                title_full = title_full.split(', ')
                title = title_full[0].strip()
                if(len(title_full)>1):
                    year = title_full[1].replace('by henri matisse', '').strip()
                    year = re.sub('\D','',year)
                    if(year and int(year)>1900):
                        self.years.append(year)
                        self.titles.append(title)
            #Extract the image using the custom pipeline and write a file with all other info
            yield {'image_urls': [image], 'titles':[title]}

    def closed(self, reason):
        df = pd.DataFrame()
        df['year'] = self.years
        df['title'] = self.titles
        df.to_csv('{}/data.csv'.format(DATA_FOLDER),index=False)

process = CrawlerProcess({
      'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
      , "BOT_NAME": 'imagespider'
      , "ITEM_PIPELINES": {
          'src.scraper.custom_image_pipeline.CustomImagesPipeline': 1,
      }
      , "IMAGES_STORE": '{}/images'.format(DATA_FOLDER)
  })
