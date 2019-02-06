import scrapy
from scrapy import signals
from scrapy.crawler import CrawlerProcess
import pandas as pd
import scrapy
from scrapy.xlib.pydispatch import dispatcher


class MyItem(scrapy.Item):
    # ... other item fields ...
    image_urls = scrapy.Field()
    images = scrapy.Field()


class BlogSpider(scrapy.Spider):
    name = "quotes"

    years = []
    titles = []
    '''
    custom_settings = {
        "DOWNLOAD_DELAY": 1,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 10
    }
    '''

    def start_requests(self):
        dispatcher.connect(self.spider_closed, signals.spider_closed)

        urls = [
            'https://www.jackson-pollock.org/jackson-pollock-paintings.jsp',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # Get all albums paintings list:
        paintings = response.css('table a')
        for t in paintings:
            if len(t.css('a').xpath('@href')) == 1:
                #   Get title and URL of the image
                paintingUrl = 'https://www.jackson-pollock.org{}'.format(t.css('a').xpath('@href').extract()[0])
                #   Extact the image and save a file with the data inside
                yield scrapy.Request(url=paintingUrl, callback=self.extractData)

    def extractData(self, response):
        print(response)
        image = 'https://www.jackson-pollock.org' + response.css('table img').xpath('@src').extract()[0]
        #   Get the title and the year
        title_full = response.css('h1::text').extract()[1].strip().lower()
        title_full = title_full.split(', ')
        title = title_full[0].strip()
        year = title_full[1].replace('by jackson pollock', '').strip()

        self.years.append(year)
        self.titles.append(title)

        #   Extract the image using the custom pipeline and write a file with all other info
        yield {'image_urls': [image], 'titles':[title]}

    def spider_closed(self, spider):
        df = pd.DataFrame()
        df['year'] = self.years
        df['title'] = self.titles

        df.to_csv('./data/data.csv',index=False)
        return

process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    , "BOT_NAME": 'imagespider'
    , "ITEM_PIPELINES": {
        'custom_image_pipeline.CustomImagesPipeline': 1,
    }
    , "IMAGES_STORE": "C:\\Users\\Andrea\\PycharmProjects\\pollock-analysis\\data\\images"
})

process.crawl(BlogSpider)
process.start()  # the script will block here until the crawling is finished
print('hello')
