from scrapy.pipelines.images import ImagesPipeline
import scrapy

class CustomImagesPipeline(ImagesPipeline):

    def get_media_requests(self, item, info):
        # use 'accession' as name for the image when it's downloaded
        image_url = item.get('image_urls')[0]
        title = item.get('titles')[0]
        return scrapy.Request(image_url, meta={'title':title})

    # write in current folder using the name we chose before
    def file_path(self, request, response=None, info=None):
        return '{}.jpg'.format(request.meta['title'])
