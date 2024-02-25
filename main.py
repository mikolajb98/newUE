from flask import Flask, request
from flask_restful import Resource, Api
import cv2
import requests
import numpy as np


app = Flask(__name__)
api = Api(app)


# konfigurowanie detektora osób
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# rozpoznawanie z pliku wskazanego w zmiennej statycznej
class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('images/dworzec_wro.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(2, 2))
        # winStride zmiana tego parametru zwieksza/zmnijsza dokladnsosc
        # - im mniejsza tym dokladniej lczy przesuniecie okienka
        return {'count': len(boxes)}


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class UrlPeopleCounter (Resource):
    def get(self):
        urlParam = request.args.get('url', '')
        # https://bi.im-g.pl/im/6e/4d/1c/z29678446AMP,Lotnisko-w-Pyrzowicach-ma-wielkie-plany--Beda-spor.jpg
        if not urlParam:
            return {"error": "Url missing or empty"}, 400
        urlResponse = requests.get(urlParam)
        if urlResponse.status_code != 200:
            return {"error": "Image load fail"}, 400
        image_array = np.frombuffer(urlResponse.content, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        return {"count": len(boxes)}

class PostPeopleCounter(Resource):
    def post(self):
        try:
            data = request.get_data()
            array = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)

            if image is None:
                return {'error': 'Image load fail"'}, 500

            boxes, weights = hog.detectMultiScale(image, winStride=(8, 8))

            return {'count': len(boxes)}
        except Exception as e:
            return {'error': f'Error occured: {str(e)}'}, 500


api.add_resource(HelloWorld, '/test')
api.add_resource(PeopleCounter, '/')
api.add_resource(UrlPeopleCounter, '/url-counter')
#http://127.0.0.1:8000/url-counter?url=https://bi.im-g.pl/im/6e/4d/1c/z29678446AMP,Lotnisko-w-Pyrzowicach-ma-wielkie-plany--Beda-spor.jpg
api.add_resource(PostPeopleCounter, '/post-counter')
#curl -X POST -H "Content-Type: image/jpeg" --data-binary "@\Users\mikol\OneDrive\Pulpit\lotnisko.jpg" http://127.0.0.1:8000/post-counter


if __name__ == '__main__':
    app.run(debug=True, port=8000)
