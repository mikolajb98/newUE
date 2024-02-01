from flask import Flask
from flask_restful import Resource, Api
import cv2

app = Flask(__name__)
api = Api(app)


# konfigurowanie detektora osób
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# rozpoznawanie z pliku wskazanego w zmiennej statycznej
class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('images/dworzec_kato.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(4, 4))
        # winStride zmiana tego parametru zwieksza/zmnijsza dokladnsosc
        # - im mniejsza tym dokladniej lczy przesuniecie okienka

        return {'count': len(boxes)}



class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(HelloWorld, '/test')  # endpoint test
api.add_resource(PeopleCounter, '/')  # endpoint main

if __name__ == '__main__':
    app.run(debug=True)