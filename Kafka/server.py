from http.server import BaseHTTPRequestHandler, HTTPServer
from Producer import Kafka_producer
from Consumer import Kafka_consumer


class MyHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.do_POST()
        # self.wfile.write("这是一个http后台服务。".encode())

    def do_POST(self):
        userid = self.headers.get('user')
        image = self.headers.get('image')
        self.send_response(200)
        self.end_headers()
        result = {
            'image': image,
            'user': userid
        }

        producer = Kafka_producer("G4master", 9092, "inputImage")
        producer.sendjsondata(result)

        consumer = Kafka_consumer("G4master", 9092, "outputResult")
        jsonData = consumer.getUserFeedback(userid)

        self.wfile.write(jsonData.encode())


def main():
    try:
        server = HTTPServer(('10.244.12.12', 11450), MyHandler) #启动服务
        print('Welcome to the server.......')
        server.serve_forever()  # 一直运行
    except KeyboardInterrupt:
        print('Shutting done server')
        server.socket.close()


if __name__ == '__main__':
    main()
