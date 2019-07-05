from http.server import BaseHTTPRequestHandler, HTTPServer
from Producer import Kafka_producer
from Consumer import Kafka_consumer
from io import BytesIO
import cgi
import json

class MyHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.do_POST()
        # self.wfile.write("这是一个http后台服务。".encode())

    def do_POST(self):
        self.send_header('Content-type', 'text/html')
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                environ={'REQUEST_METHOD': 'POST'})

        userid = form.getvalue("user")
        image = form.getvalue("image")

        result = {
            'image': image,
            'user': userid
        }
        print(userid)

        producer = Kafka_producer("G4master", 9092, "inputImage")
        producer.sendjsondata(result)

        consumer = Kafka_consumer("G4master", 9092, "outputResult")
        jsonData = consumer.getUserFeedback(userid)
        json_str = json.dumps(jsonData)
        json_encode = json_str.encode("utf-8")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json_encode)
        #self.send_response(200)
        #self.end_headers()


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
