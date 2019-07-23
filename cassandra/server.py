from http.server import BaseHTTPRequestHandler, HTTPServer
from cassandra.cluster import Cluster
from io import BytesIO
import cgi
import json

class MyHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.do_POST()
        # self.wfile.write("这是一个http后台服务。".encode())

    def do_POST(self):
        #get post form
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                environ={'REQUEST_METHOD': 'POST'})
        userid = form.getvalue("user")







        global session
        query = "select * from records where id = %s AND time <= %s "
        rows = session.execute(query, (id,t))

        jsonData = {

            'user': userid
        }



        json_str = json.dumps(jsonData)
        json_encode = json_str.encode("utf-8")

        self.send_response(200)
        self.end_headers()
        self.wfile.write(json_encode)


def main():
    try:
        # link to cluster
        cluster = Cluster(['G401', 'G402'])  # 随意写两个就能找到整个集群
        session = cluster.connect()
        global session
        server = HTTPServer(('10.244.12.12', 11450), MyHandler) #启动服务
        print('Welcome to the server.......')
        server.serve_forever()  # 一直运行
    except KeyboardInterrupt:
        print('Shutting done server')
        server.socket.close()


if __name__ == '__main__':
    main()

