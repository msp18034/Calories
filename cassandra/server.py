from http.server import BaseHTTPRequestHandler, HTTPServer
from cassandra.cluster import Cluster
from io import BytesIO
import cgi
import json
import datetime

class MyHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.do_POST()
        # self.wfile.write("这是一个http后台服务。".encode())

    def do_POST(self):
        #get post form
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                environ={'REQUEST_METHOD': 'POST'})
        userid = form.getvalue("user")
        now = datetime.datetime.now()
        day = datetime.timedelta(days=1)
        before24 = now-day

        global session
        query = "select json * from records where id = %s AND time > %s AND time <= %s "
        rows = session.execute(query, (userid, before24, now))
        results = []
        for row in rows:
            jsonData = row[0]
            json_str = json.dumps(jsonData)
            results.append(json_str)
        results_str = '|'.join(results)
        results_encode = results_str.encode("utf-8")

        self.send_response(200)
        self.end_headers()
        self.wfile.write(results_encode)


def main():
    try:
        # link to cluster
        cluster = Cluster(['G401', 'G402'])  # 随意写两个就能找到整个集群
        session = cluster.connect("fooddiary")
        global session
        server = HTTPServer(('10.244.12.12', 11451), MyHandler) #启动服务
        print('Welcome to the server.......')
        server.serve_forever()  # 一直运行
    except KeyboardInterrupt:
        print('Shutting done server')
        server.socket.close()


if __name__ == '__main__':
    main()

