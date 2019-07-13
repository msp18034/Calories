"""
根据FoodList.txt中食物的名字，在薄荷食物库中搜索，获取营养信息
由于中餐名字多样复杂，搜索结果第一条并不一定就是需要的数据
还需人工进一步对生成的数据进行审核和修改
本库中的nutrition.csv是人工修正完毕的
"""
import requests
import re
import csv

class Spider(object):
    def __init__(self):
        self.search_url = 'http://www.boohee.com/food/search?keyword='
        self.query_url = 'http://www.boohee.com/shiwu/'
        self.headers = {
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'
        }
        self.encoding = 'utf-8'

    def get_id(self, name):
        try:
            print((self.search_url + name))
            r = requests.get(self.search_url + name, headers = self.headers)
            r.raise_for_status()
            if self.encoding == None:
                r.encoding = r.apparent_encoding
            else:
                r.encoding = self.encoding
            res = re.findall(r'<div id="main">(.*?)</li>', r.text, re.S)
            if len(res) > 0:
                res = re.findall(r'/shiwu/(.*?)\'', res[0], re.S)
                if len(res) > 0:
                    return res[0]
        except Exception as e:
            return None

    def get_query(self, id):
        name, heat, carbo, fat, protein, fibrinous = ('','','','','','')
        try:
            r = requests.get(self.query_url + id, headers = self.headers)
            r.raise_for_status()
            if self.encoding == None:
                r.encoding = r.apparent_encoding
            else:
                r.encoding = self.encoding
            res = re.findall(r'<h3>(.*?)的热量和减肥功效</h3>', r.text, re.S)
            if len(res) > 0:
                name = res[0].strip()
            res = re.findall(r'热量\(大卡\)</span><span class="dd"><span class="stress red1">(.*?)</span>', r.text, re.S)
            if len(res) > 0:
                heat = res[0].strip()
            res = re.findall(r'碳水化合物\(克\)</span><span class="dd">(.*?)</span>', r.text, re.S)
            if len(res) > 0:
                carbo = res[0].strip()
            res = re.findall(r'脂肪\(克\)</span><span class="dd">(.*?)</span>', r.text, re.S)
            if len(res) > 0:
                fat = res[0].strip()
            res = re.findall(r'蛋白质\(克\)</span><span class="dd">(.*?)</span>', r.text, re.S)
            if len(res) > 0:
                protein = res[0].strip()
            res = re.findall(r'纤维素\(克\)</span><span class="dd">(.*?)</span>', r.text, re.S)
            if len(res) > 0:
                fibrinous = res[0].strip()
                if fibrinous == '一':
                    fibrinous = 0
            return name, heat, carbo, fat, protein, fibrinous
        except Exception as e:
            return None

    def get_data(self, name):
        print("get_data name:", name)
        r = self.get_id(name)
        if r:
            r = self.get_query(r)
            return r
        else:
            return None


if __name__ == '__main__':
    s = Spider()
    food_path = "172FoodList.txt"

    with open(food_path, encoding='utf-8') as f:
        food_names = f.readlines()
    foods = [c.strip() for c in food_names]

    with open("nutrition.csv", 'w', newline='') as bb:
        writer = csv.writer(bb)

        for i in range(len(foods)):
            name, heat, carbo, fat, protein, fibrinous = s.get_data(foods[i])
            #info: (序号, 食物名，返回的食物名，热量，碳水化合物，脂肪，蛋白质，纤维素)
            info = (i+1, foods[i], name, heat, carbo, fat, protein, fibrinous)
            writer.writerow(info)
            print(info)
