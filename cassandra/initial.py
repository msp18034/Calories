from cassandra.cluster import Cluster
cluster = Cluster(['G401','G402']) # 随意写两个就能找到整个集群
session = cluster.connect()

# 查看所有的keyspace
session.set_keyspace("system_schema")
rows = session.execute("SELECT * FROM system_schema.keyspaces")
for row in rows:
     print(row)
# 更改授权
session.execute("ALTER KEYSPACE system_auth  WITH REPLICATION = {'class' : 'NetworkTopologyStrategy', 'datacenter1' : 3}")
# 在命令行进行 bin/nodetool repair system_auth

# 创建keyspace fooddiary（使用的时候都是小写！）
session.execute("CREATE KEYSPACE IF NOT EXISTS foodDiary WITH REPLICATION = { 'class' :'NetworkTopologyStrategy', 'datacenter1' : 3}")
session.execute("use foodDiary")
# session = cluster.connect("fooddiary")


"""
KEYSPACE: foodDiary
Users 用户昵称，密码，身高（m）,体重（kg）,年龄，性别(M/F) 
		|    id   | password| height | weight|   age   |  sex  |
		| VARCHAR | VARCHAR | FLOAT  | FLOAT | TINYINT |VARCHAR|

Records 用户昵称，时间，图片，食物种类，热量cal，碳水化合物g，脂肪g，蛋白质g，纤维素g
		|   id   |   time  | photo|    food   |   calorie  |   carbo    |     fat    |   protein  |    fiber   |
		|VARCHAR |TIMESTAMP| TEXT | LIST<TEXT>| LIST<FLOAT>| LIST<FLOAT>| LIST<FLOAT>| LIST<FLOAT>| LIST<FLOAT>|

"""
# 创建user表
session.execute("CREATE TABLE IF NOT EXISTS users (id VARCHAR, password VARCHAR, height FLOAT, weight FLOAT, age TINYINT, sex VARCHAR, PRIMARY KEY ( id ) )")
# 创建records表
session.execute("CREATE TABLE IF NOT EXISTS records (id VARCHAR, time TIMESTAMP, photo TEXT, food LIST<TEXT>, calorie LIST<FLOAT>, "\
				"carbo LIST<FLOAT>, fat LIST<FLOAT>, protein LIST<FLOAT>, fiber LIST<FLOAT>, PRIMARY KEY ( id, time ) )")

# 插入数据
session.execute("INSERT INTO users (id, password) VALUES ('test', 'test')")

from datetime import datetime, timedelta
t = datetime.now()
id = 'test'
food = ['food1','food2']
calorie = [1.23, 45.23]
session.execute(
    "INSERT INTO records (id, time, food, calorie)VALUES (%s, %s, %s, %s)",
    (id, t, food, calorie)
)


# 输出所有数据
rows = session.execute('SELECT * FROM users')
for row in rows:
    print(row)

rows = session.execute("SELECT * FROM records")
for row in rows:
    print(row)


# 输出特定用户特定时间的数据
query = "select * from records where id = %s AND time <= %s "
rows = session.execute(query, (id,t))
