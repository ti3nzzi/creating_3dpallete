import psycopg2
from psycopg2 import sql


class DataBaseManager:
    def __init__(self) -> None:
        DATABASE="postgres"
        USER="postgres"
        PASSWORD="postgres"
        HOST="localhost"
        PORT="5438"

        self.schema_name = "pallets"
        self.table_name = 'test_results'

        self.conn = self.__connection(DATABASE, USER, PASSWORD, HOST, PORT)
        self.__create_schema_if_not_exists()

    def __connection(self, dbname, user, password, host, port) -> None:
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        return conn
    
    def __create_schema_if_not_exists(self):

        create_schema_query = f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};"
        create_table_query = f"CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.table_name} (id SERIAL PRIMARY KEY, dim_w FLOAT, dim_h FLOAT, dim_l FLOAT, box_volume FLOAT, box_pos VARCHAR(255));"

        with self.conn, self.conn.cursor() as cur:
            cur.execute(create_schema_query)
            cur.execute(create_table_query)

    def insert_results(self, dim_w: float, dim_h: float, dim_l: float, box_volume: float, box_pos: float) -> None:

        insert_query = f"INSERT INTO {self.schema_name}.{self.table_name}(dim_w,dim_h,dim_l,box_volume,box_pos) VALUES(%s,%s,%s,%s,%s)"

        with self.conn, self.conn.cursor() as cur:
            cur.execute(insert_query, (dim_w, dim_h, dim_l, box_volume, box_pos))