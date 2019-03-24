import yaml


config = yaml.full_load(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yaml")))

ODBC_DIRECTIVES = [
    "DRIVER={ODBC Driver 17 for SQL Server}",
    "PORT={}".format(config['sql_server_port']),
    "SERVER={}".format(config['sql_server']),
    "DATABASE={}".format(config['sql_database']),
    "UID={}".format(config['sql_user']),
    "PWD={}".format(config['sql_pwd']),
]
ELASTIC_CLOUD_USER = config['elastic_cloud_user']
ELASTIC_CLOUD_PWD = config['elastic_cloud_pwd']
