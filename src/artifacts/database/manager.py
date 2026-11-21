import sqlite3
from pathlib import Path
from typing import Dict, Any
from contextlib import closing

class CancerRepository:

    def __init__(self) -> None:
        self._db_location = Path(__file__).resolve().parents[0]
        self._cancer_db = self._db_location.joinpath("cancer.db")
        if not self._cancer_db.exists():
            self._initialize_db()
        else:
            self._validate_connection()

    def _initialize_db(self) -> None:
        print("Creating database...")
        with closing(sqlite3.connect(self._cancer_db)) as con:
            with con:
                cursor = con.cursor()
                with self._db_location.joinpath("schema.sql").open() as schema:
                    sql_statement = schema.read()
                cursor.executescript(sql_statement)

    def _validate_connection(self) -> None:
        with closing(sqlite3.connect(self._cancer_db)) as con:
            with con:
                cursor = con.cursor()
                tables = ["observation", "prediction"]
                for table in tables:
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                        (table,)
                    )
                    if cursor.fetchone() is None:
                        raise RuntimeError(f"Missing database table: {table}")
        print("Database already exists")

    def insert(self, tumor_data:Dict[str, Any], prediction:Dict[str, Any], model_info:Dict[str, Any]) -> None:

        columns = list(tumor_data.keys())
        values = list(tumor_data.values())
        n_cols = 31 * "?," + "?"
        observation = f"INSERT INTO observation ({",".join(columns)}) VALUES ({n_cols});"

        with closing(sqlite3.connect(self._cancer_db)) as con:
            with con:
                cursor = con.cursor()
                cursor.execute(observation, values)
                observation_id = cursor.lastrowid

        new_prediction = prediction.copy()
        new_prediction["observation_id"] = observation_id
        new_prediction["model_name"] = model_info["name"]
        new_prediction["model_version"] = model_info["version"]
        del new_prediction["patient_id"]

        columns = ",".join(list(new_prediction.keys()))
        values = tuple(new_prediction.values())
        n_cols = (len(values) - 1) * "?," + "?"
        new_prediction = f"INSERT INTO prediction ({columns}) VALUES ({n_cols});"

        with closing(sqlite3.connect(self._cancer_db)) as con:
            with con:
                cursor = con.cursor()
                cursor.execute(new_prediction, values)

    def select(self, sql):

        with closing(sqlite3.connect(self._cancer_db)) as con:
            with con:
                cursor = con.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
        return rows

if __name__ == "__main__":

    db = CancerRepository()
    rows1 = db.select("SELECT * FROM prediction;")
    for row in rows1:
        print(row)
    rows2 = db.select("SELECT * FROM observation;")
    for row in rows2:
        print(row)
