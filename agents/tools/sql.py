import sqlite3
from langchain.tools import Tool

from pydantic.v1 import BaseModel
from typing import List

conn = sqlite3.connect("db.sqlite")


def list_tables():
    c = conn.cursor()
    c.execute("SELECT name from sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)

def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occured: {str(err)}"

# forces langchain to use the function arguments that we specify instead of renaming it to _arg1, and so
class RunQueryArgsSchema(BaseModel):
    query: str

# for one argument Tool can be used, StructuredTool should be used if multiple arguments,
run_query_tool = Tool.from_function(
        name = "run_sqlite_query",
        description = "Run a sqlite query",
        func = run_sqlite_query,
        args_schema = RunQueryArgsSchema # argument for run_sqlite_query
    )

def describe_tables(table_names):
    c = conn.cursor()
    tables = ', '.join("'"+ table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables})")
    return '\n'.join(row[0] for row in rows if row[0] is not None)

# forces langchain to use the function arguments that we specify instead of renaming it to _arg1, and so
class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]

describe_tables_tool =  Tool.from_function(
        name = "describe_tables",
        description = "Give a list of table names, return the schema of the table",
        func = describe_tables,
        args_schema = DescribeTablesArgsSchema  # argument for describe_tables
    )