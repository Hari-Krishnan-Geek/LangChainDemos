from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel


def write_report(filename, html):
    with open(filename,  'w') as f:
        f.write(html)

# forces langchain to use the function arguments that we specify instead of renaming it to _arg1, and so
class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str


# StructuredTool should be used if multiple arguments, for one argument Tool can be used
write_report_tool = StructuredTool.from_function(
        name = "write_report",
        description = "Write an html file to disk. Use this toll whenever someone asks for a report",
        func = write_report,
        args_schema = WriteReportArgsSchema # argument for write_report
    )