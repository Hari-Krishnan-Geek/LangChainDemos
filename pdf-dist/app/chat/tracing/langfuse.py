from langfuse.client import Langfuse
import os

langfuse = Langfuse(
    os.environ["LANGFUSE_PUBLIC_KEY"],
    os.environ["LANGFUSE_SECRET_KEY"],
    host = ""
)