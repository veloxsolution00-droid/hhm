from app.ingestion.extractor import extract
import inspect
source = inspect.getsource(extract)
print(source)