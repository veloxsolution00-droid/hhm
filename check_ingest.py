from app.ingestion.extractor import ingest
import inspect
source = inspect.getsource(ingest)
print(source)