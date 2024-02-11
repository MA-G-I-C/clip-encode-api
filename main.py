from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")

app = FastAPI()


class HealthCheck(BaseModel):
    status: str = "OK"


class Request(BaseModel):
    query: str


class Response(Request):
    embedding: str


@app.get("/")
def get_health() -> HealthCheck:
    return HealthCheck()


@app.post("/encode", response_model=Response)
async def get_embedding(request: Request) -> Response:
    embedding = model.encode(request.query) 
    response = Response(query=request.query, embedding=embedding)
    return response


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

if __name__ == "__main__":
    main()