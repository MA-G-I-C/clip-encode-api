from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from sentence_transformers import SentenceTransformer

port = 8000

model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")

app = FastAPI()


class HealthCheck(BaseModel):
    status: str = "OK"


class Request(BaseModel):
    query: str


class Response(Request):
    embedding: List[float]


@app.get("/")
def get_health() -> HealthCheck:
    return HealthCheck()


@app.post("/encode", response_model=Response)
async def get_embedding(request: Request) -> Response:
    embedding = model.encode(request.query) 
    response = Response(query=request.query, embedding=embedding)
    return response


def main() -> None:
    uvicorn.run("main:app", port=port, reload=True)


if __name__ == "__main__":
    main()