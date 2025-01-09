import uvicorn
from simple_ai.server import app as v1_app
from fastapi import APIRouter, FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, FileResponse


router = APIRouter()

version_prefix = "/v1"
sai_app = FastAPI()
sai_app.mount(version_prefix, v1_app)


def serve_app(app=sai_app, host="0.0.0.0", port=10999):
    uvicorn.run(app=app, host=host, port=port)


if __name__ == "__main__":
    serve_app()
