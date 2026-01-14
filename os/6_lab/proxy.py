import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
import requests

CPP_SERVER = "http://localhost:8080"

app = FastAPI()

# ===== Фикс CORS  =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_FILE = os.path.join(BASE_DIR, "site.html")

# ===== Главная страница =====
@app.get("/", response_class=FileResponse)
def root():
    return "site.html"

# ===== Прокси запросы к C++ серверу ===== (Надо для обхода CORS политики браузера) 
# PS.хак для обхода политики браузера, т.к. C++ сервер надо дополнительно настраивать для CORS
def proxy_request(path: str, request: Request):
    url = f"{CPP_SERVER}{path}"

    try:
        r = requests.request(
            method=request.method,
            url=url,
            params=dict(request.query_params),
            headers={"Content-Type": "application/json"},
            timeout=5
        )
    except requests.RequestException as e:
        return Response(
            content=f'{{"error":"{str(e)}"}}',
            status_code=502,
            media_type="application/json"
        )

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("Content-Type", "application/json")
    )


# ===== /current =====
@app.api_route("/current", methods=["GET", "OPTIONS"])
async def current(request: Request):
    return proxy_request("/current", request)


# ===== /stats =====
@app.api_route("/stats", methods=["GET", "OPTIONS"])
async def stats(request: Request):
    return proxy_request("/stats", request)
