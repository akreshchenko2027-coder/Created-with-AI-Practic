from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.detector import detect, DetectionError

app = FastAPI(title="YOLOv8 Local Inference Service")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25)
):
    if not (0.0 <= conf_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="Поріг confidence повинен бути від 0.0 до 1.0")

    try:
        image_bytes = await file.read()
        result = detect(image_bytes, conf_threshold=conf_threshold)
        return JSONResponse(content=result)
    except DetectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка сервера: {str(e)}")