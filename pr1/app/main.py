from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.weather import get_weather, CityNotFoundError, ServiceUnavailableError, WeatherError

app = FastAPI(title="Погода — ПР1")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"weather": None, "error": None}
    )

@app.post("/", response_class=HTMLResponse)
def get_weather_ui(request: Request, city: str = Form(...)):
    weather_data, error_message = None, None
    try:
        weather_data = get_weather(city)
    except CityNotFoundError as e:
        error_message = str(e)
    except ServiceUnavailableError as e:
        error_message = f"Сервіс недоступний: {e}"
    except WeatherError as e:
        error_message = f"Помилка: {e}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"weather": weather_data, "error": error_message, "city": city}
    )

@app.get("/api/weather")
def api_weather(city: str):
    try:
        return get_weather(city)
    except CityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except WeatherError as e:
        raise HTTPException(status_code=400, detail=str(e))