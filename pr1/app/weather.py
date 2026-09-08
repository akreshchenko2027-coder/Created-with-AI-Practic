import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 3.0

class WeatherError(Exception):
    pass

class CityNotFoundError(WeatherError):
    pass

class ServiceUnavailableError(WeatherError):
    pass

def get_weather(city_name: str) -> dict:
    city_clean = city_name.strip()
    if not city_clean:
        raise CityNotFoundError("Назва міста не може бути порожньою.")

    # 1. Пошук координат
    geo_params = {"name": city_clean, "count": 1, "language": "uk", "format": "json"}
    try:
        geo_response = requests.get(GEOCODING_URL, params=geo_params, timeout=TIMEOUT)
    except requests.exceptions.Timeout:
        raise ServiceUnavailableError("Час очікування відповіді від геосервісу вичерпано.")
    except requests.exceptions.RequestException as e:
        raise ServiceUnavailableError(f"Помилка мережі: {e}")

    if geo_response.status_code >= 500:
        raise ServiceUnavailableError("Сервіс геокодування тимчасово недоступний.")
    elif geo_response.status_code >= 400:
        raise WeatherError("Помилка запиту до геосервісу.")

    try:
        geo_data = geo_response.json()
    except ValueError:
        raise ServiceUnavailableError("Некоректний JSON від геосервісу.")

    if "results" not in geo_data or not geo_data["results"]:
        raise CityNotFoundError(f"Місто '{city_clean}' не знайдено.")

    city_info = geo_data["results"][0]
    lat, lon = city_info.get("latitude"), city_info.get("longitude")
    display_name = city_info.get("name", city_clean)

    # 2. Отримання погоди
    forecast_params = {"latitude": lat, "longitude": lon, "current_weather": True}
    try:
        forecast_response = requests.get(FORECAST_URL, params=forecast_params, timeout=TIMEOUT)
    except requests.exceptions.Timeout:
        raise ServiceUnavailableError("Час очікування відповіді від сервісу погоди вичерпано.")
    except requests.exceptions.RequestException as e:
        raise ServiceUnavailableError(f"Помилка мережі: {e}")

    if forecast_response.status_code >= 500:
        raise ServiceUnavailableError("Сервіс погоди тимчасово недоступний.")
    elif forecast_response.status_code >= 400:
        raise WeatherError("Помилка запиту до сервісу погоди.")

    try:
        forecast_data = forecast_response.json()
    except ValueError:
        raise ServiceUnavailableError("Некоректний JSON від сервісу погоди.")

    if "current_weather" not in forecast_data:
        raise ServiceUnavailableError("Відсутні дані про погоду у відповіді API.")

    current = forecast_data["current_weather"]
    return {
        "city": display_name,
        "temperature": current.get("temperature"),
        "windspeed": current.get("windspeed")
    }