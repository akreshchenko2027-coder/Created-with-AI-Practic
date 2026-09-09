import os
import time
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIError, AuthenticationError, RateLimitError

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "gemini-3.8-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30.0"))

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def load_context() -> str:
    """Завантажує вміст файлу context.md."""
    context_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "context.md")
    if os.path.exists(context_path):
        with open(context_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Правила магазину відсутні."


def ask(question: str, context: str = None) -> str:
    """Головна функція для обробки запиту користувача."""
    if context is None:
        context = load_context()

    system_instruction = (
        "Ви — ввічливий та чіткий помічник служби підтримки інтернет-магазину.\n"
        "Правила роботи:\n"
        "1. Відповідайте на запитання виключно на основі наданого Контексту (Правил магазину).\n"
        "2. Якщо відповіді немає в Контексті, прямо відповідайте: 'Вибачте, у мене немає інформації щодо цього питання у правилах магазину.' Не вигадуйте факти!\n"
        "3. Якщо звернення неоднозначне або нечітке, попросіть уточнити деталі.\n"
        "4. Відповідайте зрозумілою та дружньою мовою."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "system", "content": f"ДОВІДКОВИЙ КОНТЕКСТ (ПРАВИЛА МАГАЗИНУ):\n{context}"},
        {"role": "user", "content": question}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=TIMEOUT
        )
        return response.choices[0].message.content or "Порожня відповідь від моделі."

    except RateLimitError:
        raise RuntimeError("Перевищено ліміт запитів до API (429). Спробуйте пізніше.")
    except AuthenticationError:
        raise RuntimeError("Помилка автентифікації API-ключа (401/403). Перевірте .env.")
    except APIConnectionError:
        raise RuntimeError("Таймаут або помилка мережевого з'єднання з API.")
    except APIError as e:
        raise RuntimeError(f"Помилка API OpenAI/Gemini: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Помилка при виконанні запиту: {str(e)}")