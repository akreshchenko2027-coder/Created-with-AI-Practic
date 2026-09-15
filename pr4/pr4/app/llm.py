import os
import time
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, RateLimitError, AuthenticationError
from app.schema import output_schema, validate

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30.0"))
TOKEN_BUDGET = int(os.getenv("LLM_TOKEN_BUDGET", "3000"))

SYSTEM_PROMPT = """Ти — ввічливий помічник служби підтримки інтернет-магазину.
Твоя задача: допомагати клієнтам, спираючись ВИКЛЮЧНО на надані правила магазину.

ОБМЕЖЕННЯ:
1. Якщо відповіді немає в правилах, скажи, що не знаєш, і запропонуй перевести на оператора (transfer_to_operator: true). Не вигадуй інформацію.
2. Якщо запит неоднозначний, постав уточнювальне запитання (needs_clarification: true).
3. Якщо клієнт називає номер замовлення (дві літери і шість цифр), обов'язково збережи його в поле order_number.
4. Поле topic має приймати ТІЛЬКИ одне з цих значень: "order_status", "return_exchange", "delivery_issue", "general_question", "other".
5. Відповідай українською мовою.

ФОРМАТ ВІДПОВІДІ:
Ти повинен повернути суворий JSON, який відповідає переданій схемі.

ПРИКЛАДИ:
Клієнт: "Де моє замовлення AB123456?"
Відповідь: {"reply": "Ваше замовлення AB123456 зараз комплектується.", "topic": "order_status", "based_on_rules": true, "needs_clarification": false, "transfer_to_operator": false, "order_number": "AB123456"}

Клієнт: "Як вирощувати помідори?"
Відповідь: {"reply": "На жаль, я можу допомогти лише з питаннями щодо нашого магазину. Бажаєте, я з'єднаю вас з оператором?", "topic": "other", "based_on_rules": false, "needs_clarification": false, "transfer_to_operator": true, "order_number": null}
"""

class LLMError(Exception):
    pass

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)
    return _client

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / 2.5)

def fit_budget(history: list[dict], budget: int) -> list[dict]:
    fitted_history = []
    current_tokens = 0
    
    for msg in reversed(history):
        msg_text = msg.get("content", "")
        if isinstance(msg_text, str):
            msg_tokens = estimate_tokens(msg_text)
            if current_tokens + msg_tokens <= budget:
                fitted_history.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
                
    return fitted_history

def build_messages(message: str, history: list[dict], context: str) -> list[dict]:
    messages = []
    
    full_system_prompt = f"{SYSTEM_PROMPT}\n\nПРАВИЛА МАГАЗИНУ:\n{context}"
    messages.append({"role": "system", "content": full_system_prompt})
    
    system_tokens = estimate_tokens(full_system_prompt)
    message_tokens = estimate_tokens(message)
    reserve = MAX_TOKENS
    available_budget_for_history = TOKEN_BUDGET - system_tokens - message_tokens - reserve
    
    if available_budget_for_history > 0 and history:
        fitted_history = fit_budget(history, available_budget_for_history)
        messages.extend(fitted_history)
        
    messages.append({"role": "user", "content": message})
    
    return messages

def ask(message: str, history: list[dict], context: str) -> dict:
    client = get_client()
    messages = build_messages(message, history, context)
    
    max_retries = 2
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content
            usage = response.usage
            
            cleaned_content = raw_content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()

            validated_result = validate(cleaned_content)
            
            validated_result = validate(raw_content)
            
            elapsed = time.time() - start_time
            
            return {
                "result": validated_result,
                "model": response.model,
                "elapsed": round(elapsed, 2),
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0
                }
            }
            
        except (ValueError, TypeError) as e:
            if attempt < max_retries - 1:
                messages.append({"role": "assistant", "content": raw_content})
                messages.append({"role": "user", "content": f"Помилка валідації JSON: {str(e)}. Виправ структуру та поверни правильний JSON."})
                continue
            else:
                raise LLMError(f"Модель не змогла згенерувати валідний JSON після {max_retries} спроб. Остання помилка: {str(e)}")
                
        except AuthenticationError:
            raise LLMError("Помилка авторизації: невірний API ключ.")
        except RateLimitError:
            raise LLMError("Перевищено ліміт запитів до API (Rate Limit).")
        except APITimeoutError:
            raise LLMError("Таймаут: сервіс не відповів вчасно.")
        except APIError as e:
            raise LLMError(f"Помилка сервісу моделі: {str(e)}")
        except Exception as e:
            raise LLMError(f"Невідома помилка: {str(e)}")