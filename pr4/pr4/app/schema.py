import json
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from typing import Optional

class TopicEnum(str, Enum):
    order_status = "order_status"
    return_exchange = "return_exchange"
    delivery_issue = "delivery_issue"
    general_question = "general_question"
    other = "other"

class AssistantResponse(BaseModel):
    reply: str = Field(description="Текст відповіді для клієнта. Звертатися ввічливо.")
    topic: TopicEnum = Field(description="Тема звернення з фіксованого переліку.")
    based_on_rules: bool = Field(description="Чи відповідь ґрунтується виключно на правилах магазину.")
    needs_clarification: bool = Field(description="Чи потрібно уточнити деталі у клієнта.")
    transfer_to_operator: bool = Field(description="Чи потрібно перевести розмову на живого оператора.")
    order_number: Optional[str] = Field(default=None, description="Номер замовлення, якщо клієнт його назвав у розмові. Формат: дві літери і шість цифр.")

def output_schema() -> dict:
    return AssistantResponse.model_json_schema()

def validate(raw: str) -> dict:
    try:
        parsed_data = json.loads(raw)
        validated_data = AssistantResponse(**parsed_data)
        return validated_data.model_dump()
    except json.JSONDecodeError as e:
        raise ValueError(f"Відповідь не є коректним JSON: {str(e)}")
    except ValidationError as e:
        raise ValueError(f"JSON не відповідає схемі: {str(e)}")