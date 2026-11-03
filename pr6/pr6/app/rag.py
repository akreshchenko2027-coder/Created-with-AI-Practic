"""Конвеєр відповіді за базою знань: від питання до відповіді з джерелами.

Це єдине місце, яке знає всі кроки послідовно: знайти фрагменти
(`retrieval.retrieve`), зібрати з них контекст (`retrieval.build_context`),
спитати модель (`llm.ask`), перевірити те, що вона повернула, і скласти
результат для веб-рівня. Модулі-учасники одне про одного не знають:
пошук не знає про модель, модель — про індекс, веб-рівень — ні про що з
цього.

Якщо придатних фрагментів немає, модель не викликається. Посилання
перевіряються щодо фактично переданого контексту, а користувачеві
показуються лише ті джерела, на які модель послалася.
"""

from dataclasses import dataclass, field

from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .llm import LLMError, ask
from .retrieval import Source, build_context, retrieve


@dataclass
class Answer:
    """Результат конвеєра — те, з чим працює веб-рівень.

    `text` — відповідь для клієнта. `found` — чи відповідь спирається на
    базу знань (після вашої перевірки, а не зі слів моделі). `sources` —
    джерела, які показуються під відповіддю. `retrieved` — фрагменти,
    які бачила модель, для налагодження: за ними видно, чия це помилка —
    пошуку чи генерації. `elapsed` — час за етапами, наприклад
    `{"retrieval": 0.04, "generation": 1.9}`. `usage` — токени запиту й
    відповіді, якщо модель викликалась.
    """

    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(
    question: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> Answer:
    """Знайти контекст, викликати модель і перевірити її посилання."""
    import time

    started = time.perf_counter()
    hits = retrieve(question, index, keyword_index, filters)
    context, context_sources = build_context(hits)
    retrieval_elapsed = time.perf_counter() - started
    retrieved = [
        Hit(chunk=source.chunk, score=source.score)
        for source in context_sources
    ]
    if not context_sources:
        return Answer(
            text="У наданій базі знань немає достатньої інформації, щоб відповісти на це питання.",
            found=False,
            elapsed={"retrieval": retrieval_elapsed, "generation": 0.0},
        )

    generated = ask(question, context)
    result = generated["response"]
    sources_by_ref = {source.ref: source for source in context_sources}
    citations = result["citations"]
    invalid_refs = [ref for ref in citations if ref not in sources_by_ref]
    if invalid_refs:
        raise LLMError(
            "Модель послалася на фрагмент, якого не було в контексті: "
            + ", ".join(map(str, invalid_refs))
        )
    if result["found"] and not citations:
        raise LLMError("Модель позначила відповідь знайденою, але не навела джерел")

    cited_sources = [sources_by_ref[ref] for ref in citations]
    return Answer(
        text=result["answer"].strip(),
        found=result["found"],
        sources=cited_sources,
        retrieved=retrieved,
        model=generated["model"],
        elapsed={
            "retrieval": retrieval_elapsed,
            "generation": generated["elapsed"],
        },
        usage=generated["usage"],
    )
