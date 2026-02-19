from app.agent.retrieval_strategies import reduce_structural_noise


def _item(content: str, *, title: str = "", heading: str = "", source_type: str = "") -> dict:
    return {
        "content": content,
        "metadata": {
            "row": {
                "source_type": source_type,
                "metadata": {
                    "title": title,
                    "heading": heading,
                },
            }
        },
    }


def test_reduce_structural_noise_prefers_intro_editorial_when_no_body():
    items = [
        _item(
            "Tabla de contenido\nIntroduccion ........ 1\nCapitulo 1 ........ 7\nCapitulo 2 ........ 20",
            title="Table of contents",
        ),
        _item(
            "Prefacio. Esta introduccion presenta el objetivo y alcance del libro.",
            title="Prefacio",
            source_type="front_matter",
        ),
        _item(
            "Prólogo. El texto introduce la metodologia del curso.",
            title="Prólogo",
            source_type="front_matter",
        ),
    ]

    filtered = reduce_structural_noise(items, "que dice la introducción?")

    assert len(filtered) <= 3
    assert (
        "prefacio" in filtered[0]["content"].lower() or "prólogo" in filtered[0]["content"].lower()
    )


def test_reduce_structural_noise_keeps_all_for_toc_query():
    items = [
        _item(
            "Tabla de contenido\nCapitulo 1 .... 7\nCapitulo 2 .... 20", title="Table of contents"
        ),
        _item("Tabla de contenido\nA.1 .... 400\nA.2 .... 410", title="Table of contents"),
    ]

    filtered = reduce_structural_noise(items, "muestrame la tabla de contenido")

    assert filtered == items
