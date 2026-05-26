"""Cuenta palabras del cuerpo de main.tex, excluyendo:
- comentarios LaTeX
- bibliografia (\\bibliography y posterior)
- notas al pie (\\footnote)
- entornos matematicos (equation, align, $..$, \\[..\\])
- contenido de tablas (entornos tabular*)
- comandos LaTeX (sin contar nombres de comandos como palabras)
"""
from __future__ import annotations
import re
from pathlib import Path

MAIN = Path("entrega/tesis/main.tex")


def strip_cmd_with_arg(text: str, cmd: str) -> str:
    """Elimina ocurrencias de \\cmd{...} respetando llaves anidadas."""
    pattern = re.compile(r"\\" + re.escape(cmd) + r"\s*\{")
    out, i = [], 0
    while i < len(text):
        m = pattern.match(text, i)
        if m:
            i = m.end()
            depth = 1
            while i < len(text) and depth > 0:
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def main() -> None:
    text = MAIN.read_text(encoding="utf-8")

    # 1) Quitar comentarios
    text = re.sub(r"(?<!\\)%.*", "", text)

    # 2) Cortar desde \bibliography{ en adelante
    text = re.sub(r"\\bibliography\{.*", "", text, flags=re.DOTALL)

    # 3) Quitar comandos con un argumento que no aporta texto al cuerpo
    for cmd in ("footnote", "label", "cite", "citep", "citet", "ref",
                "eqref", "autoref", "input"):
        text = strip_cmd_with_arg(text, cmd)

    # 4) Quitar entornos matematicos
    for env in ("equation", "align", "eqnarray", "displaymath", "gather"):
        text = re.sub(
            r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}",
            " ", text, flags=re.DOTALL,
        )
    # display math \[..\] y inline \(..\)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.DOTALL)
    # math $..$ (no contamos formulas como palabras)
    text = re.sub(r"\$[^$]*\$", " ", text)

    # 5) Quitar tablas (tabular*) y figuras (figure)
    for env in ("tabular", "tabularx", "longtable"):
        text = re.sub(
            r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}",
            " ", text, flags=re.DOTALL,
        )

    # 6) Quitar el resto de comandos LaTeX (con argumento opcional en [])
    text = re.sub(r"\\[a-zA-Z@]+\*?\s*(?:\[[^\]]*\])?", " ", text)

    # 7) Quitar llaves restantes y tildes no-rompibles
    text = text.replace("~", " ").replace("\\\\", " ")
    text = re.sub(r"[{}]", " ", text)

    # 8) Tokenizar palabras (alfabeticas, soporta acentos espanoles)
    words = re.findall(
        r"[A-Za-zÀ-ſ][A-Za-zÀ-ſ'\-]*", text
    )
    print(f"WORDS_BODY={len(words)}")


if __name__ == "__main__":
    main()
