# Template de dissertação em LaTeX (MacTeX)

## Compilação

1. Instalar o MacTeX.
2. Abrir o Terminal nesta pasta.
3. Executar:

```bash
latexmk -xelatex main.tex
```

O template usa **Arial** quando a fonte está disponível no macOS. Caso contrário, usa TeX Gyre Heros, uma alternativa métrica semelhante.

## Estrutura

- `main.tex`: metadados, páginas preliminares e ordem dos capítulos.
- `istdissertacao.cls`: formatação geral.
- `capitulos/`: conteúdo da dissertação.
- `figuras/`: logótipos e imagens.
- `referencias.bib`: bibliografia BibLaTeX/Biber.

## Regras incorporadas

- A4 e margens de 2,5 cm.
- Arial, 10 pt, espaçamento 1,5.
- Sem cabeçalho; número de página no rodapé.
- Páginas preliminares em numeração romana e corpo principal em árabe.
- Legendas de quadros em cima e de figuras em baixo.
- Equações numeradas por capítulo.
- Capa com logótipo de 5 cm e área de imagem com altura máxima de 5 cm.
- Resumo/Abstract, palavras-chave, índices e lista de símbolos.

## Versão definitiva

Descomentar `\versaodefinitiva` em `main.tex` para remover “Documento Provisório”.
