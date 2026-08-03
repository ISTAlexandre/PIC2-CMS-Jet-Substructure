\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{istdissertacao}[2026/07/15 Template de dissertacao baseado no modelo fornecido]
\LoadClass[10pt,a4paper,oneside]{book}

\RequirePackage{iftex}
\ifPDFTeX
  \ClassError{istdissertacao}{Compile com XeLaTeX ou LuaLaTeX}{No MacTeX use: latexmk -xelatex main.tex}
\fi
\RequirePackage{fontspec}
\IfFileExists{/System/Library/Fonts/Supplemental/Arial.ttf}{%
\setmainfont[Path=/System/Library/Fonts/Supplemental/,
  BoldFont={Arial Bold.ttf},
  ItalicFont={Arial Italic.ttf},
  BoldItalicFont={Arial Bold Italic.ttf}]{Arial.ttf}
\setsansfont[Path=/System/Library/Fonts/Supplemental/,
  BoldFont={Arial Bold.ttf},
  ItalicFont={Arial Italic.ttf},
  BoldItalicFont={Arial Bold Italic.ttf}]{Arial.ttf}
}{%
  \IfFileExists{/Library/Fonts/Arial.ttf}{%
    \setmainfont[Path=/Library/Fonts/]{Arial.ttf}%
    \setsansfont[Path=/Library/Fonts/]{Arial.ttf}%
  }{%
    \setmainfont{Noto Sans}\setsansfont{Noto Sans}%
  }%
}
\renewcommand{\familydefault}{\sfdefault}

\RequirePackage[portuguese,english]{babel}
\RequirePackage[a4paper,top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm]{geometry}
\RequirePackage{setspace}
\onehalfspacing
\RequirePackage{microtype}
\RequirePackage{graphicx}
\RequirePackage{etoolbox}
\RequirePackage{amsmath,amssymb}
\RequirePackage{booktabs,tabularx,array}
\RequirePackage{textcomp}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{titlesec}
\RequirePackage{tocloft}
\RequirePackage{enumitem}
\RequirePackage{csquotes}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[backend=biber,style=numeric-comp,sorting=none,maxcitenames=2,maxbibnames=99]{biblatex}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55\baselineskip}
\setlength{\footskip}{1.25cm}
\pagestyle{plain}

\titleformat{\chapter}[hang]
  {\normalfont\bfseries\fontsize{14}{17}\selectfont}
  {\thechapter}{0.8em}{\MakeUppercase}
\titlespacing*{\chapter}{0pt}{0pt}{1.2\baselineskip}
\titleformat{\section}{\normalfont\bfseries\fontsize{11}{13}\selectfont}{\thesection}{0.65em}{}
\titleformat{\subsection}{\normalfont\bfseries\fontsize{10}{12}\selectfont}{\thesubsection}{0.65em}{}
\titleformat{\subsubsection}{\normalfont\bfseries\fontsize{10}{12}\selectfont}{\thesubsubsection}{0.65em}{}

\captionsetup{font=small,labelfont=bf,labelsep=endash}
\captionsetup[table]{position=top,skip=5pt,name=Quadro}
\captionsetup[figure]{position=bottom,skip=5pt,name=Figura}
\numberwithin{equation}{chapter}

\renewcommand{\contentsname}{Índice}
\renewcommand{\listfigurename}{Lista de Figuras}
\renewcommand{\listtablename}{Lista de Quadros}
\setlength{\cftbeforechapskip}{2pt}
\setlength{\cftbeforesecskip}{0pt}

\newcommand{\ISTtitulo}{Título da Dissertação}
\newcommand{\ISTsubtitulo}{}
\newcommand{\ISTautor}{Nome completo do Candidato}
\newcommand{\ISTgrau}{Mestre em}
\newcommand{\ISTcurso}{Nome do Curso}
\newcommand{\ISTorientadores}{Título Nome Completo\\Título Nome Completo}
\newcommand{\ISTpresidente}{Título Nome Completo}
\newcommand{\ISTvogais}{Título Nome Completo\\Título Nome Completo}
\newcommand{\ISTdata}{Mês e ano}
\newcommand{\ISTimagem}{}
\newcommand{\ISTacademialogo}{}
\newif\ifISTprovisorio
\ISTprovisoriotrue

\newcommand{\titulodissertação}[1]{\renewcommand{\ISTtitulo}{#1}}
\newcommand{\subtitulodissertação}[1]{\renewcommand{\ISTsubtitulo}{#1}}
\newcommand{\autordissertação}[1]{\renewcommand{\ISTautor}{#1}}
\newcommand{\graudissertação}[1]{\renewcommand{\ISTgrau}{#1}}
\newcommand{\cursodissertação}[1]{\renewcommand{\ISTcurso}{#1}}
\newcommand{\orientadoresdissertação}[1]{\renewcommand{\ISTorientadores}{#1}}
\newcommand{\presidentejuri}[1]{\renewcommand{\ISTpresidente}{#1}}
\newcommand{\vogaisjuri}[1]{\renewcommand{\ISTvogais}{#1}}
\newcommand{\datadissertação}[1]{\renewcommand{\ISTdata}{#1}}
\newcommand{\imagemcapa}[1]{\renewcommand{\ISTimagem}{#1}}
\newcommand{\logoacademia}[1]{\renewcommand{\ISTacademialogo}{#1}}
\newcommand{\versaodefinitiva}{\ISTprovisoriofalse}

\newcommand{\capadissertação}{%
  \begin{titlepage}
  \thispagestyle{empty}
  \begin{minipage}[t]{0.49\textwidth}
    \includegraphics[width=5cm]{figuras/logo-tecnico-lisboa.png}
  \end{minipage}%
  \begin{minipage}[t]{0.49\textwidth}\raggedleft
    \ifdefempty{\ISTacademialogo}{}{\includegraphics[height=1.6cm]{\ISTacademialogo}}
  \end{minipage}
  \vspace{0.7cm}
  \ifdefempty{\ISTimagem}{%
    \rule{0pt}{5cm}%
  }{%
    \begin{center}\includegraphics[width=\textwidth,height=5cm,keepaspectratio]{\ISTimagem}\end{center}%
  }
  \vfill
  \begin{center}
    {\bfseries\fontsize{16}{19}\selectfont \ISTtitulo\par}
    \vspace{0.55cm}
    {\bfseries\fontsize{14}{17}\selectfont \ISTsubtitulo\par}
    \vspace{0.85cm}
    {\bfseries\fontsize{14}{17}\selectfont \ISTautor\par}
    \vspace{0.75cm}
    {\fontsize{12}{14}\selectfont Dissertação para obtenção do Grau de \ISTgrau\ em\par}
    \vspace{0.35cm}
    {\bfseries\fontsize{16}{19}\selectfont \ISTcurso\par}
    \vspace{0.75cm}
    {\fontsize{12}{14}\selectfont Orientador(es)\par\ISTorientadores\par}
    \vspace{0.7cm}
    {\bfseries\fontsize{14}{17}\selectfont Júri\par}
    \vspace{0.2cm}
    {\fontsize{12}{14}\selectfont Presidente: \ISTpresidente\par
    Orientador: \ISTorientadores\par
    Vogais: \ISTvogais\par}
    \vspace{0.7cm}
    {\bfseries\fontsize{14}{17}\selectfont \ISTdata\par}
    \ifISTprovisorio\vspace{0.25cm}{\fontsize{12}{14}\selectfont (Documento Provisório)\par}\fi
  \end{center}
  \vfill
  \end{titlepage}
}

\newcommand{\declaracao}{%
  \cleardoublepage\thispagestyle{plain}
  {\bfseries\fontsize{12}{14}\selectfont Declaração\par}\vspace{0.7em}
  Declaro que o presente documento é um trabalho original da minha autoria e que cumpre todos os requisitos do Código de Conduta e Boas Práticas da Universidade de Lisboa.
  \cleardoublepage
}

\newcommand{\frontmatterIST}{\frontmatter\pagestyle{plain}}
\newcommand{\mainmatterIST}{\mainmatter\pagestyle{plain}}
