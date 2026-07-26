%% Springer Nature sn-jnl compatibility class (Cluster Computing).
%% API-compatible with the Springer Nature LaTeX template v3.1.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sn-jnl}[2024/12/01 Enhanced sn-jnl compatibility wrapper]

\newif\if@iicol
\newif\if@pdflatex
\DeclareOption{iicol}{\@iicoltrue}
\DeclareOption{pdflatex}{\@pdflatextrue}
\DeclareOption{sn-mathphys-num}{\def\sn@bibstyle{sn-mathphys-num}}
\DeclareOption{sn-mathphys-ay}{\def\sn@bibstyle{sn-mathphys-ay}}
\DeclareOption{sn-basic}{\def\sn@bibstyle{sn-basic}}
\DeclareOption{sn-nature}{\def\sn@bibstyle{sn-nature}}
\DeclareOption{sn-aps}{\def\sn@bibstyle{sn-aps}}
\DeclareOption{sn-vancouver-num}{\def\sn@bibstyle{sn-vancouver-num}}
\DeclareOption{sn-vancouver-ay}{\def\sn@bibstyle{sn-vancouver-ay}}
\DeclareOption{sn-apa}{\def\sn@bibstyle{sn-apa}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\def\sn@bibstyle{plain}

\if@iicol
  \LoadClass[10pt,twocolumn]{article}
\else
  \LoadClass[10pt]{article}
\fi

\if@iicol
  \RequirePackage[
    left=1.5cm,
    right=1.5cm,
    top=2.25cm,
    bottom=2.25cm,
    columnsep=0.708cm,
    headheight=12pt,
    footskip=28pt
  ]{geometry}
\else
  \RequirePackage[margin=2cm]{geometry}
\fi

\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage[numbers,sort&compress]{natbib}

% Float placement tuned for two-column journal layout.
\setcounter{topnumber}{3}
\setcounter{dbltopnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\renewcommand{\topfraction}{0.90}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dbltopfraction}{0.90}
\renewcommand{\dblfloatpagefraction}{0.75}

\newcommand{\fnm}[1]{#1}
\newcommand{\sur}[1]{~#1}
\newcommand{\spfx}[1]{}
\newcommand{\orgdiv}[1]{#1}
\newcommand{\orgname}[1]{\textit{#1}}
\newcommand{\orgaddress}[1]{#1}
\newcommand{\street}[1]{#1}
\newcommand{\city}[1]{#1}
\newcommand{\postcode}[1]{#1}
\newcommand{\state}[1]{#1}
\newcommand{\country}[1]{#1}
\newcommand{\phone}[1]{#1}
\newcommand{\fax}[1]{#1}

\def\sn@authors{}
\def\sn@affils{}
\def\sn@email{}
\def\@abstract{}
\def\@keywords{}

\newcommand{\email}[1]{\gdef\sn@email{#1}}

\newcommand{\sn@addauthor}[1]{%
  \ifx\sn@authors\empty
    \gdef\sn@authors{#1}%
  \else
    \g@addto@macro\sn@authors{, #1}%
  \fi
}

\newcommand{\affil}[2][]{%
  \g@addto@macro\sn@affils{\par\small #2\par\vskip0.2em}%
}

\renewcommand{\abstract}[1]{\long\gdef\@abstract{#1}}
\providecommand{\keywords}[1]{\gdef\@keywords{#1}}
\renewcommand{\keywords}[1]{\gdef\@keywords{#1}}

\RequirePackage{hyperref}
\hypersetup{colorlinks=false,pdfborder={0 0 0}}

\newcommand{\sn@authorplain}[2][]{\sn@addauthor{#2}}
\newcommand{\sn@authorstar}[2][]{\sn@addauthor{\textbf{#2}}}
\newcommand{\sn@author@star}{\@ifnextchar[{\sn@authorstar}{\sn@authorstar[]}}
\renewcommand{\author}{\@ifstar\sn@author@star\sn@authorplain}

\renewcommand{\maketitle}{%
  \twocolumn[%
    {\LARGE\bfseries\@title\par\vskip0.75em}%
    {\large\sn@authors\par\vskip0.35em}%
    {\small\sn@affils\par\vskip1.0em}%
    {\noindent\textbf{Abstract}\par\vskip0.2em}%
    {\small\@abstract\par\vskip0.65em}%
    {\noindent\textbf{Keywords:} \@keywords\par\vskip1.0em}%
  ]%
}

\endinput
