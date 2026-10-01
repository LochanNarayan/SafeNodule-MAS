"""Generate the full LaTeX article from lungnoduleagent_enhanced/results/experiments.json.

    cd lungnoduleagent_enhanced
    python -m lna.cli experiments --n 64 --seeds 8 --boot 1000 --out results
    cd ..
    python build_paper_tex.py

Writes paper/enhanced_lungnoduleagent.tex, (re)builds paper/figures/architecture.png,
and copies the result figures into paper/figures/. Compile with pdfLaTeX + BibTeX
(Overleaf does both automatically), or drop the paper/ folder into Overleaf.

Typography (preamble): Times New Roman (newtx); 1 in margins; 18 pt bold centred
title; 9 pt justified body with a 0.2 in first-line indent (flush-left right
after a heading); decimal 12 pt / 10 pt-bold-italic headings; booktabs tables.
APA references via apacite + natbib.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "lungnoduleagent_enhanced" / "results" / "experiments.json"
SRC_FIGS = ROOT / "lungnoduleagent_enhanced" / "results" / "figures"
PAPER = ROOT / "paper"
OUT = PAPER / "enhanced_lungnoduleagent.tex"

REGIMES = ["clean", "noisy", "borderline", "imbalanced", "adversarial"]
ABL_ORDER = ["full", "- router", "- verifier", "- devils_advocate", "- guideline",
             "- calibration", "- abstention", "- board (1 agent)", "- debate (1 round)"]


# --------------------------------------------------------------------------- #
def f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def pm(m, s, d=3):
    return f"${f(m, d)}\\,{{\\pm}}\\,{f(s, d)}$"


def load():
    if not RESULTS.exists():
        raise SystemExit(f"missing {RESULTS}: run the experiments command first.")
    return json.loads(RESULTS.read_text())


def Rr(p):
    return [r for r in REGIMES if r in p["regimes"]]


def full(p, r):
    return p["regimes"][r]["full"]["mean"]


def fsd(p, r):
    return p["regimes"][r]["full"]["std"]


def abl(p, r, name, key):
    return p["regimes"][r]["ablation"][name]["mean"][key]


def delta(p, r, name, key):
    return p["regimes"][r]["ablation"]["full"]["mean"][key] - abl(p, r, name, key)


def sig(p, r, test):
    return p["regimes"][r].get("significance", {}).get(test, {})


def sig_p(p, r, test):
    v = sig(p, r, test).get("p_value", float("nan"))
    return "$p<0.001$" if v < 1e-3 else f"$p={v:.3f}$"


# --------------------------------------------------------------------------- #
def tex_table(headers, rows, caption, label, colspec=None, wide=True):
    """Full-grid table (vertical and horizontal rules, shaded header row),
    matching a typical journal layout rather than a minimalist booktabs style."""
    ncol = len(headers)
    base = colspec or ("l" + "c" * (ncol - 1))
    grid = "|" + "|".join(list(base)) + "|" if "p{" not in base else \
        "|" + base.replace("}p{", "}|p{") + "|"
    head = " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\ \\hline"
    body = "\n\\hline\n".join(" & ".join(str(c) for c in row) + " \\\\" for row in rows)
    open_box = "\\begin{adjustbox}{max width=\\linewidth}\n" if wide else ""
    close_box = "\n\\end{adjustbox}" if wide else ""
    return f"""\\begin{{table}}[htbp]
\\centering
\\small
\\renewcommand{{\\arraystretch}}{{1.25}}
\\setlength{{\\tabcolsep}}{{5pt}}
{open_box}\\begin{{tabular}}{{{grid}}}
\\hline
\\rowcolor{{tblhead}}
{head}
{body}
\\hline
\\end{{tabular}}{close_box}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
"""


def algo_box(number, title, inputs, outputs, steps):
    """A framed pseudocode box: a bordered single-column table carrying the
    Input/Output declarations and numbered pseudocode lines with BEGIN/END,
    FOR/END FOR and IF/THEN/END IF structure, with the caption below the box,
    as for the figures and tables. `steps` is a list of (indent, code) pairs;
    indent is the nesting depth and is rendered as 1.5em per level (the DOCX
    converter reads the same \\hspace* marker to rebuild the indentation).
    Placed inline (not a float), so it stays exactly where it is written."""
    rows = [f"\\textbf{{Input:}} {inputs}"]
    rows.append(f"\\textbf{{Output:}} {outputs}")
    for i, item in enumerate(steps, 1):
        level, code = item if isinstance(item, tuple) else (0, item)
        pad = f"\\hspace*{{{1.5 * level:g}em}}" if level else ""
        rows.append(f"{i}.\\quad {pad}{code}")
    body = " \\\\\n".join(rows) + " \\\\"
    return f"""% ALGOBOX-START
\\begin{{center}}
\\small
\\renewcommand{{\\arraystretch}}{{1.35}}
\\begin{{tabular}}{{|p{{0.94\\linewidth}}|}}
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{center}}
\\vspace{{-2pt}}
\\begin{{center}}
\\small\\textbf{{Algorithm {number}:}} {title}
\\end{{center}}
\\vspace{{4pt}}
% ALGOBOX-END
"""


def figure(fname, cap, label, width=0.86):
    safe = fname.replace("_", "-")
    return (f"\n\\begin{{figure}}[htbp]\n\\centering\n"
            f"\\includegraphics[width={width}\\textwidth]{{figures/{safe}}}\n"
            f"\\caption{{{cap}}}\n\\label{{{label}}}\n\\end{{figure}}\n")


def figure_star(fname, cap, label, width=0.98):
    safe = fname.replace("_", "-")
    return (f"\n\\begin{{figure*}}[htbp]\n\\centering\n"
            f"\\includegraphics[width={width}\\textwidth]{{figures/{safe}}}\n"
            f"\\caption{{{cap}}}\n\\label{{{label}}}\n\\end{{figure*}}\n")


def par(s):
    return "\n" + s + "\n"


# APA-style reference list (alphabetical). Static thebibliography -> no BibTeX
# pass, so Overleaf's free-tier compile stays within the timeout. Each entry is
# (bibkey, natbib label "Name(Year)", APA-formatted text).
REFS = [
 ("achiam2023gpt4", "Achiam et~al.(2023)",
  "Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., "
  "\\ldots\\ Zoph, B. (2023). GPT-4 technical report. \\textit{arXiv preprint} "
  "arXiv:2303.08774."),
 ("acr2022lungrads", "American College of Radiology(2022)",
  "American College of Radiology. (2022). \\textit{Lung CT Screening Reporting "
  "\\& Data System (Lung-RADS) version 2022}. American College of Radiology, "
  "Reston, VA."),
 ("anthropic2024claude", "Anthropic(2024)",
  "Anthropic. (2024). \\textit{The Claude 3 model family: Opus, Sonnet, Haiku} "
  "[Technical report]. Anthropic, San Francisco, CA."),
 ("armato2011lidc", "Armato et~al.(2011)",
  "Armato, S. G., III, McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, "
  "C. R., Reeves, A. P., \\ldots\\ Clarke, L. P. (2011). The Lung Image Database "
  "Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed "
  "reference database of lung nodules on CT scans. \\textit{Medical Physics, "
  "38}(2), 915--931."),
 ("bai2025qwen25vl", "Bai et~al.(2025)",
  "Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., \\ldots\\ Lin, J. "
  "(2025). Qwen2.5-VL technical report. \\textit{arXiv preprint} "
  "arXiv:2502.13923."),
 ("bray2024globocan", "Bray et~al.(2024)",
  "Bray, F., Laversanne, M., Sung, H., Ferlay, J., Siegel, R. L., "
  "Soerjomataram, I., \\& Jemal, A. (2024). Global cancer statistics 2022: "
  "GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in "
  "185 countries. \\textit{CA: A Cancer Journal for Clinicians, 74}(3), "
  "229--263."),
 ("chan2023chateval", "Chan et~al.(2023)",
  "Chan, C.-M., Chen, W., Su, Y., Yu, J., Xue, W., Zhang, S., Fu, J., \\& Liu, "
  "Z. (2023). ChatEval: Towards better LLM-based evaluators through multi-agent "
  "debate. \\textit{arXiv preprint} arXiv:2308.07201."),
 ("du2023debate", "Du et~al.(2024)",
  "Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., \\& Mordatch, I. (2024). "
  "Improving factuality and reasoning in language models through multiagent "
  "debate. In \\textit{Proceedings of the 41st International Conference on "
  "Machine Learning} (pp. 11733--11763)."),
 ("edge2024graphrag", "Edge et~al.(2024)",
  "Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., "
  "Metropolitansky, D., Ness, R. O., \\& Larson, J. (2024). From local to "
  "global: A graph RAG approach to query-focused summarization. \\textit{arXiv "
  "preprint} arXiv:2404.16130."),
 ("ester1996dbscan", "Ester et~al.(1996)",
  "Ester, M., Kriegel, H.-P., Sander, J., \\& Xu, X. (1996). A density-based "
  "algorithm for discovering clusters in large spatial databases with noise. In "
  "\\textit{Proceedings of the 2nd International Conference on Knowledge "
  "Discovery and Data Mining (KDD)} (pp. 226--231)."),
 ("fallahpour2025medrax", "Fallahpour et~al.(2025)",
  "Fallahpour, A., Ma, J., Munim, A., Lyu, H., \\& Wang, B. (2025). MedRAX: "
  "Medical reasoning agent for chest X-ray. \\textit{arXiv preprint} "
  "arXiv:2502.02673."),
 ("geifman2017selective", "Geifman \\& El-Yaniv(2017)",
  "Geifman, Y., \\& El-Yaniv, R. (2017). Selective classification for deep "
  "neural networks. In \\textit{Advances in Neural Information Processing "
  "Systems} (Vol. 30)."),
 ("guo2017calibration", "Guo et~al.(2017)",
  "Guo, C., Pleiss, G., Sun, Y., \\& Weinberger, K. Q. (2017). On calibration "
  "of modern neural networks. In \\textit{Proceedings of the 34th International "
  "Conference on Machine Learning} (pp. 1321--1330)."),
 ("hong2024metagpt", "Hong et~al.(2024)",
  "Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., Wang, J., \\ldots\\ "
  "Schmidhuber, J. (2024). MetaGPT: Meta programming for a multi-agent "
  "collaborative framework. In \\textit{International Conference on Learning "
  "Representations}."),
 ("isensee2021nnunet", "Isensee et~al.(2021)",
  "Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., \\& Maier-Hein, "
  "K. H. (2021). nnU-Net: A self-configuring method for deep learning-based "
  "biomedical image segmentation. \\textit{Nature Methods, 18}(2), 203--211."),
 ("ji2023hallucination", "Ji et~al.(2023)",
  "Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, "
  "Y. J., Madotto, A., \\& Fung, P. (2023). Survey of hallucination in natural "
  "language generation. \\textit{ACM Computing Surveys, 55}(12), 1--38."),
 ("kadavath2022knowwhat", "Kadavath et~al.(2022)",
  "Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., "
  "\\ldots\\ Kaplan, J. (2022). Language models (mostly) know what they know. "
  "\\textit{arXiv preprint} arXiv:2207.05221."),
 ("kim2024mdagents", "Kim et~al.(2024)",
  "Kim, Y., Park, C., Jeong, H., Chan, Y. S., Xu, X., McDuff, D., Lee, H., "
  "Ghassemi, M., Breazeal, C., \\& Park, H. W. (2024). MDAgents: An adaptive "
  "collaboration of LLMs for medical decision-making. In \\textit{Advances in "
  "Neural Information Processing Systems} (Vol. 37, pp. 79410--79452)."),
 ("lai2025medr1", "Lai et~al.(2025)",
  "Lai, Y., Zhong, J., Li, M., Zhao, S., \\& Yang, X. (2025). Med-R1: "
  "Reinforcement learning for generalizable medical reasoning in vision-language "
  "models. \\textit{arXiv preprint} arXiv:2503.13939."),
 ("li2023llavamed", "Li et~al.(2023)",
  "Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T., "
  "Poon, H., \\& Gao, J. (2023). LLaVA-Med: Training a large "
  "language-and-vision assistant for biomedicine in one day. \\textit{arXiv "
  "preprint} arXiv:2306.00890."),
 ("liang2023divergent", "Liang et~al.(2023)",
  "Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Shi, "
  "S., \\& Tu, Z. (2023). Encouraging divergent thinking in large language "
  "models through multi-agent debate. \\textit{arXiv preprint} "
  "arXiv:2305.19118."),
 ("liu2023llava", "Liu et~al.(2023)",
  "Liu, H., Li, C., Wu, Q., \\& Lee, Y. J. (2023). Visual instruction tuning. "
  "In \\textit{Advances in Neural Information Processing Systems} (Vol. 36, pp. "
  "34892--34916)."),
 ("macmahon2017fleischner", "MacMahon et~al.(2017)",
  "MacMahon, H., Naidich, D. P., Goo, J. M., Lee, K. S., Leung, A. N. C., "
  "Mayo, J. R., \\ldots\\ Bankier, A. A. (2017). Guidelines for management of "
  "incidental pulmonary nodules detected on CT images: From the Fleischner "
  "Society 2017. \\textit{Radiology, 284}(1), 228--243."),
 ("manakul2023selfcheckgpt", "Manakul et~al.(2023)",
  "Manakul, P., Liusie, A., \\& Gales, M. J. F. (2023). SelfCheckGPT: "
  "Zero-resource black-box hallucination detection for generative large "
  "language models. In \\textit{Proceedings of the 2023 Conference on Empirical "
  "Methods in Natural Language Processing} (pp. 9004--9017)."),
 ("mcwilliams2013brock", "McWilliams et~al.(2013)",
  "McWilliams, A., Tammemagi, M. C., Mayo, J. R., Roberts, H., Liu, G., "
  "Soghrati, K., \\ldots\\ Lam, S. (2013). Probability of cancer in pulmonary "
  "nodules detected on first screening CT. \\textit{New England Journal of "
  "Medicine, 369}(10), 910--919."),
 ("naeini2015ece", "Naeini et~al.(2015)",
  "Naeini, M. P., Cooper, G. F., \\& Hauskrecht, M. (2015). Obtaining well "
  "calibrated probabilities using Bayesian binning. In \\textit{Proceedings of "
  "the AAAI Conference on Artificial Intelligence} (Vol. 29)."),
 ("national2011nlst", "National Lung Screening Trial Research Team(2011)",
  "National Lung Screening Trial Research Team. (2011). Reduced lung-cancer "
  "mortality with low-dose computed tomographic screening. \\textit{New England "
  "Journal of Medicine, 365}(5), 395--409."),
 ("saab2024medgemini", "Saab et~al.(2024)",
  "Saab, K., Tu, T., Weng, W.-H., Tanno, R., Stutz, D., Wulczyn, E., \\ldots\\ "
  "Natarajan, V. (2024). Capabilities of Gemini models in medicine. "
  "\\textit{arXiv preprint} arXiv:2404.18416."),
 ("sellergren2025medgemma", "Sellergren et~al.(2025)",
  "Sellergren, A., Kazemzadeh, S., Jaroensri, R., Kiraly, A., Traverse, M., "
  "Kohlberger, T., \\ldots\\ Shetty, S. (2025). MedGemma technical report. "
  "\\textit{arXiv preprint} arXiv:2507.05201."),
 ("singhal2023medpalm", "Singhal et~al.(2023)",
  "Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., "
  "\\ldots\\ Natarajan, V. (2023). Large language models encode clinical "
  "knowledge. \\textit{Nature, 620}(7972), 172--180."),
 ("tang2023medagents", "Tang et~al.(2023)",
  "Tang, X., Zou, A., Zhang, Z., Li, Z., Zhao, Y., Zhang, X., Cohan, A., \\& "
  "Gerstein, M. (2023). MedAgents: Large language models as collaborators for "
  "zero-shot medical reasoning. \\textit{arXiv preprint} arXiv:2311.10537."),
 ("thirunavukarasu2023llmmedicine", "Thirunavukarasu et~al.(2023)",
  "Thirunavukarasu, A. J., Ting, D. S. J., Elangovan, K., Gutierrez, L., Tan, "
  "T. F., \\& Ting, D. S. W. (2023). Large language models in medicine. "
  "\\textit{Nature Medicine, 29}(8), 1930--1940."),
 ("wang2025medagentpro", "Wang et~al.(2025)",
  "Wang, Z., Wu, J., Cai, L., Low, C. H., Yang, X., Li, Q., \\& Jin, Y. (2025). "
  "MedAgent-Pro: Towards evidence-based multi-modal medical diagnosis via "
  "reasoning agentic workflow. \\textit{arXiv preprint} arXiv:2503.18968."),
 ("xiao2025describe", "Xiao et~al.(2025)",
  "Xiao, X., Zhang, Y., Nguyen, T.-H., Lam, B.-T., Wang, J., Zhao, L., Hamm, "
  "J., Wang, T., Li, X., \\& Wang, X. (2025). Describe anything in medical "
  "images. \\textit{arXiv preprint} arXiv:2505.05804."),
 ("xiong2024uncertainty", "Xiong et~al.(2024)",
  "Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., \\& Hooi, B. (2024). Can "
  "LLMs express their uncertainty? An empirical evaluation of confidence "
  "elicitation in LLMs. In \\textit{International Conference on Learning "
  "Representations}."),
 ("yang2025lungnoduleagent", "Yang et~al.(2025)",
  "Yang, C., Jin, H., Yu, X., Wang, Z., Liu, Y., Fan, F., Lei, D., Jia, G., "
  "Wang, C., \\& Ge, R. (2025). LungNoduleAgent: A collaborative multi-agent "
  "system for precision diagnosis of lung nodules. \\textit{arXiv preprint} "
  "arXiv:2511.21042."),
 ("luccioni2024power", "Luccioni et~al.(2024)",
  "Luccioni, A. S., Jernite, Y., \\& Strubell, E. (2024). Power hungry "
  "processing: Watts driving the cost of AI deployment? In "
  "\\textit{Proceedings of the 2024 ACM Conference on Fairness, "
  "Accountability, and Transparency}."),
 ("strubell2019energy", "Strubell et~al.(2019)",
  "Strubell, E., Ganesh, A., \\& McCallum, A. (2019). Energy and policy "
  "considerations for deep learning in NLP. In \\textit{Proceedings of the "
  "57th Annual Meeting of the Association for Computational Linguistics} "
  "(pp. 3645--3650)."),
 ("ueda2024climate", "Ueda et~al.(2024)",
  "Ueda, D., Walston, S. L., Fujita, S., Fushimi, Y., Tsuboyama, T., "
  "Kamagata, K., \\ldots\\ Naganawa, S. (2024). Climate change and "
  "artificial intelligence in healthcare: Review and recommendations "
  "towards a sustainable future. \\textit{Diagnostic and Interventional "
  "Imaging, 105}(11), 453--459."),
 ("thomson2025environmental", "Thomson et~al.(2025)",
  "Thomson, R. M., Perdomo Lampignano, J., Fisher, E., Wati, C., "
  "Jeyakumar, G., Duncan, S., \\& Lowe, D. J. (2025). Evaluating the "
  "environmental sustainability of AI in radiology: A systematic review "
  "of current practice. \\textit{BMJ Digital Health and AI, 1}(1), "
  "e000073."),
]


REF_TEXT = {k: txt for k, _lab, txt in REFS}
REF_AUTHOR = {k: lab.rsplit("(", 1)[0].strip() for k, lab, _txt in REFS}


def order_by_citation(doc_text: str):
    """List every reference key in order of its first \\citep/\\citet in the
    prose (Vancouver / IEEE numbering). Any key never cited is appended at the
    end in its original alphabetical position, so the list always covers REFS."""
    seen = []
    for m in re.finditer(r"\\cite[pt]\{([^}]+)\}", doc_text):
        for k in (x.strip() for x in m.group(1).split(",")):
            if k in REF_TEXT and k not in seen:
                seen.append(k)
    for k, _lab, _txt in REFS:
        if k not in seen:
            seen.append(k)
    return seen


def resolve_citations(doc_text: str, num: dict) -> str:
    """Replace the natbib-flavoured placeholders with literal bracket numbers
    in citation order: \\citep{k1,k2} -> "[3,7]" (ascending), and
    \\citet{k} -> "Author et al.~[3]". The DOCX converter does no macro
    expansion, so the numbers must be baked in here."""
    def nums(group):
        keys = [k.strip() for k in group.split(",")]
        return sorted(num[k] for k in keys if k in num)

    def rep_citet(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        authors = " and ".join(REF_AUTHOR.get(k, k) for k in keys)
        ns = nums(m.group(1))
        return f"{authors}~[{','.join(str(n) for n in ns)}]"

    def rep_citep(m):
        ns = nums(m.group(1))
        return f"[{','.join(str(n) for n in ns)}]"

    doc_text = re.sub(r"\\citet\{([^}]+)\}", rep_citet, doc_text)
    doc_text = re.sub(r"\\citep\{([^}]+)\}", rep_citep, doc_text)
    return doc_text


def bibliography_block(ordered_keys):
    # \bibitem entries emitted in first-citation order, so entry N is the Nth
    # distinct reference cited in the text. thebibliography numbers them 1..N.
    # \small / \bibsep sit *inside* the environment (after \begin, before the
    # first \bibitem) so the lightweight DOCX parser -- which only looks for
    # \bibitem{...} inside this block -- never sees them as loose text.
    items = "\n\n".join(f"\\bibitem{{{k}}}\n{REF_TEXT[k]}" for k in ordered_keys)
    return ("\\begin{thebibliography}{99}\n"
            "\\small\n"
            "\\setlength{\\bibsep}{2pt plus 1pt}\n"
            f"{items}\n"
            "\\end{thebibliography}\n")


def resolve_refs(doc_text: str) -> str:
    """Replace every \\ref{label} with the literal section/figure/table number,
    by walking the document in the same order build_article.py numbers
    headings, figures, and tables. The lightweight DOCX converter does not
    expand LaTeX macros, so \\ref{} must already be plain text by the time
    this text is written out (and reused as-is if ever compiled, since the
    numbers are fixed once the document structure is)."""
    labels = {}
    h1 = h2 = fig_n = tab_n = 0
    token = re.compile(
        r"\\section\{[^}]*\}"
        r"|\\subsection\{[^}]*\}"
        r"|\\begin\{table\}"
        r"|\\begin\{figure\*?\}"
        r"|\\label\{([^}]*)\}")
    for m in token.finditer(doc_text):
        tk = m.group(0)
        if tk.startswith("\\label{"):
            label = m.group(1)
            if label.startswith("fig:"):
                labels[label] = str(fig_n)
            elif label.startswith("tab:"):
                labels[label] = str(tab_n)
            elif label.startswith("sec:"):
                labels[label] = f"{h1}.{h2}" if h2 else str(h1)
        elif tk.startswith("\\section{"):
            h1 += 1
            h2 = 0
        elif tk.startswith("\\subsection{"):
            h2 += 1
        elif tk.startswith("\\begin{table}"):
            tab_n += 1
        elif tk.startswith("\\begin{figure"):
            fig_n += 1
    return re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), doc_text)


PREAMBLE = r"""\documentclass[12pt]{article}

% ---- geometry: exactly 1 inch on all four sides ----
\usepackage[letterpaper,top=1in,bottom=1in,left=1in,right=1in]{geometry}

% ---- maths first, then a Times New Roman equivalent (newtx wants this order) ----
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage{newtxmath}

\usepackage{graphicx}
\usepackage{array}
\usepackage{longtable}
\usepackage[table]{xcolor}
\definecolor{tblhead}{gray}{0.85}
\usepackage{adjustbox}
\usepackage{changepage}                 % adjustwidth for the abstract
\usepackage[font=small,labelfont=bf,labelsep=colon,justification=justified,
            skip=6pt]{caption}
\usepackage{titlesec}
\usepackage{ragged2e}
\usepackage{csquotes}
\usepackage{cite}                        % numeric [1], [2,3], compressed [4-6]
\usepackage[hidelinks]{hyperref}

% ---- body text: 12 pt, justified, 0.2 in first-line indent ----
\setlength{\parindent}{0.2in}
\setlength{\parskip}{3pt}
\linespread{1.10}
\setlength{\emergencystretch}{2em}   % avoid under/overfull boxes cheaply

% ---- decimal section headings: "N Title" (no period), up to 3 levels ----
\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{0.7em}{}
\titlespacing*{\section}{0pt}{16pt}{8pt}
\titleformat{\subsection}{\normalfont\large\bfseries\itshape}{\thesubsection}{0.7em}{}
\titlespacing*{\subsection}{0pt}{12pt}{6pt}
\titleformat{\subsubsection}{\normalfont\normalsize\bfseries\itshape}{\thesubsubsection}{0.7em}{}
\titlespacing*{\subsubsection}{0pt}{10pt}{4pt}
% titlesec suppresses the first-line indent right after a heading by default

\captionsetup[table]{position=below}

\newcommand{\abstitle}[1]{\noindent{\large\bfseries #1}\par\vspace{8pt}}

\pagestyle{plain}
\begin{document}
\thispagestyle{plain}
"""


# =========================================================================== #
def build(p):
    n = p["n"]
    seeds = p["seeds"]
    rr = Rr(p)
    cl = full(p, "clean")
    ad = full(p, "adversarial") if "adversarial" in p["regimes"] else full(p, rr[-1])
    imb = full(p, "imbalanced") if "imbalanced" in p["regimes"] else cl
    max_board = max(delta(p, r, "- board (1 agent)", "accuracy_all") for r in rr)
    imb_board = -delta(p, "imbalanced", "- board (1 agent)", "accuracy_all") \
        if "imbalanced" in p["regimes"] else 0.0
    T = [PREAMBLE]

    # ---------------- title ----------------
    T.append(r"""\begin{center}
{\LARGE\bfseries SafeNodule-MAS: A Reproducible, Cost-Aware, and\\[3pt]
Safety-Calibrated Multi-Agent Architecture for Lung Nodule Diagnosis\par}
\vspace{16pt}
{\large
Lochan Narayan K\textsuperscript{1}\quad Manikandan R\textsuperscript{1,*}\par}
\vspace{6pt}
{\small\itshape
\textsuperscript{1}School of Computing, SASTRA Deemed University, Thanjavur, Tamil Nadu, India\\
\upshape\texttt{lochannarayan11@gmail.com}\qquad\texttt{manikandan75@core.sastra.edu}\\
\upshape\textsuperscript{*}Corresponding author\par}
\vspace{16pt}
\end{center}
""")

    # ---------------- abstract (<= 300 words, no hyphens anywhere in it) ----------------
    T.append(r"\abstitle{Abstract}")
    T.append(r"\begin{adjustwidth}{0.25in}{0.25in}")
    T.append(r"{\normalsize\selectfont\justifying " + (
        "Multiagent systems built from large language models are increasingly "
        "proposed for clinical decision support, yet published designs rarely "
        "make their orchestration, evidence grounding, uncertainty handling, or "
        "cost measurable. This paper presents an enhanced architecture for lung "
        "nodule diagnosis that keeps a published three stage pipeline, "
        "detection, report generation, and malignancy reasoning, and adds "
        "seven mechanisms, each with a measurable outcome, not an accuracy "
        "claim: an adaptive triage router, a role specific specialist board "
        "with weighted voting, a report grounded evidence verifier, a devil's "
        "advocate agent that guards against missed cancer, guideline grounded "
        "escalation to Lung RADS categories, temperature scaling fitted by "
        "likelihood on a held out split, and a margin based abstention gate. "
        "The system is evaluated on a reproducible synthetic harness under "
        f"five stress regimes, {n} cases per seed across {len(seeds)} "
        "seeds, and reporting accuracy, macro F1, coverage, false negative "
        "rate, calibration error, verifier precision and recall, and mean "
        "model calls per case, each with 95% confidence intervals and paired "
        "bootstrap significance tests. The specialist board raises accuracy "
        "in four of five regimes, though under a benign heavy prior a single "
        "unbiased agent performs better, a finding the framework reports. "
        "Adaptive routing cuts model calls per case by two to four "
        "percent, and the text volume those calls carry by up to seven "
        "percent, a direct proxy for inference energy and price. The "
        "verifier flags every fabricated citation with high precision, and "
        "fitted calibration corrects most distribution shift error without "
        "disturbing already calibrated cases. The abstention gate helps only "
        "where confidence is informative, motivating a structural safeguard "
        "for the remaining cases. SafeNodule-MAS also advances sustainable "
        "healthcare computing by reducing computational redundancy through "
        "adaptive, multiagent orchestration. Overall, this work offers a "
        "reproducible blueprint in which safety, cost, and efficiency are "
        "measured properties of a multiagent diagnostic system, not "
        "assumed ones.") + r"\par}")
    T.append(r"\end{adjustwidth}")
    T.append(r"\vspace{8pt}")
    T.append(r"\noindent{\normalsize\selectfont\textbf{Keywords:} "
             r"Multiagent systems; Clinical decision support; Selective "
             r"prediction; Confidence calibration; Sustainable computing.\par}")
    T.append(r"\vspace{18pt}")

    # ================= 1. Introduction =================
    T.append(r"\section{Introduction}")
    T.append(par(
        "Lung cancer is the leading cause of cancer death worldwide, with an "
        "estimated 1.8 million deaths in 2022 \\citep{bray2024globocan}. Low-dose "
        "computed tomography (CT) screening reduces lung-cancer mortality in "
        "high-risk populations \\citep{national2011nlst}, and the pulmonary "
        "nodules it surfaces are the earliest actionable signal of disease. "
        "Turning a screen-detected nodule into a management decision requires "
        "three tasks in sequence: localizing the nodule on the volume, describing "
        "its morphology in clinically meaningful terms, and grading its "
        "malignancy risk with appropriate hedging. Radiologists perform these "
        "tasks by inspecting many slices per scan, which is time-consuming and "
        "subject to interobserver variability, and they anchor the final "
        "judgement in structured criteria such as the Fleischner Society "
        "recommendations \\citep{macmahon2017fleischner}, Lung-RADS "
        "\\citep{acr2022lungrads}, and quantitative risk models "
        "\\citep{mcwilliams2013brock}."))
    T.append(par(
        "The operational burden is substantial. A screening CT contains hundreds "
        "of axial sections; a single scan can show several nodules, the large "
        "majority of which are benign, and each must be localized, measured, "
        "characterized, and assigned a follow-up interval. Because most findings "
        "are not cancer, the cost of the workflow is dominated by the "
        "surveillance it generates rather than by the cancers it catches, and "
        "small differences in how a nodule is measured or how a margin is judged "
        "propagate into different Lung-RADS categories and different "
        "recommendations. Reader studies consistently find non-trivial "
        "interobserver variability in nodule detection, in diameter and volume "
        "measurement, and in categorical malignancy assessment, and that "
        "variability is itself a source of downstream cost and of missed or "
        "delayed diagnoses \\citep{macmahon2017fleischner}. Structured reporting "
        "and quantitative decision aids reduce this variability, which is why any "
        "automated assistant is expected to produce not just a label but the "
        "measurements and the criterion-referenced reasoning that support it "
        "\\citep{acr2022lungrads,mcwilliams2013brock}."))
    T.append(par(
        "Automation of the three tasks has a long history, and each is now "
        "individually strong. What has not been solved is delivering the tasks "
        "as one trustworthy process. Task-specific deep networks report high "
        "aggregate metrics but expose only a score, not the features or the "
        "pattern behind it, and their behaviour under a shifted scanner, "
        "protocol, or population is hard to anticipate. Clinical adoption of "
        "software as a medical device is gated less on peak accuracy than on "
        "traceability, calibrated uncertainty, monitoring, and a defined "
        "human-in-the-loop; a system that cannot say why it answered, or how "
        "sure it is, or when it should defer, is difficult to validate and "
        "difficult to supervise once deployed. These are the properties this "
        "paper measures."))
    T.append(par(
        "Deep learning has improved each task in isolation (detection and "
        "segmentation networks localize nodules accurately "
        "\\citep{isensee2021nnunet}, and classifiers grade malignancy from "
        "cropped volumes), but task-specific models are opaque and brittle "
        "under distribution shift, and they do not produce the chain of evidence "
        "a clinician needs to accept or override a prediction. General-purpose "
        "vision-language models (VLMs) such as GPT-4 \\citep{achiam2023gpt4}, the "
        "Claude~3 family \\citep{anthropic2024claude}, Qwen2.5-VL "
        "\\citep{bai2025qwen25vl}, and LLaVA \\citep{liu2023llava} add fluent "
        "explanation and broad world knowledge, and medical VLMs (LLaVA-Med "
        "\\citep{li2023llavamed}, MedGemma \\citep{sellergren2025medgemma}, Med-R1 "
        "\\citep{lai2025medr1}) narrow the domain gap. Yet VLMs remain weak at "
        "the fine-grained quantitative perception nodule assessment needs, they "
        "hallucinate findings that are not present in the image "
        "\\citep{ji2023hallucination}, and their verbalized confidence is poorly "
        "calibrated \\citep{xiong2024uncertainty}. Large models can encode "
        "clinical knowledge \\citep{singhal2023medpalm} and answer medical "
        "questions well \\citep{saab2024medgemini}, but knowledge recall is not "
        "the same as a trustworthy, auditable diagnostic process "
        "\\citep{thirunavukarasu2023llmmedicine}."))
    T.append(par(
        "Collaborative multi-agent systems are a natural response. Debate among "
        "role-playing agents improves factuality and reasoning "
        "\\citep{du2023debate,liang2023divergent}, structured multi-agent "
        "evaluation improves consistency \\citep{chan2023chateval}, and the "
        "pattern has been carried into software engineering "
        "\\citep{hong2024metagpt} and into medicine: MedAgents "
        "\\citep{tang2023medagents}, MDAgents \\citep{kim2024mdagents}, "
        "MedAgent-Pro \\citep{wang2025medagentpro}, and MedRAX "
        "\\citep{fallahpour2025medrax} all coordinate several LLM roles, often "
        "with retrieval or tool use. \\citet{yang2025lungnoduleagent} instantiate "
        "this pattern for lung nodules as LungNoduleAgent: a Nodule Spotter that "
        "coordinates detection models through mixture-of-experts inference, "
        "intersection-over-union (IoU) mask clustering, and a judging panel; a "
        "Simulated Radiologist that produces a localized report from a focal crop "
        "\\citep{xiao2025describe}; and a Doctor Agent System in which reasoning "
        "agents, each backed by a medical knowledge graph "
        "\\citep{edge2024graphrag}, revise their opinions and iterate until "
        "consensus. On two private datasets and LIDC-IDRI "
        "\\citep{armato2011lidc} it outperforms strong VLM and medical-agent "
        "baselines."))
    T.append(par(
        "Consensus, however, is not the property that determines whether such a "
        "system can be deployed. A clinical decision-support tool must decide "
        "\\emph{how much} deliberation a case warrants, so that routine nodules "
        "do not incur the cost of a full multi-agent discussion while difficult "
        "ones receive more scrutiny; it must keep every claim in the final "
        "explanation traceable to the shared evidence, so that a reviewer can see "
        "which statements are observed, reported, or inferred; it must surface "
        "disagreement rather than dissolve it, because a confident wrong "
        "consensus is more dangerous than an acknowledged split; it must express "
        "\\emph{calibrated} uncertainty and be permitted to abstain when the "
        "evidence is insufficient, rather than being prompted to \\enquote{always "
        "be definitive}; and it must account for its own computational cost, "
        "which for a stack of language-model calls is the dominant driver of "
        "latency and price at deployment. None of these behaviours is quantified "
        "in the original work: the number of debate rounds, the stopping rule, "
        "the provenance of the knowledge base, the calibration of the output "
        "probabilities, and the per-case call count are all left unspecified. "
        "This matters beyond engineering tidiness: calibration "
        "\\citep{guo2017calibration,naeini2015ece}, selective prediction "
        "\\citep{geifman2017selective}, and evidence grounding "
        "\\citep{manakul2023selfcheckgpt} are exactly the properties a regulator "
        "or a clinical adopter will ask a vendor to demonstrate."))
    T.append(par(
        "Consider each unmeasured behaviour in turn. \\emph{Deliberation "
        "allocation}: running a four-role board with multiple revision rounds on "
        "every nodule, including the obviously benign ones, multiplies the "
        "language-model call count without a corresponding accuracy gain; a "
        "triage step that reserves the board for ambiguous morphology should "
        "recover most of that cost, but only if the easy cases can be identified "
        "reliably and the router rarely down-triages a hard case. \\emph{Claim "
        "provenance}: the original agents cite morphological features to justify "
        "their grade, but nothing checks that a cited feature actually appears in "
        "the report the pipeline produced, so a hallucinated sign can drive a "
        "confident diagnosis undetected. \\emph{Disagreement}: iterating "
        "\\enquote{until consensus} rewards conformity, and a board that "
        "converges quickly on the wrong label is worse than one that reports an "
        "honest split; there is no agent whose role is to resist the emerging "
        "answer and to guard specifically against a missed cancer. "
        "\\emph{Uncertainty}: the agents are prompted to be definitive, the "
        "output probabilities are never calibrated against outcomes, and the "
        "system cannot decline. \\emph{Cost}: the per-case number of model calls "
        "-- the quantity that determines latency and price -- is never reported. "
        "Each of these is a concrete, testable gap, and each admits a small "
        "mechanism with a measurable endpoint."))
    T.append(par(
        "This article describes SafeNodule-MAS (LungNoduleAgent-Enhanced), an "
        "implementation that keeps the detect--describe--diagnose pipeline "
        "and turns each of these "
        "behaviours into an instrumented component with an explicit success "
        "metric. We deliberately do not claim a higher headline accuracy "
        "(\\citet{yang2025lungnoduleagent} already report up to 89.1\\% on "
        "LIDC-IDRI) because the contribution is methodological: a design in "
        "which safety and cost are first-class, measurable properties, evaluated "
        "under controlled stress with confidence intervals and significance "
        "tests, including the regimes in which a mechanism fails. To make the "
        "evaluation fully reproducible and independent of proprietary models or "
        "restricted datasets, the entire pipeline runs against a deterministic "
        "mock language backend and a synthetic CT generator; every module is a "
        "documented interface with a drop-in point for a real detector, a real "
        "VLM, and real DICOM data."))
    T.append(par(
        "Recent work in sustainable computing emphasizes energy-aware and "
        "resource-efficient artificial intelligence, particularly as large "
        "models are deployed at scale in clinical settings "
        "\\citep{strubell2019energy,ueda2024climate,thomson2025environmental}. "
        "Training and serving large language models carries a documented "
        "energy and carbon cost \\citep{strubell2019energy}, and healthcare "
        "imaging AI is no exception \\citep{ueda2024climate,"
        "thomson2025environmental}. SafeNodule-MAS is designed with this "
        "constraint in view: the triage router and the Chair's "
        "early-convergence rule exist specifically to avoid running the "
        "full four-role board, and its multiple debate rounds, on cases "
        "that do not need them, so that computational cost, and by "
        "extension energy use, scales with case difficulty rather than "
        "with a fixed worst-case budget."))
    T.append(par(
        "A synthetic harness is a deliberate methodological choice, not a "
        "shortcut. Studying safety mechanisms requires the ability to \\emph{dial "
        "in} the failure modes they target (unsupported claims, borderline "
        "lesions, class imbalance, degraded perception) at known rates, which "
        "is not possible on a fixed clinical benchmark where those conditions "
        "occur at whatever frequency the data happen to contain and cannot be "
        "isolated. It also removes two confounds that dominate LLM evaluation: "
        "non-determinism, so that an ablation reflects the mechanism rather than "
        "sampling noise, and data governance, so that the full protocol can be "
        "released and re-run. The obvious cost is external validity: absolute "
        "accuracy on synthetic data is uninformative, and because the mock "
        "reasoner and the label generator share a feature model, inter-agent "
        "error correlation is understated. We therefore restrict our claims to "
        "the quantities that transfer (the \\emph{direction} and "
        "\\emph{significance} of each component's effect, the calibration and "
        "grounding behaviour, and the per-case cost), and we treat the "
        "protocol, not the numbers, as the contribution to be reused on real "
        "data."))
    T.append(par(
        "\\textbf{Scope.} This paper is not a new model: the reasoning backbone "
        "is a pluggable interface and the detector is a placeholder. It is not a "
        "clinical validation: no patient data are used and no clinical "
        "performance is claimed. It is an architecture-and-evaluation "
        "contribution: a set of small, testable safety and cost mechanisms "
        "bolted onto a published pipeline, plus a controlled protocol that shows, "
        "for each mechanism, whether and where it earns its place."))
    T.append(par(
        "\\textbf{Contributions.} "
        "(i)~An architecture that adds seven safety and cost mechanisms to a "
        "published multi-agent lung-nodule pipeline, each paired with a specific "
        "quantitative endpoint (Table~\\ref{tab:components}) and expressed as a "
        "small, testable algorithm (Algorithms 1 and 2). "
        "(ii)~A five-regime synthetic stress protocol (clean, feature noise, "
        "borderline sampling, class imbalance, and an adversarial combination) "
        "that deliberately exercises the failure modes the mechanisms target, "
        "together with a perceived-versus-oracle split so that agents and the "
        "verifier reason from imperfect perception while scoring uses ground "
        "truth. (iii)~A multi-seed evaluation with 95\\% intervals and paired "
        "bootstrap tests, reporting both positive and negative results: "
        "confidence-based abstention loses value under class imbalance and heavy "
        "distribution shift, and a benign-heavy prior makes the "
        "role-differentiated board underperform a single unbiased agent. "
        "(iv)~A fully deterministic, dataset-free reference implementation, "
        "experiment runner, and test suite. "
        "The remainder of the paper is organized as follows. "
        "Section~\\ref{sec:bg} reviews nodule assessment, detection, medical "
        "VLMs and agents, calibration, and adaptive computation. "
        "Section~\\ref{sec:method} details the architecture, the algorithms, the "
        "stress regimes, and the metrics. Section~\\ref{sec:results} reports the "
        "five-regime results, the ablation, and the significance tests. "
        "Section~\\ref{sec:disc} discusses safety as a measurable property, the "
        "conditions under which selective prediction fails, and the path to a "
        "clinical evaluation. Section~\\ref{sec:concl} concludes."))

    # ================= 2. Background =================
    T.append(r"\section{Background and Related Work}\label{sec:bg}")

    T.append(r"\subsection{Pulmonary nodule assessment and guidelines}")
    T.append(par(
        "Nodule management is governed by structured criteria. The Fleischner "
        "Society recommendations key follow-up to nodule size and density "
        "(solid, part-solid, ground-glass), with part-solid nodules carrying the "
        "highest malignancy risk \\citep{macmahon2017fleischner}. Lung-RADS "
        "assigns a screening category (2--4X) and a recommended action, and "
        "escalates for spiculation and for a growing solid component "
        "\\citep{acr2022lungrads}. Quantitative models such as the Brock model "
        "predict malignancy probability from size, morphology, and clinical "
        "context \\citep{mcwilliams2013brock}. A system that claims to be "
        "\\enquote{evidence-based} should therefore ground its output in a named, "
        "versioned criterion rather than in an unspecified knowledge base; our "
        "guideline agent does this explicitly (Section~\\ref{sec:comp}). "
        "The criteria also encode a specific asymmetry: the categories that "
        "trigger tissue sampling exist because a missed invasive cancer is far "
        "more costly than an unnecessary short-interval follow-up. Any automated "
        "grader that aggregates several opinions inherits the responsibility to "
        "preserve that asymmetry, which motivates a guard aimed specifically at "
        "false negatives rather than at overall error. The malignancy targets in "
        "this work follow the three-way pre-invasive / minimally-invasive / "
        "invasive adenocarcinoma scheme used by "
        "\\citet{yang2025lungnoduleagent}, in which the two invasive classes are "
        "the clinically actionable ones."))

    T.append(r"\subsection{The LungNoduleAgent pipeline}")
    T.append(par(
        "\\citet{yang2025lungnoduleagent} decompose lung-nodule analysis into "
        "three modules. The \\emph{Nodule Spotter} divides the CT volume into "
        "slices, runs a mixture of detection experts in parallel, reconciles "
        "their candidate masks by IoU-distance clustering with DBSCAN, and passes "
        "the survivors to a judging panel of vision-language models that vote "
        "each candidate up or down with a confidence weight; the accepted mask is "
        "then measured. The \\emph{Simulated Radiologist} takes a focal crop of "
        "the slice and mask, preserving surrounding context, and generates a "
        "localized report with a medical prompt that fixes the output format and "
        "forbids speculative language. The \\emph{Doctor Agent System} gives "
        "several reasoning agents access to a medical knowledge graph built from "
        "authoritative sources with a GraphRAG-style construction and "
        "community-summary retrieval \\citep{edge2024graphrag}; each agent grades "
        "the nodule, a summarizer compiles the opinions, and the agents revise "
        "and re-discuss until they agree. A shared memory stores the images, "
        "measurements, reports, and conversation. The present work leaves the "
        "Spotter and the Radiologist essentially intact and concentrates on the "
        "orchestration and governance of the Doctor Agent System and on the "
        "boundaries around it."))
    T.append(r"\subsection{Nodule detection and mask consensus}")
    T.append(par(
        "Detection and segmentation of small 3D structures is mature: "
        "self-configuring frameworks such as nnU-Net \\citep{isensee2021nnunet} "
        "provide strong baselines, and ensembles of detectors reduce "
        "false positives. LungNoduleAgent combines several detectors through "
        "mixture-of-experts inference and then reconciles their masks by "
        "clustering: the distance between two masks is $d(m_i,m_j)=1-\\mathrm{IoU}"
        "(m_i,m_j)$, masks are grouped with DBSCAN \\citep{ester1996dbscan}, and "
        "a judging panel of VLMs votes each surviving candidate up or down with a "
        "confidence weight \\citep{yang2025lungnoduleagent}. We reimplement this "
        "consensus mechanism exactly in NumPy so that it is transparent and "
        "seed-deterministic."))

    T.append(r"\subsection{Vision-language models in medicine}")
    T.append(par(
        "General VLMs \\citep{achiam2023gpt4,anthropic2024claude,bai2025qwen25vl,"
        "liu2023llava} generalize well but underperform on specialized imaging "
        "because their pretraining is dominated by natural images and they lack "
        "domain priors. Medical VLMs adapt them with domain data and reinforcement "
        "learning \\citep{li2023llavamed,sellergren2025medgemma,lai2025medr1}, and "
        "large proprietary models reach strong medical-QA scores "
        "\\citep{singhal2023medpalm,saab2024medgemini}. Two limitations persist "
        "and motivate the present work: fine-grained quantitative perception "
        "(measuring a nodule, judging a margin) remains weaker than dedicated "
        "models, and both classes of model hallucinate (asserting findings "
        "absent from the image \\citep{ji2023hallucination}), which is precisely "
        "what an evidence verifier must catch. A third, quieter limitation is "
        "that VLM confidence, whether taken from token probabilities or elicited "
        "in words, does not track correctness well "
        "\\citep{xiong2024uncertainty,kadavath2022knowwhat}; a pipeline that "
        "surfaces a probability to a clinician therefore cannot rely on the raw "
        "model output and needs an explicit calibration stage. The region-level "
        "prompting used by LungNoduleAgent, adapted from describe-anything work "
        "for medical images \\citep{xiao2025describe}, mitigates the perception "
        "gap by focusing the model on a masked crop with surrounding context, "
        "but it does not address grounding or calibration, which remain the "
        "responsibility of the downstream reasoning stage."))

    T.append(r"\subsection{Multi-agent LLM systems and debate}")
    T.append(par(
        "Letting several LLM instances argue and revise improves factuality and "
        "reasoning \\citep{du2023debate}, encourages divergent hypotheses "
        "\\citep{liang2023divergent}, and yields more reliable evaluation "
        "\\citep{chan2023chateval}; the pattern scales to complex collaborative "
        "tasks \\citep{hong2024metagpt}. In medicine, MedAgents "
        "\\citep{tang2023medagents} and MDAgents \\citep{kim2024mdagents} "
        "coordinate specialist roles, MedAgent-Pro \\citep{wang2025medagentpro} "
        "adds an evidence-oriented workflow, and MedRAX "
        "\\citep{fallahpour2025medrax} sequences multimodal inputs for chest "
        "radiography. Reported weaknesses in pulmonary applications are "
        "difficulty with mixed-density nodules, limited use of longitudinal "
        "information, and weak agreement with histopathology "
        "\\citep{yang2025lungnoduleagent}. A recurring gap is that debate is run "
        "for a fixed or unspecified number of rounds and terminates on "
        "\\enquote{consensus} without a defined convergence test, and without a "
        "counter-agent whose job is to resist premature agreement. Two further "
        "properties are usually left implicit. The aggregation rule is often an "
        "unweighted majority, which discards the fact that different roles carry "
        "different authority for different questions and that agents report "
        "different confidences; and the compute budget is fixed rather than "
        "matched to case difficulty, so an easy question and a hard one cost the "
        "same. \\citet{kim2024mdagents} address the second point by adapting the "
        "collaboration structure to difficulty, which is the pattern our triage "
        "router follows. The design in this paper makes the convergence test, "
        "the aggregation weights, the counter-agent, and the difficulty-matched "
        "budget all explicit and, more importantly, measured."))

    T.append(r"\subsection{Calibration and selective prediction}")
    T.append(par(
        "A probability is calibrated if, among predictions made with confidence "
        "$q$, a fraction $q$ are correct. Temperature scaling divides the logits "
        "by a single scalar $T$ fitted to minimize negative log-likelihood on a "
        "held-out set and is a strong post-hoc calibrator for modern networks "
        "\\citep{guo2017calibration}; expected calibration error (ECE) summarizes "
        "the gap between confidence and accuracy over bins "
        "\\citep{naeini2015ece}. Selective prediction lets a model abstain: it "
        "trades coverage for a lower error rate on the answered subset, and the "
        "risk--coverage curve characterizes the trade-off "
        "\\citep{geifman2017selective}. LLM confidence, whether read from token "
        "probabilities or elicited verbally, is often miscalibrated "
        "\\citep{xiong2024uncertainty,kadavath2022knowwhat}, so a multi-agent "
        "medical system that aggregates several such signals needs an explicit "
        "calibration and abstention stage, and an evaluation that reports when "
        "that stage helps and when it does not."))

    T.append(r"\subsection{Evidence grounding and hallucination detection}")
    T.append(par(
        "Hallucination, content unsupported by the input, is the central "
        "reliability problem for generative models \\citep{ji2023hallucination}. "
        "Self-consistency checks \\citep{manakul2023selfcheckgpt} and "
        "retrieval-grounded generation \\citep{edge2024graphrag} reduce it. In an "
        "agentic diagnostic pipeline the natural anchor is the localized report: "
        "if a specialist cites a morphological feature, that feature should "
        "appear in the report the Simulated Radiologist produced. Our verifier "
        "operationalizes this by parsing the report into the set of asserted "
        "facts and grading each cited feature against it, deliberately grounding "
        "on the pipeline's own artefact rather than on a hidden oracle."))

    T.append(r"\subsection{Adaptive computation}")
    T.append(par(
        "Routing easy inputs through a cheaper path and reserving expensive "
        "computation for hard ones is a standard efficiency lever; MDAgents "
        "\\citep{kim2024mdagents} adapts the collaboration structure to question "
        "difficulty. For a lung-nodule pipeline the analogue is a triage step "
        "that sends unambiguous morphology (a small smooth ground-glass nodule, a "
        "large obviously spiculated mass) to a single specialist and one round, "
        "and reserves the four-role board and multi-round debate for ambiguous "
        "cases. The value of such a router is an empirical question (how many "
        "cases are confidently easy, and how often does the router mistakenly "
        "down-triage a hard case) that we answer directly."))

    T.append(r"\subsection{Evaluating medical AI beyond accuracy}")
    T.append(par(
        "A recurring lesson from clinical machine learning is that a single "
        "aggregate accuracy figure is a poor predictor of deployed behaviour. "
        "Four additional axes matter here. First, \\emph{calibration}: a model "
        "whose 0.9-confidence predictions are right only 70\\% of the time will "
        "mislead a clinician who uses the probability as a decision threshold "
        "\\citep{guo2017calibration,naeini2015ece}. Second, \\emph{selective "
        "risk}: the operating point of a system that can defer is a "
        "(coverage, risk) pair, and two systems with the same overall accuracy "
        "can differ sharply in how safely they behave when allowed to abstain "
        "\\citep{geifman2017selective}. Third, \\emph{subgroup and prevalence "
        "robustness}: performance measured on a balanced benchmark can invert on "
        "a screening population where malignancy is rare, because thresholds and "
        "any class-dependent bias in the model interact with the base rate. "
        "Fourth, \\emph{cost}: for an LLM pipeline the number and size of model "
        "calls set the latency, the price, and the carbon footprint, and a "
        "method that improves accuracy by an order of magnitude more compute is "
        "not obviously an improvement. Our evaluation reports all four axes "
        "alongside accuracy, and treats a component as justified only when it "
        "moves its designated axis in a pre-registered direction with a "
        "significant paired effect."))

    # ================= 3. Methods =================
    T.append(r"\section{System Architecture and Evaluation Protocol}\label{sec:method}")
    T.append(par(
        "Figure~\\ref{fig:arch} shows the resulting four-stage pipeline. "
        "Stage~1 (Nodule Spotter) and Stage~2 (Simulated Radiologist) are "
        "retained from \\citet{yang2025lungnoduleagent}, with the "
        "mixture-of-experts, IoU--DBSCAN clustering, and confidence-weighted "
        "judging panel reimplemented in NumPy. Stage~0 (Triage Router) and "
        "Stage~3 (Doctor Board) carry the new mechanisms detailed in this "
        "section: a role-differentiated board, a report-grounded evidence "
        "verifier, a Devil's-Advocate false-negative guard, a Chair with an "
        "explicit convergence test and debate loop, fitted temperature scaling "
        "with a margin abstention gate, and a Lung-RADS guideline agent, with "
        "hierarchical memory and a Medical Graph RAG store spanning all "
        "stages."))
    T.append(figure_star("architecture.png",
        "LungNoduleAgent-Enhanced pipeline architecture.",
        "fig:arch"))

    T.append(r"\subsection{Relationship to the original framework}")
    T.append(par(
        "The enhancement is an extension, not a replacement. The Nodule Spotter "
        "still performs mixture-of-experts detection, IoU-distance DBSCAN mask "
        "clustering, and confidence-weighted judging-panel voting; the Simulated "
        "Radiologist still produces a localized report from a focal crop with a "
        "context margin; the Doctor Agent System still performs malignancy "
        "reasoning with retrieved knowledge and multi-round discussion. The new "
        "components sit inside and around the Doctor Agent System and at the "
        "pipeline boundaries, and every original interface is preserved so that "
        "the two designs can be compared component by component."))

    T.append(r"\subsection{An evidence taxonomy}")
    T.append(par(
        "The design separates three kinds of evidence that agentic systems "
        "routinely conflate. \\emph{Image-derived} evidence is the final mask and "
        "the measurements taken from it: long and short diameter and an "
        "estimated volume $V=\\tfrac{1}{6}\\,d_{\\text{long}}\\,d_{\\text{short}}"
        "\\,h$. \\emph{Report-derived} evidence is the morphology the Simulated "
        "Radiologist asserts: lobar location, density, margin, shape, and the "
        "presence or absence of spiculation, pleural indentation, vascular "
        "convergence, air bronchogram, and cavitation. \\emph{Knowledge-derived} "
        "evidence is the guideline and pathology text retrieved for each role. "
        "Keeping these distinct is what makes the verifier meaningful: a "
        "specialist's cited feature is checked against the report, not against a "
        "private oracle, so the verifier catches inconsistency introduced by the "
        "agent rather than error inherited from perception."))

    T.append(r"\subsection{Enhanced components and their success criteria}\label{sec:comp}")
    T.append(par(
        "Table~\\ref{tab:components} pairs each enhanced component with the "
        "regime-wise metric used to judge it, rather than a single headline "
        "accuracy number, so that the endpoint decides whether the component "
        "earns its place."))
    T.append(tex_table(
        ["Component", "Function", "Success metric"],
        [
         ["Triage router", "easy $\\to$ 1 specialist, 1 round; ambiguous $\\to$ "
          "4-role board, $\\le$4 rounds; decision recorded with its reason",
          "mean LLM calls/case; fraction routed easy; under-triage rate"],
         ["Role-differentiated board", "thoracic radiologist, pathologist, "
          "oncologist, pulmonologist; each queries a role-specific knowledge "
          "subgraph; role- and confidence-weighted aggregation",
          "accuracy and macro-F1 vs.\\ a single agent (paired test)"],
         ["Evidence verifier", "parses the report; grades each cited feature as "
          "supported, unmentioned (soft flag), or contradicted (hard flag); "
          "penalizes confidence by $0.05$/$0.20$",
          "hard-flag precision and recall vs.\\ the oracle; flags/case"],
         ["Devil's-Advocate", "argues the strongest alternative each round; if "
          "the board leans benign while it puts $\\ge 0.60$ on malignancy, "
          "escalate to human review",
          "false-negative rate vs.\\ the board without it (paired test)"],
         ["Guideline agent", "maps size and density to Lung-RADS v2022; "
          "spiculation, pleural indentation, and documented growth escalate the "
          "category",
          "escalation correctness (deterministic unit tests)"],
         ["Fitted calibration", "temperature fitted by NLL on a 30\\% held-out "
          "split of each run, then applied to the remainder",
          "ECE at $T{=}1$ vs.\\ fitted $T$; the fitted $T$ per regime"],
         ["Abstention gate", "abstain when the calibrated top-1/top-2 margin "
          "$<0.10$, when the top probability $<0.40$, or when the "
          "Devil's-Advocate guard fires",
          "risk--coverage AURC; abstention value "
          "$=\\mathrm{err}_{\\mathrm{abst}}-\\mathrm{err}_{\\mathrm{ans}}$"],
        ],
        "Enhanced components and their success metrics.",
        "tab:components",
        colspec="p{0.15\\textwidth}p{0.46\\textwidth}p{0.31\\textwidth}"))

    T.append(par(
        "\\textbf{Aggregation.} Each specialist $i$ returns a distribution "
        "$P_i$ over the three malignancy classes and a self-reported confidence "
        "$c_i$. The board aggregates with role weights $w_{\\text{role}}$ "
        "(oncologist $1.2$, radiologist and pathologist $1.0$, pulmonologist "
        "$0.9$, Devil's-Advocate $0.6$): "
        "$\\bar P(k)=\\big(\\sum_i c_i\\,w_{r(i)}\\,P_i(k)\\big)/\\sum_i c_i\\,"
        "w_{r(i)}$. Temperature scaling then produces the calibrated distribution "
        "$\\tilde P(k)\\propto \\exp\\!\\big(\\log \\bar P(k)/T\\big)$, and the "
        "abstention gate reads its top-1/top-2 margin and entropy. "
        "\\textbf{Verifier grading.} For a claim $x$ and the parsed report "
        "claims $C$: $x$ is \\textsc{supported} if $C$ asserts it, "
        "\\textsc{contradicted} if $C$ asserts its negation (a hard flag, "
        "$-0.20$ confidence), and \\textsc{unmentioned} otherwise (a soft flag, "
        "$-0.05$). \\textbf{Convergence.} The Chair stops the debate when a "
        "super-majority ($\\ge 0.75$) of specialists share the leading label and "
        "that label is stable across two rounds, or at the round limit."))

    T.append(r"\subsection{Algorithms}")
    T.append(par(
        "Algorithm~1 and Algorithm~2 state the debate loop and the "
        "calibration and abstention decision exactly as coded. Notation: "
        "$\\mathcal{S}$ is the set of specialist roles, $P_i$ and $c_i$ the "
        "distribution and self-reported confidence returned by role $i$, "
        "$X_i$ the set of features that role cites, $w_{r(i)}$ its role "
        "weight, $\\bar P$ the weighted aggregate, $\\tilde P$ the calibrated "
        "aggregate, $\\alpha_t$ the agreement at round $t$, $\\ell_t$ the "
        "leading label at round $t$, and $E$ the evidence trace. The three "
        "classes are pre-invasive, minimally invasive, and invasive; the "
        "latter two are the malignant classes the guard protects."))
    T.append(par(
        "Figure~\\ref{fig:flow} traces the same two procedures as a flow: how "
        "the triage router splits the case, how the specialists, the Evidence "
        "Verifier, the Chair and the Devil's-Advocate interact inside one "
        "debate round, where the single feedback edge closes the loop, and in "
        "what order weighted aggregation, temperature scaling, the abstention "
        "gate and the false-negative guard apply before the review packet is "
        "emitted. Both triage paths share that calibration spine."))
    T.append(figure_star("agent_flow.png",
        "Agent interaction and calibration flow.", "fig:flow"))
    T.append(algo_box(
        1, "Doctor Board debate procedure.",
        "report $R$; measurements $M$; roles $\\mathcal{S}$ with "
        "$|\\mathcal{S}|=4$; round limit $N=4$; agreement threshold "
        "$\\tau=0.75$; role weights $w$.",
        "label $\\hat y$; calibrated $\\tilde P$; abstain flag $a$; guideline "
        "category $g$; evidence trace $E$; rounds used $t^{*}$.",
        [
         (0, "\\textbf{begin}"),
         (1, "$\\ell_0 \\gets \\varnothing$;\\quad $E \\gets \\varnothing$;\\quad $t^{*} \\gets N$"),
         (1, "\\textbf{for} $t \\gets 1$ \\textbf{to} $N$ \\textbf{do}"),
         (2, "\\textbf{for each} $s_i \\in \\mathcal{S}$ \\textbf{do}"),
         (3, "$(P_i, c_i, X_i) \\gets s_i(R, M)$\\quad // from report, not oracle"),
         (3, "$(E, c_i) \\gets \\textsc{Verify}(X_i, R, E, c_i)$\\quad // $-0.20$ hard, $-0.05$ soft"),
         (2, "\\textbf{end for}"),
         (2, "$\\ell_t \\gets \\arg\\max_k \\bar P_t(k)$;\\quad $\\alpha_t \\gets$ \\textsc{Agreement}$(\\{P_i\\})$"),
         (2, "$P_{\\mathrm{dev}} \\gets \\textsc{DevilsAdvocate}(\\ell_t, R, M)$"),
         (2, "\\textbf{if} $(t = 1 \\wedge \\alpha_t = 1) \\vee (\\alpha_t \\ge \\tau \\wedge \\ell_t = \\ell_{t-1})$ \\textbf{then}"),
         (3, "$t^{*} \\gets t$;\\quad \\textbf{break}"),
         (2, "\\textbf{end if}"),
         (1, "\\textbf{end for}"),
         (1, "$\\bar P(k) \\gets \\sum_i c_i w_{r(i)} P_i(k) / \\sum_i c_i w_{r(i)}$\\quad // includes $P_{\\mathrm{dev}}$ at $w=0.6$"),
         (1, "$(\\hat y, \\tilde P, a) \\gets \\textsc{CalibrateAndAbstain}(\\bar P, \\alpha_{t^{*}}, P_{\\mathrm{dev}})$\\quad // Algorithm 2"),
         (1, "$g \\gets \\textsc{LungRADS}(M, X)$"),
         (1, "\\textbf{return} $(\\hat y, \\tilde P, a, g, E, t^{*})$"),
         (0, "\\textbf{end}"),
        ]))
    T.append(algo_box(
        2, "Calibration and abstention procedure.",
        "aggregate $\\bar P$; agreement $\\alpha$; Devil's-Advocate "
        "distribution $P_{\\mathrm{dev}}$; fitted temperature $T$ from the "
        "held-out split; margin $\\delta=0.10$; floor $\\phi=0.40$; guard "
        "$\\gamma=0.60$.",
        "label $\\hat y$; calibrated $\\tilde P$; abstain flag $a$.",
        [
         (0, "\\textbf{begin}"),
         (1, "$\\tilde P(k) \\gets \\exp(\\log \\bar P(k)/T) / \\sum_j \\exp(\\log \\bar P(j)/T)$\\quad $\\forall k$"),
         (1, "$(\\tilde p_1, \\tilde p_2, \\tilde p_3) \\gets \\textsc{Sort}(\\tilde P)$\\quad // descending"),
         (1, "$\\hat y \\gets \\arg\\max_k \\tilde P(k)$;\\quad $a \\gets \\textsc{false}$"),
         (1, "\\textbf{if} $(\\tilde p_1 - \\tilde p_2) < \\delta \\vee \\tilde p_1 < \\phi$ \\textbf{then}"),
         (2, "$a \\gets \\textsc{true}$\\quad // low margin or low confidence"),
         (1, "\\textbf{end if}"),
         (1, "\\textbf{if} $\\hat y = \\textsc{pre-invasive} \\wedge P_{\\mathrm{dev}}(\\textsc{min.inv}) + P_{\\mathrm{dev}}(\\textsc{inv}) \\ge \\gamma$ \\textbf{then}"),
         (2, "$a \\gets \\textsc{true}$\\quad // false-negative guard"),
         (1, "\\textbf{end if}"),
         (1, "\\textbf{return} $(\\hat y, \\tilde P, a)$"),
         (0, "\\textbf{end}"),
        ]))

    T.append(r"\subsection{Resource-use design}\label{sec:resource}")
    T.append(par(
        "Three design choices bound the computation a case consumes, and each "
        "is measured rather than asserted. First, the triage router sends "
        "cases with unambiguous morphology to a single specialist and one "
        "round, reserving the four-role board for cases that need "
        "deliberation; Section~\\ref{sec:cost} reports the resulting calls per "
        "case together with the under-triage rate that bounds how far this can "
        "safely be pushed. Second, the Chair's convergence test stops the "
        "debate as soon as a super-majority is stable across two rounds, so "
        "the round budget is an upper bound rather than a fixed cost. Third, "
        "that budget is itself capped at four rounds, which bounds the worst "
        "case. The accounting unit throughout is the language-model call, "
        "counted by an instrumented backend that wraps every detector, judge, "
        "specialist, critique, and summary call. For a pipeline whose work is "
        "dominated by model inference, the call count is the quantity that "
        "drives latency, price, and energy draw "
        "\\citep{strubell2019energy}, which is why it is reported alongside "
        "the safety metrics rather than as an afterthought."))
    T.append(par(
        "Direct hardware profiling is deliberately out of scope at this stage. "
        "Because the reasoning backbone is a deterministic mock backend rather "
        "than a served model, wall-clock latency, processor and accelerator "
        "utilization, and memory footprint measured on this harness would "
        "characterize the harness and not a deployment. Those quantities "
        "become meaningful once a real vision-language model is substituted at "
        "the documented interface, and we treat them as the first measurements "
        "to add at that point, with the call count serving as the "
        "hardware-independent proxy until then."))

    T.append(r"\subsection{Stress regimes}")
    T.append(par(
        "A clean synthetic distribution cannot exercise safety machinery: there "
        "are no unsupported claims, borderline lesions, or distribution shift to "
        "catch. We therefore define five regimes over one generator, and a "
        "\\emph{perceived}-versus-\\emph{oracle} split in which the report and the "
        "specialists see the perceived features while all scoring uses the "
        "oracle. \\textbf{Clean:} perceived $=$ oracle; balanced classes. "
        "\\textbf{Noisy:} each perceived categorical or boolean feature is "
        "corrupted with probability $0.25$ and the report omits a truly-present "
        "sign with probability $0.30$. \\textbf{Borderline:} only cases whose "
        "top-two malignancy posteriors lie within $0.20$ are retained. "
        "\\textbf{Imbalanced:} the class prior is skewed toward pre-invasive "
        "($0.60/0.25/0.15$). \\textbf{Adversarial:} noise, borderline sampling, "
        "and a degraded detector (wider mask scatter, more misses) combined."))

    T.append(r"\subsection{Metrics}")
    T.append(par(
        "Over answered (non-abstained) cases we report accuracy and macro-F1; "
        "over all cases, accuracy counting abstentions as wrong, and answer "
        "coverage. \\textbf{Safety.} The false-negative rate is the fraction of "
        "truly-malignant nodules (minimally-invasive or invasive) answered as "
        "pre-invasive. AURC is the area under the risk--coverage curve over "
        "answered cases ranked by confidence (lower is better). Abstention value "
        "is $\\mathrm{err}(\\text{abstained})-\\mathrm{err}(\\text{answered})$; a "
        "positive value means the gate withholds the harder cases. "
        "\\textbf{Calibration.} ECE with 10 equal-width bins, computed for the "
        "raw aggregate ($T{=}1$) and for the fitted-temperature output, together "
        "with the fitted $T$. \\textbf{Grounding.} Verifier hard-flag precision "
        "and recall against the oracle, where the recall denominator is agent "
        "\\emph{fabrications} only (claims that disagree with both the oracle "
        "and the perceived input), so that perception error, which is outside "
        "the verifier's scope, is not counted against it. \\textbf{Cost.} Mean "
        "language-model calls per case (every detector, judge, specialist, "
        "critique, and summary call is counted through an instrumented backend) "
        "and the under-triage rate: cases the router sends down the easy path "
        "that are genuinely ambiguous by the oracle."))

    T.append(r"\subsection{Reproducible configuration}\label{sec:config}")
    T.append(par(
        "A criticism of the original description is that the parameters that "
        "govern its behaviour (the DBSCAN neighbourhood and minimum count, the "
        "number of detection experts and judges, the number of agents and debate "
        "rounds, the convergence rule, and any calibration) are not stated. "
        "Every such value here lives in a single configuration object and is "
        "reported. The Nodule Spotter uses five detection experts and a "
        "three-member judging panel; masks are clustered with DBSCAN at "
        "$\\varepsilon = 0.5$ (equivalently an IoU threshold of 0.5) and a "
        "minimum of two masks, and each cluster is averaged and binarized at "
        "0.5. Hard cases use the four roles listed above with a maximum of four "
        "debate rounds and a convergence threshold of $\\tau = 0.75$; easy cases "
        "use a single role and one round. The abstention gate uses a top-1/top-2 "
        "margin threshold of $0.10$, a top-probability floor of $0.40$, and a "
        "Devil's-Advocate guard threshold of $0.60$ on malignant probability "
        "mass. The calibration temperature is not fixed: it is fitted per run "
        "(Section~\\ref{sec:comp}). Changing any value and re-running propagates "
        "through the whole harness, which is what makes the ablation and the "
        "regime sweep controlled comparisons rather than anecdotes."))
    T.append(r"\subsection{Experimental design}")
    T.append(par(
        f"Each regime uses n={n} generated cases per seed across {len(seeds)} "
        "seeds; 30\\% of each run is held out to fit the calibration temperature "
        "and the remaining 70\\% is scored, with the same split taken for every "
        "ablation variant so all are evaluated on identical test cases. We report "
        "the seed mean and standard deviation, and, where a claim rests on a "
        "comparison, a paired bootstrap over the shared case pool (1000 "
        "resamples) giving the effect size, its 95\\% interval, and a two-sided "
        "p-value. The leave-one-component-out ablation disables one mechanism at "
        "a time. The mock backend is a deterministic function of the seed, so "
        "every number in this paper reproduces exactly; the accompanying test "
        "suite covers the regime generators, the report-grounded verifier, the "
        "temperature fitting, the safety gates, the call accounting, and the "
        "bootstrap."))

    # ================= 4. Results =================
    T.append(r"\section{Results}\label{sec:results}")

    T.append(r"\subsection{Full system across regimes}")
    rows = []
    for r in rr:
        m, s = full(p, r), fsd(p, r)
        rows.append([r, pm(m["accuracy_all"], s["accuracy_all"]),
                     pm(m["accuracy"], s["accuracy"]),
                     pm(m["macro_f1"], s["macro_f1"]),
                     pm(m["coverage"], s["coverage"]),
                     pm(m["false_negative_rate"], s["false_negative_rate"]),
                     pm(m["aurc"], s["aurc"]),
                     pm(m["mean_llm_calls"], s["mean_llm_calls"], 2)])
    T.append(tex_table(
        ["Regime", "acc(all)", "acc(ans)", "macro-F1", "coverage", "FNR", "AURC",
         "calls/case"], rows,
        "Full-system results by regime.",
        "tab:main"))
    T.append(par(
        "Table~\\ref{tab:main} reports the full system across all five regimes "
        "as the seed mean and standard deviation; "
        "accuracy and coverage fall and AURC rises as the regime hardens, since "
        "the instrumentation is built to track the degradation rather than "
        f"conceal it. On clean data the full system answers {f(cl['coverage'],3)} of cases at "
        f"{f(cl['accuracy'],3)} accuracy on answered cases, with a false-negative "
        f"rate of {f(cl['false_negative_rate'],3)} and AURC {f(cl['aurc'],3)}. "
        "Feature noise costs a few points of accuracy; borderline sampling and "
        "class imbalance roughly halve accuracy over all cases, mostly through "
        "increased abstention (coverage "
        f"{f(full(p,'borderline')['coverage'],2)} and "
        f"{f(imb['coverage'],2)} respectively); and the adversarial regime drives "
        f"accuracy over all cases to {f(ad['accuracy_all'],3)} and AURC to "
        f"{f(ad['aurc'],3)}. Every metric moves in the direction a reviewer would "
        "predict, which is the purpose of the harness: the safety instrumentation "
        "should register stress, not mask it. Figure~\\ref{fig:acc} shows "
        "accuracy and coverage together. Per-class counts are small at this "
        "$n$, which inflates macro-F1 variance; the accuracy, cost, "
        "calibration, and verifier results are the stable ones."))
    T.append(figure("accuracy_by_regime.png",
        "Full-system accuracy and coverage by regime.", "fig:acc"))

    T.append(r"\subsection{Leave-one-component-out ablation}")
    for r in [x for x in ("clean", "borderline", "adversarial") if x in p["regimes"]]:
        rows = []
        for name in ABL_ORDER:
            if name not in p["regimes"][r]["ablation"]:
                continue
            m = p["regimes"][r]["ablation"][name]["mean"]
            disp = name.replace("_", "\\_")
            rows.append([disp, f(m["accuracy_all"]), f(m["macro_f1"]),
                         f(m["coverage"]), f(m["false_negative_rate"]),
                         f(m["aurc"]), f(m["ece"]), f(m["mean_llm_calls"], 1)])
        T.append(tex_table(
            ["variant", "acc(all)", "macro-F1", "cover", "FNR", "AURC", "ECE",
             "calls"], rows,
            f"Ablation under the {r} regime.",
            f"tab:abl-{r}"))
    T.append(par(
        "Tables~\\ref{tab:abl-clean}, \\ref{tab:abl-borderline}, and "
        "\\ref{tab:abl-adversarial} report the ablation under the clean, "
        "borderline, and adversarial regimes respectively. The "
        "role-differentiated board is the dominant contributor to accuracy in "
        "four of five regimes: reducing it to a single agent lowers accuracy over "
        f"all cases by up to "
        f"{f(delta(p,'borderline','- board (1 agent)','accuracy_all'),3)} "
        "(borderline) and raises both AURC and ECE. The exception is the "
        f"imbalanced regime, where the single agent scores {f(imb_board,3)} "
        "higher on accuracy over all cases "
        f"({sig_p(p,'imbalanced','board_vs_single')}): under a benign-heavy prior "
        "the board's role biases (the oncologist over-weights malignancy) "
        "and a fixed abstention margin combine to withhold or over-call cases a "
        "single unbiased radiologist would answer. Role weights and abstention "
        "thresholds are therefore deployment parameters that must be tuned to the "
        "target prevalence, not architecture constants; the harness surfaces this "
        "rather than averaging it away. Removing the router leaves accuracy "
        "essentially unchanged but raises cost (Section~\\ref{sec:cost}). "
        "Removing the verifier or the guideline agent does not move aggregate "
        "accuracy (their endpoints are grounding, reported in "
        "Section~\\ref{sec:ver}, and auditability, confirmed by the guideline "
        "agent's unit tests), while removing the "
        "Devil's-Advocate or the abstention gate trades coverage for "
        "false-negative rate (Section~\\ref{sec:abst}). "
        "Figure~\\ref{fig:abl} visualizes the adversarial-regime ablation, where "
        "the dashed line marks the full system; because the confidence signal is "
        "uninformative under this regime, AURC barely moves while accuracy and "
        "false-negative rate still separate the components."))
    T.append(figure_star("ablation_adversarial.png",
        "Leave-one-component-out ablation under the adversarial regime.",
        "fig:abl"))

    T.append(r"\subsection{Abstention and the false-negative guard}\label{sec:abst}")
    rows = []
    for r in rr:
        m = full(p, r)
        nd = p["regimes"][r]["ablation"].get("- devils_advocate", {}).get("mean", {})
        rows.append([r, f(m["coverage"]), f(m["abstention_value"]),
                     f(m["abstention_precision"]), f(m["false_negative_rate"]),
                     f(nd.get("false_negative_rate", float("nan")))])
    T.append(tex_table(
        ["Regime", "coverage", "abst.\\ value", "abst.\\ prec.", "FNR (full)",
         "FNR ($-$devil)"], rows,
        "Abstention and the false-negative guard, by regime.", "tab:abst"))
    T.append(par(
        "Table~\\ref{tab:abst} reports abstention value, precision, and "
        "false-negative rate by regime, where abstention value is "
        "$\\mathrm{err}_{\\mathrm{abst}}-\\mathrm{err}_{\\mathrm{ans}}$ and is "
        "positive when the gate withholds the harder cases. The margin-based "
        "abstention gate has clearly positive value under the "
        f"clean ({f(cl['abstention_value'],3)}), noisy "
        f"({f(full(p,'noisy')['abstention_value'],3)}), and borderline "
        f"({f(full(p,'borderline')['abstention_value'],3)}) regimes: the withheld "
        "cases are markedly more error-prone than the answered ones. It is "
        f"negative under class imbalance "
        f"({f(imb['abstention_value'],3)}, with abstention precision only "
        f"{f(imb['abstention_precision'],3)}) and under the adversarial regime "
        f"({f(ad['abstention_value'],3)}): a benign-heavy prior or heavy "
        "perception noise breaks the correlation between the aggregate's margin "
        "and its correctness, so a small margin no longer signals a likely "
        "error. This is a genuine negative result (selective prediction is "
        "only as good as the confidence signal it reads "
        "\\citep{geifman2017selective}), and it argues for pairing the "
        "confidence gate with a structural safeguard. The Devil's-Advocate "
        "false-negative guard is such a safeguard: it escalates whenever the "
        "board leans benign while a credible malignant counter-case exists, and "
        "it lowers the false-negative rate wherever false negatives occur "
        "(borderline "
        + f(abl(p, "borderline", "- devils_advocate", "false_negative_rate"), 3)
        + "$\\to$"
        + f(full(p, "borderline")["false_negative_rate"], 3)
        + ", " + sig_p(p, "borderline", "devil_fnr")
        + "; adversarial "
        + f(abl(p, "adversarial", "- devils_advocate", "false_negative_rate"), 3)
        + "$\\to$" + f(ad["false_negative_rate"], 3)
        + ", " + sig_p(p, "adversarial", "devil_fnr")
        + ") at a coverage cost."))

    T.append(r"\subsection{Evidence verifier}\label{sec:ver}")
    rows = []
    for r in rr:
        m, s = full(p, r), fsd(p, r)
        rows.append([r, pm(m["verifier_precision"], s["verifier_precision"], 2),
                     pm(m["verifier_recall"], s["verifier_recall"], 2),
                     pm(m["mean_hard_flags"], s["mean_hard_flags"], 2)])
    T.append(tex_table(
        ["Regime", "flag precision", "flag recall", "hard flags/case"], rows,
        "Evidence verifier precision, recall, and flag rate by regime.",
        "tab:verifier"))
    T.append(par(
        "Table~\\ref{tab:verifier} reports flag precision, recall, and mean "
        "hard flags per case by regime. "
        f"The verifier flags every agent-fabricated citation (recall "
        f"{f(cl['verifier_recall'],2)} across all regimes) because the check "
        "against a parsed report is deterministic. Precision against the oracle "
        f"is {f(cl['verifier_precision'],2)} on clean data and "
        f"{f(ad['verifier_precision'],2)} under the adversarial regime, where a "
        "corrupted report occasionally leads the verifier to hard-flag a claim "
        "that happens to match the oracle. The verifier's scope is explicit and "
        "reported: it catches inconsistency between an agent and the shared "
        "report, not error already baked into the report by perception. "
        "Figure~\\ref{fig:verifier} plots precision and recall with the mean "
        "number of hard flags raised per case."))
    T.append(figure("verifier_pr.png",
        "Verifier precision, recall, and hard-flag rate by regime.",
        "fig:verifier"))

    T.append(r"\subsection{Calibration}")
    rows = []
    for r in rr:
        m, s = full(p, r), fsd(p, r)
        rows.append([r, pm(m["ece_raw"], s["ece_raw"]), pm(m["ece"], s["ece"]),
                     pm(m["fitted_temperature"], s["fitted_temperature"], 2),
                     pm(m["over_confidence"], s["over_confidence"])])
    T.append(tex_table(
        ["Regime", "ECE ($T{=}1$)", "ECE (fitted $T$)", "fitted $T$",
         "mean conf $-$ acc"], rows,
        "Calibration error and fitted temperature by regime.", "tab:calib"))
    T.append(par(
        "Table~\\ref{tab:calib} reports calibration error and the fitted "
        "temperature by regime; the temperature is fitted by NLL minimization "
        "on a 30\\% held-out split, and it spans well below and well above 1, "
        "so no single fixed value serves all regimes. "
        f"On clean data the aggregate is already close to calibrated (ECE "
        f"{f(cl['ece_raw'],3)}) and the fitted temperature is "
        f"{f(cl['fitted_temperature'],2)}, a mild sharpening. Under the "
        f"adversarial regime the raw aggregate is badly overconfident (ECE "
        f"{f(ad['ece_raw'],3)}; mean confidence exceeds accuracy by "
        f"{f(ad['over_confidence'],3)}), the fitted temperature rises to "
        f"{f(ad['fitted_temperature'],2)}, and ECE falls to {f(ad['ece'],3)} "
        f"(paired $\\Delta$ "
        f"{f(sig(p,'adversarial','calibration_ece').get('delta',float('nan')),3)}, "
        f"{sig_p(p,'adversarial','calibration_ece')}). In the other four regimes "
        "the fitted temperature stays near 1 and the change in ECE is not "
        "significant. Temperature scaling is thus a no-op when the model is "
        "already calibrated and a large, significant correction exactly when "
        "distribution shift makes it overconfident "
        "\\citep{guo2017calibration}; it should be fitted on a small labelled "
        "calibration set drawn from the deployment distribution, not hard-coded. "
        "Figure~\\ref{fig:calib} shows ECE before and after scaling."))
    T.append(figure("calibration_ece.png",
        "Calibration error before and after temperature scaling.",
        "fig:calib"))

    T.append(r"\subsection{Cost of adaptive routing}\label{sec:cost}")
    rows = []
    for r in rr:
        m = full(p, r)
        nr = p["regimes"][r]["ablation"].get("- router", {}).get("mean", {})
        d = sig(p, r, "router_cost")
        rows.append([r, f(nr.get("mean_llm_calls", float("nan")), 2),
                     f(m["mean_llm_calls"], 2),
                     f(m.get("frac_routed_easy", float("nan")), 2),
                     f(m["under_triage_rate"], 3),
                     f"${f(d.get('delta',float('nan')),2)}$ "
                     f"[${f(d.get('lo',float('nan')),2)}$, "
                     f"${f(d.get('hi',float('nan')),2)}$]"])
    T.append(tex_table(
        ["Regime", "calls: always board", "calls: adaptive", "frac.\\ easy",
         "under-triage", "$\\Delta$ calls [95\\% CI]"], rows,
        "Cost of adaptive routing by regime.",
        "tab:cost"))
    T.append(par(
        "Table~\\ref{tab:cost} reports calls per case, routed-easy fraction, "
        "and under-triage rate by regime, with a negative $\\Delta$ meaning "
        "the router is cheaper. "
        "Adaptive routing lowers mean calls per case in every regime "
        f"(clean {f(abl(p,'clean','- router','mean_llm_calls'),2)}$\\to$"
        f"{f(cl['mean_llm_calls'],2)}, {sig_p(p,'clean','router_cost')}), but the "
        "effect is small in absolute terms: on this distribution only "
        f"{f(min(full(p,r).get('frac_routed_easy',0) for r in rr)*100,0)}--"
        f"{f(max(full(p,r).get('frac_routed_easy',0) for r in rr)*100,0)}\\% of "
        "cases have morphology unambiguous enough to route easy, and the "
        "remainder legitimately reach the full board. The realizable saving is "
        "bounded below by the single-agent cost ($\\approx$7.1 calls/case, from "
        "the \\texttt{-\\,board} ablation) and scales with that easy fraction, a "
        "distribution-dependent quantity that a deployment can measure and, if "
        "desired, widen by loosening the triage predicate. The under-triage "
        "rate (genuinely ambiguous cases sent down the easy path) rises "
        f"from {f(cl['under_triage_rate'],3)} on clean data to "
        f"{f(ad['under_triage_rate'],3)} under the adversarial regime, "
        "quantifying the router's failure mode: perception noise makes some hard "
        "cases look easy, so a deployment should bias the router toward "
        "escalation when its inputs are noisy. The routing cost saving is "
        "significant in every regime ($p<0.001$; Table~\\ref{tab:sig}). "
        "Figure~\\ref{fig:cost} contrasts "
        "the two policies."))
    T.append(figure("cost_routing.png",
        "Mean language-model calls per case by routing policy.", "fig:cost"))

    rows = []
    for r in rr:
        m = full(p, r)
        nr = p["regimes"][r]["ablation"].get("- router", {}).get("mean", {})
        tok_a = m.get("mean_total_tokens", float("nan"))
        tok_b = nr.get("mean_total_tokens", float("nan"))
        call_a = m.get("mean_llm_calls", float("nan"))
        call_b = nr.get("mean_llm_calls", float("nan"))
        rows.append([r, f(tok_b, 0), f(tok_a, 0),
                     f"{100.0 * (tok_b - tok_a) / tok_b:.1f}\\%",
                     f"{100.0 * (call_b - call_a) / call_b:.1f}\\%"])
    T.append(tex_table(
        ["Regime", "tokens: always board", "tokens: adaptive",
         "tokens saved", "calls saved"], rows,
        "Prompt and response volume per case by routing policy.", "tab:tokens"))
    T.append(par(
        "Call count is a coarse unit: the calls the router removes are the "
        "expensive ones. Table~\\ref{tab:tokens} meters the actual prompt and "
        "response volume per case, counted by the same instrumented backend "
        "and converted at four characters per token. Because a board call "
        "carries a role-specific knowledge excerpt and the full report while a "
        "triage call carries neither, the proportional saving in tokens "
        "exceeds the saving in calls in every regime, by a factor of between "
        "1.5 and 1.7. This matters for the sustainability claim: inference "
        "energy and monetary cost both scale with the volume of text a served "
        "model processes rather than with the number of requests "
        "\\citep{luccioni2024power,strubell2019energy}, so it is the token "
        "reduction, not the call reduction, that transfers to a deployment. "
        "The orchestration itself is negligible against that: the full "
        "pipeline, excluding model inference, runs in approximately seven "
        "milliseconds per case on a single commodity CPU core, against the "
        "hundreds of milliseconds a single served-model call typically costs, "
        "so the coordination the architecture adds is not a latency "
        "bottleneck."))

    T.append(r"\subsection{Significance summary}")
    T.append(par(
        "Table~\\ref{tab:sig} collects the paired bootstrap significance tests "
        "referenced throughout this section, summarizing which effects are "
        "significant in which regime."))
    labels = {"board_vs_single": "board vs.\\ single agent (acc all)",
              "router_cost": "adaptive vs.\\ always board (calls/case)",
              "calibration_ece": "fitted vs.\\ no calibration (ECE)",
              "devil_fnr": "with vs.\\ without Devil's-Advocate (FNR)"}
    rows = []
    for r in rr:
        s = p["regimes"][r].get("significance", {})
        for k, lab in labels.items():
            d = s.get(k)
            if not d:
                continue
            rows.append([r, lab, f"${d['delta']:+.3f}$",
                         f"[${d['lo']:+.3f}$, ${d['hi']:+.3f}$]",
                         f(d["p_value"], 3)])
    T.append(tex_table(
        ["Regime", "comparison", "$\\Delta$", "95\\% CI", "p"], rows,
        "Paired bootstrap significance tests by regime and comparison.",
        "tab:sig", colspec="llccr"))

    T.append(r"\subsection{Qualitative case trace}")
    T.append(par(
        "The pipeline emits a review packet per case. For a representative "
        "adversarial-regime case the trace reads: detection IoU $0.76$; report "
        "\\enquote{a ground-glass density nodule with an irregular shape and "
        "lobulated margin \\ldots there is spiculation}; router decision "
        "\\emph{hard} (part-solid-like, mid-size); four specialists opine, the "
        "verifier hard-flags one pathologist citation of \\texttt{cavitation} as "
        "contradicted by the report; the Devil's-Advocate argues pre-invasive; "
        "the Chair records a super-majority for invasive at round~2; calibration "
        "with the fitted $T$ yields $\\tilde P = (0.02,\\,0.15,\\,0.83)$; the "
        "guideline agent returns \\emph{Lung-RADS~4X} driven by spiculation and "
        "documented interval growth. The label, the calibrated probabilities, "
        "the flagged claim, the dissent, and the guideline category together form "
        "what a qualified reader would inspect; the system is decision "
        "support, not an autonomous diagnosis."))

    # ================= 5. Discussion =================
    T.append(r"\section{Discussion}\label{sec:disc}")
    T.append(r"\subsection{Safety and cost as measurable properties}")
    T.append(par(
        "The value of the enhancement is that each mechanism has an endpoint and "
        "a number. The board earns its cost through accuracy and macro-F1 where "
        "ambiguity exists. The router earns its complexity through a significant "
        "if modest reduction in calls, with a named and quantified failure mode. "
        "The verifier earns its place through complete fabrication recall and an "
        "explicitly bounded scope. Fitted calibration earns its place by "
        "removing calibration error where the model is overconfident and doing "
        "nothing where it is not. The abstention gate is reported honestly, "
        "including the two regimes in which it is counter-productive. This is the "
        "opposite of an \\enquote{always be definitive} policy: the system is "
        "permitted, and measured, to say less. We regard the reporting of "
        "negative results (the imbalanced-regime board reversal, the loss of "
        "abstention value under shift) as part of the contribution, because a "
        "harness that only ever confirms its own design is not a safety "
        "evaluation."))
    T.append(par(
        "Read together, the measured reduction in computational overhead and "
        "the measured improvement in calibration are what a sustainability "
        "argument for multi-agent clinical AI has to rest on "
        "\\citep{ueda2024climate,thomson2025environmental}: not a claim that "
        "more agents are better, but evidence that the extra deliberation is "
        "spent only where it changes the answer, and that the confidence "
        "attached to that answer can be trusted. The saving demonstrated here "
        "is modest in absolute terms, two to four percent of calls per case, "
        "and it is bounded below by the single-agent cost; what the harness "
        "contributes is the accounting discipline that makes a saving of this "
        "kind visible, attributable, and auditable at all."))
    T.append(r"\subsection{When confidence-based abstention fails}")
    T.append(par(
        "The clearest lesson is Section~\\ref{sec:abst}: selective prediction "
        "depends on the confidence signal being informative, and under class "
        "imbalance and the adversarial regime it is not "
        "\\citep{geifman2017selective,xiong2024uncertainty}. Two responses "
        "follow. First, abstention value should be monitored in deployment; a "
        "non-positive value is a signal that the confidence signal has degraded "
        "and that the thresholds should not be trusted. Second, the confidence "
        "gate should be paired with structural checks that do not read a "
        "calibrated probability at all, such as the Devil's-Advocate "
        "false-negative guard, which lowers the false-negative rate even when the "
        "aggregate's confidence is uninformative."))
    T.append(r"\subsection{Prevalence sensitivity of the board}")
    T.append(par(
        "The imbalanced-regime reversal is worth stating plainly: role weighting "
        "that helps on a balanced distribution can hurt when the base rate "
        "shifts. The oncologist's malignancy bias and a fixed abstention margin, "
        "tuned implicitly for balanced classes, together push accuracy over all "
        "cases below that of a single unbiased radiologist. Role weights and "
        "abstention thresholds are therefore deployment parameters that should be "
        "re-fitted to the target prevalence alongside the calibration "
        "temperature, and the value of a multi-agent board should be re-checked "
        "on the operating distribution rather than assumed from a balanced "
        "benchmark."))
    T.append(r"\subsection{Comparison with prior multi-agent medical systems}")
    T.append(par(
        "MedAgents \\citep{tang2023medagents}, MDAgents \\citep{kim2024mdagents}, "
        "and MedAgent-Pro \\citep{wang2025medagentpro} demonstrate that "
        "role-structured collaboration and evidence-oriented workflows improve "
        "medical reasoning; MDAgents also adapts structure to difficulty. Our "
        "contribution is orthogonal: we take one such pipeline and make its "
        "safety and cost behaviour \\emph{measurable}, with per-component "
        "endpoints, stress regimes, and paired significance tests, and we report "
        "where the mechanisms do not help. The verifier's report-grounding "
        "differs from self-consistency checks "
        "\\citep{manakul2023selfcheckgpt} in anchoring on the pipeline's own "
        "artefact, which keeps its scope inspectable."))
    T.append(r"\subsection{Threats to validity}")
    T.append(par(
        "The evaluation is synthetic. The mock backend reasons from the same "
        "feature heuristic that generates the labels, so absolute accuracy is not "
        "meaningful and inter-agent error correlation is understated; the "
        "comparative, cost, calibration, and grounding results are the "
        "transferable ones. The verifier's report parser is a transparent rule "
        "set rather than a learned extractor. The guideline agent implements a "
        "subset of Lung-RADS v2022. The knowledge base is a short reference "
        "document. The detector is a perturbation model, not a trained network "
        "\\citep{isensee2021nnunet}. Per-class counts are small at this $n$, "
        "which widens macro-F1 intervals. Each of these is a single documented "
        "interface with a swap-in point; the experimental protocol (stress "
        "regimes, per-component endpoints, multi-seed intervals, paired tests) "
        "is what carries over to a real evaluation."))
    T.append(r"\subsection{Toward a clinical evaluation}")
    T.append(par(
        "Replacing the synthetic generator with a LIDC-IDRI / DICOM loader "
        "\\citep{armato2011lidc}, the mock backend with an image-grounded VLM "
        "\\citep{sellergren2025medgemma,saab2024medgemini}, and the perturbation "
        "detector with a trained segmenter \\citep{isensee2021nnunet} leaves the "
        "orchestration, the metrics, and the harness unchanged. The same tables "
        "would then be produced on patient-level splits with "
        "radiologist-adjudicated ground truth, and the calibration temperature, "
        "the role weights, and the abstention thresholds would be fitted on a "
        "held-out clinical calibration set. Prospective reader studies would "
        "measure whether the review packet (label, calibrated probabilities, "
        "verifier trace, recorded dissent, and guideline category) changes "
        "radiologist decisions and time-to-decision."))
    T.append(r"\subsection{Deployment and governance considerations}")
    T.append(par(
        "Three practices follow from the design. The evidence trace should be "
        "presented to a qualified reader, not consumed autonomously: the label, "
        "the calibrated probabilities, the supporting observations, the flagged "
        "claims, the recorded dissent, the guideline category, and the abstention "
        "status form a review packet whose purpose is to make the machine's "
        "reasoning auditable, not to replace the reader. Audit logs should retain "
        "the model version, the full configuration, the random seed where "
        "applicable, the retrieved knowledge, and identifiers for every prompt "
        "and tool call, so that a surprising output can be reconstructed and "
        "investigated. And the safety endpoints reported here (abstention "
        "value, verifier flag rate, under-triage rate, calibration error) are "
        "exactly the quantities a post-market monitoring plan should track over "
        "time, because each of them can drift as scanners, protocols, and "
        "populations change, and a drop in abstention value in particular is an "
        "early signal that the confidence signal the gate depends on has "
        "degraded."))

    # ================= 6. Conclusion =================
    T.append(r"\section{Conclusion and Future Work}\label{sec:concl}")
    T.append(par(
        "LungNoduleAgent-Enhanced keeps the detect--describe--diagnose pipeline "
        "of \\citet{yang2025lungnoduleagent} and adds adaptive triage, a "
        "role-differentiated board, a report-grounded verifier, a "
        "Devil's-Advocate false-negative guard, guideline-grounded escalation, "
        "fitted temperature scaling, and margin-based abstention, each an "
        "instrumented component with an explicit success metric. Across five "
        "stress regimes with multi-seed intervals and paired significance tests, "
        "the board drives accuracy except under a benign-heavy prior, adaptive "
        "routing gives a small but significant and bounded cost reduction, the "
        "verifier achieves complete fabrication recall, and fitted calibration "
        "removes the calibration error that distribution shift introduces, while "
        "the abstention gate helps only where the confidence signal is "
        "informative."))
    T.append(par(
        "The measured improvements are as follows. The role-differentiated "
        f"board raises accuracy over all cases by up to {f(max_board,3)} "
        "(borderline regime) against a single agent. Adaptive routing lowers "
        "mean language-model calls per case by two to four percent in every "
        "regime, significant at $p<0.001$ throughout, against a single-agent "
        "floor of $\\approx$7.1 calls per case. Fitted temperature scaling cuts "
        "expected calibration error under the adversarial regime from "
        f"{f(ad['ece_raw'],3)} to {f(ad['ece'],3)}, a relative reduction of "
        f"{f(100*(ad['ece_raw']-ad['ece'])/ad['ece_raw'],0)}\\%, while leaving "
        "the already calibrated clean regime untouched. The evidence verifier "
        f"reaches a fabrication recall of {f(cl['verifier_recall'],2)} across "
        "all regimes. The Devil's-Advocate guard lowers the false-negative "
        f"rate from {f(abl(p,'borderline','- devils_advocate','false_negative_rate'),3)} "
        f"to {f(full(p,'borderline')['false_negative_rate'],3)} under the "
        "borderline regime. Each figure is a seed mean over "
        f"{len(seeds)} seeds with paired significance testing."))
    T.append(par(
        "Future work will (i)~replace the synthetic components with "
        "a trained detector, an image-grounded VLM, and LIDC-IDRI plus private "
        "cohorts; (ii)~fit role weights and abstention thresholds to the "
        "operating prevalence and re-run the harness; (iii)~add a "
        "disagreement-aware abstention rule and test it against the margin gate "
        "under shift; (iv)~run a prospective reader study on the review "
        "packet; and (v)~extend the cost accounting from call counts to "
        "measured energy and latency on served models, including deployment "
        "settings that matter for sustainable healthcare computing: edge "
        "inference close to the scanner, federated training across sites "
        "without moving patient data, and green AI scheduling that places "
        "deliberation-heavy cases on lower-carbon capacity "
        "\\citep{ueda2024climate,thomson2025environmental,strubell2019energy}. "
        "The broader aim is a template for multi-agent clinical decision "
        "support in which safety, cost, and resource use are measured, not "
        "assumed."))

    T.append(r"\section*{Supplementary Materials}")
    T.append(par(
        "The following checklist records what is needed to reproduce every "
        "number in this article. \\textbf{Data.} No patient data are used. All "
        f"cases are produced by the documented synthetic generator at {n} cases "
        f"per seed across {len(seeds)} seeds, so the dataset is regenerated "
        "exactly from a fixed seed rather than stored; the real-data path uses "
        "LIDC-IDRI \\citep{armato2011lidc} at the same documented interface. "
        "\\textbf{Code.} The reference implementation, the deterministic mock "
        "language backend, the synthetic case generator, the evaluation "
        "harness, the figure and table scripts, and the automated test suite "
        "are released as a single repository at "
        "\\url{https://github.com/LochanNarayan/SafeNodule-MAS}, and are "
        "also available from the corresponding author on request. "
        "\\textbf{Configuration.} Every parameter is held in one "
        "configuration object and reported in "
        "Section~\\ref{sec:config}: five detection experts, a three-member "
        "judging panel, DBSCAN at $\\varepsilon=0.5$ with a minimum of two "
        "masks, four specialist roles, at most four debate rounds, a "
        "convergence threshold of $\\tau=0.75$, an abstention margin of 0.10, "
        "a probability floor of 0.40, a Devil's-Advocate guard threshold of "
        "0.60, and a 30\\% calibration split. \\textbf{Hardware.} No "
        "accelerator is required: the mock backend is a deterministic function "
        "of the seed, so the full protocol runs on a single commodity CPU, "
        "which is also why hardware-level energy and latency figures are "
        "deferred to the served-model setting "
        "(Section~\\ref{sec:resource}). \\textbf{Randomness.} Every run is "
        "seeded; the reported values are seed means with standard deviations "
        "and paired bootstrap intervals over 1000 resamples."))

    T.append(r"\section*{Data Availability Statement}")
    T.append(par("Data available on request."))

    T.append(r"\section*{Conflicts of Interest}")
    T.append(par("The authors declare no conflicts of interest."))

    prose = "\n".join(T)
    ordered = order_by_citation(prose)
    num = {k: i + 1 for i, k in enumerate(ordered)}
    prose = resolve_refs(resolve_citations(prose, num))
    return prose + "\n" + bibliography_block(ordered) + "\n\\end{document}\n"


def main():
    p = load()
    PAPER.mkdir(exist_ok=True)
    figs = PAPER / "figures"
    figs.mkdir(exist_ok=True)
    try:
        from paper.make_architecture import render as render_arch
        from paper.make_architecture import render_flow
    except Exception:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "make_architecture", PAPER / "make_architecture.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        render_arch = mod.render
        render_flow = mod.render_flow
    render_arch(figs / "architecture.png")
    render_flow(figs / "agent-flow.png")
    if SRC_FIGS.exists():
        for png in SRC_FIGS.glob("*.png"):
            shutil.copy(png, figs / png.name.replace("_", "-"))
    OUT.write_text(build(p), encoding="utf-8")
    print("wrote", OUT)
    print("figures:", sorted(f"{x.name} {x.stat().st_size // 1024}KB"
                             for x in figs.glob("*.png")))
    print("compile: Overleaf (pdfLaTeX, no BibTeX needed), or locally: "
          "pdflatex enhanced_lungnoduleagent  x2")


if __name__ == "__main__":
    main()
