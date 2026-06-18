import os
# Mandatory configuration for serverless Matplotlib execution on Vercel
os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file
from google import genai
from google.genai import types
import fitz  # PyMuPDF
import pandas as pd
import jinja2
import bibtexparser
import zipfile
import json
import io
import re
import base64
from datetime import datetime

app = Flask(__name__)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================
GEMINI_MODEL = "gemini-2.5-flash-lite"
SYSTEM_INSTRUCTION_BASE = (
    "You are an academic writing assistant. You organize, refine, and format "
    "real research content provided by the user. You NEVER invent facts, "
    "data, citations, or findings that were not provided. If information is "
    "missing, say so explicitly instead of making it up. Respond in plain "
    "text unless explicitly asked for JSON."
)

BIB_STYLE_MAP = {
    "IEEE": "IEEEtran",
    "ACM": "ACM-Reference-Format",
    "Springer": "spmpsci",
    "Elsevier": "elsarticle-num",
    "MDPI": "mdpi",
    "Generic": "plain",
}

# ============================================================================
# LATEX ESCAPE & TEMPLATE JINJA SETUP
# ============================================================================
def latex_escape(text):
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)

def table_to_latex(table_obj):
    title = table_obj.get("title", "Table")
    label = table_obj.get("label", "table")
    headers = table_obj.get("headers", [])
    rows = table_obj.get("rows", [])
    
    if not headers:
        return "% Empty table skipped.\n"
        
    col_spec = "l" * len(headers)
    header_str = " & ".join(str(h) for h in headers) + r" \\"
    body_rows = [" & ".join(str(cell) for cell in row) + r" \\" for row in rows]
    body_str = "\n".join(body_rows)
    safe_label = re.sub(r"\s+", "_", (label or "table").lower())
    
    return f"""\\begin{{table}}[h]
\\centering
\\caption{{{latex_escape(title)}}}
\\label{{tab:{safe_label}}}
\\begin{{tabular}}{{{col_spec}}}
\\hline
{header_str}
\\hline
{body_str}
\\hline
\\end{{tabular}}
\\end{{table}}"""

latex_jinja_env = jinja2.Environment(
    block_start_string="<%",
    block_end_string="%>",
    variable_start_string="<<",
    variable_end_string=">>",
    comment_start_string="<#",
    comment_end_string="#>",
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.BaseLoader(),
)

# Template mappings preserved exactly from source truth
IEEE_TEMPLATE = r"""\documentclass[conference]{IEEEtran}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}
\begin{document}
\title{<< title >>}
\author{
<% for author in authors %>
\IEEEauthorblockN{<< author.name >>}
\IEEEauthorblockA{<< author.department >>, << author.affiliation >>\\
Email: << author.email >>}<% if not loop.last %>\and <% endif %>
<% endfor %>
}
\maketitle
\begin{abstract}
<< abstract >>
\end{abstract}
\begin{IEEEkeywords}
<< keywords >>
\end{IEEEkeywords}
\section{Introduction}
<< introduction >>
\section{Methodology}
<< methodology >>
\section{Results}
<< results >>
<% for table in tables %>
<< table.latex >>
<% endfor %>
<% for figure in figures %>
\begin{figure}[h]
\centering
\includegraphics[width=0.8\columnwidth]{figures/<< figure.filename >>}
\caption{<< figure.caption >>}
\label{fig:<< figure.number >>}
\end{figure}
<% endfor %>
\section{Discussion}
<< discussion >>
\section{Conclusion}
<< conclusion >>
<% if future_work %>
\section{Future Work}
<< future_work >>
<% endif %>
<% if limitations %>
\section{Limitations}
<< limitations >>
<% endif %>
\bibliographystyle{<< bib_style >>}
\bibliography{references}
\end{document}"""

ACM_TEMPLATE = r"""\documentclass[sigconf]{acmart}
\usepackage{graphicx}
\begin{document}
\title{<< title >>}
<% for author in authors %>
\author{<< author.name >>}
\affiliation{
  \institution{<< author.affiliation >>}
  \department{<< author.department >>}
}
\email{<< author.email >>}
<% endfor %>
\begin{abstract}
<< abstract >>
\end{abstract}
\keywords{<< keywords >>}
\maketitle
\section{Introduction}
<< introduction >>
\section{Methodology}
<< methodology >>
\section{Results}
<< results >>
<% for table in tables %>
<< table.latex >>
<% endfor %>
<% for figure in figures %>
\begin{figure}[h]
\centering
\includegraphics[width=0.8\linewidth]{figures/<< figure.filename >>}
\caption{<< figure.caption >>}
\label{fig:<< figure.number >>}
\end{figure}
<% endfor %>
\section{Discussion}
<< discussion >>
\section{Conclusion}
<< conclusion >>
<% if future_work %>
\section{Future Work}
<< future_work >>
<% endif %>
<% if limitations %>
\section{Limitations}
<< limitations >>
<% endif %>
\bibliographystyle{<< bib_style >>}
\bibliography{references}
\end{document}"""

SPRINGER_TEMPLATE = r"""\documentclass[smallcondensed]{svjour3}
\usepackage{graphicx}
\usepackage{amsmath}
\journalname{<< target_journal >>}
\begin{document}
\title{<< title >>}
\author{
<% for author in authors %>
<< author.name >><% if not loop.last %> \and <% endif %>
<% endfor %>
}
\institute{
<% for author in authors %>
<< author.affiliation >>, << author.department >> \email{<< author.email >>}\\
<% endfor %>
}
\date{}
\maketitle
\begin{abstract}
<< abstract >>
\keywords{<< keywords >>}
\end{abstract}
\section{Introduction}
<< introduction >>
\section{Methodology}
<< methodology >>
\section{Results}
<< results >>
<% for table in tables %>
<< table.latex >>
<% endfor %>
<% for figure in figures %>
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/<< figure.filename >>}
\caption{<< figure.caption >>}
\label{fig:<< figure.number >>}
\end{figure}
<% endfor %>
\section{Discussion}
<< discussion >>
\section{Conclusion}
<< conclusion >>
<% if future_work %>
\section{Future Work}
<< future_work >>
<% endif %>
<% if limitations %>
\section{Limitations}
<< limitations >>
<% endif %>
\bibliographystyle{<< bib_style >>}
\bibliography{references}
\end{document}"""

ELSEVIER_TEMPLATE = r"""\documentclass[preprint,12pt]{elsarticle}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{lineno}
\journal{<< target_journal >>}
\begin{document}
\begin{frontmatter}
\title{<< title >>}
<% for author in authors %>
\author<% if author.corresponding %>[cor]<% endif %>{<< author.name >>}
\address{<< author.affiliation >>, << author.department >>}
<% endfor %>
<% for author in authors %>
<% if author.corresponding %>
\cortext[cor]{Corresponding author. Email: << author.email >>}
<% endif %>
<% endfor %>
\begin{abstract}
<< abstract >>
\end{abstract}
\begin{keyword}
<< keywords >>
\end{keyword}
\end{frontmatter}
\section{Introduction}
<< introduction >>
\section{Methodology}
<< methodology >>
\section{Results}
<< results >>
<% for table in tables %>
<< table.latex >>
<% endfor %>
<% for figure in figures %>
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/<< figure.filename >>}
\caption{<< figure.caption >>}
\label{fig:<< figure.number >>}
\end{figure}
<% endfor %>
\section{Discussion}
<< discussion >>
\section{Conclusion}
<< conclusion >>
<% if future_work %>
\section{Future Work}
<< future_work >>
<% endif %>
<% if limitations %>
\section{Limitations}
<< limitations >>
<% endif %>
\bibliographystyle{<< bib_style >>}
\bibliography{references}
\end{document}"""

GENERIC_TEMPLATE = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{natbib}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}
\title{<< title >>}
\author{
<% for author in authors %>
<< author.name >>\\
\small << author.affiliation >>, << author.department >>\\
\small \texttt{<< author.email >>}<% if not loop.last %>\\[1em]<% endif %>
<% endfor %>
}
\date{}
\begin{document}
\maketitle
\begin{abstract}
<< abstract >>
\end{abstract}
\noindent\textbf{Keywords:} << keywords >>
\section{Introduction}
<< introduction >>
\section{Methodology}
<< methodology >>
\section{Results}
<< results >>
<% for table in tables %>
<< table.latex >>
<% endfor %>
<% for figure in figures %>
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/<< figure.filename >>}
\caption{<< figure.caption >>}
\label{fig:<< figure.number >>}
\end{figure}
<% endfor %>
\section{Discussion}
<< discussion >>
\section{Conclusion}
<< conclusion >>
<% if future_work %>
\section{Future Work}
<< future_work >>
<% endif %>
<% if limitations %>
\section{Limitations}
<< limitations >>
<% endif %>
\bibliographystyle{<< bib_style >>}
\bibliography{references}
\end{document}"""

TEMPLATE_MAP = {
    "IEEE": IEEE_TEMPLATE,
    "ACM": ACM_TEMPLATE,
    "Springer": SPRINGER_TEMPLATE,
    "Elsevier": ELSEVIER_TEMPLATE,
    "Generic": GENERIC_TEMPLATE,
    "MDPI": GENERIC_TEMPLATE,
}

# ============================================================================
# GEMINI ENGINE ADAPTER (MODERN GOOGLE-GENAI SDK)
# ============================================================================
def call_gemini(prompt, system_instruction=None, json_mode=False, custom_key=None):
    """
    Executes an atomic API pipeline stage using the authentic google-genai structural syntax.
    Falls back to system environment configurations if a specific token context isn't passed.
    """
    api_key = custom_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None, "Missing Gemini API key. Ensure GEMINI_API_KEY is configured."

    try:
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            max_output_tokens=2048,
            response_mime_type="application/json" if json_mode else "text/plain"
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config
        )
        text = (response.text or "").strip()
        if not text:
            return None, "Gemini returned an empty response string."

        if json_mode:
            text = re.sub(r"^```(json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
            if not text.startswith("{") and not text.startswith("["):
                match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
                if match:
                    text = match.group(0)
            return json.loads(text), None

        return text, None
    except Exception as e:
        return None, str(e)

# ============================================================================
# HELPER DATA COMPILING DISPATCHERS
# ============================================================================
def reference_to_bibtex_entry(ref):
    entry_type = ref.get("entry_type") or "misc"
    key = ref.get("key") or "ref"
    field_map = [
        ("title", "title"),
        ("author", "authors"),
        ("year", "year"),
        ("journal", "venue"),
        ("doi", "doi"),
        ("url", "url"),
    ]
    lines = []
    for bib_field, ref_field in field_map:
        value = ref.get(ref_field, "")
        if value:
            lines.append(f"  {bib_field} = {{{value}}}")
    return f"@{entry_type}{{{key},\n" + ",\n".join(lines) + "\n}}"

def generate_references_bib(state):
    refs = state.get("references", [])
    if not refs:
        return "% No references added yet.\n"
    return "\n\n".join(reference_to_bibtex_entry(r) for r in refs)

def build_latex_context(state):
    pi = state.get("paper_info", {})
    citation_style = pi.get("citation_style", "Generic")
    journal_key = citation_style if citation_style in TEMPLATE_MAP else "Generic"

    authors_ctx = [{
        "name": latex_escape(a.get("name", "")),
        "affiliation": latex_escape(a.get("affiliation", "")),
        "department": latex_escape(a.get("department", "")),
        "email": latex_escape(a.get("email", "")),
        "corresponding": a.get("corresponding", False),
    } for a in state.get("authors", [])]

    figures_ctx = [{
        "filename": fig.get("filename", ""),
        "caption": latex_escape(fig.get("caption", "")),
        "number": fig.get("number", ""),
    } for fig in state.get("figures", [])]

    tables_ctx = [{
        "latex": table_to_latex(t),
    } for t in state.get("tables", [])]

    content = state.get("content", {})
    generated = state.get("generated", {})

    context = {
        "title": latex_escape(pi.get("title", "Untitled Paper")),
        "keywords": latex_escape(pi.get("keywords", "")),
        "target_journal": latex_escape(pi.get("target_journal", "")),
        "authors": authors_ctx,
        "abstract": generated.get("abstract") or "Abstract not yet generated.",
        "introduction": generated.get("introduction") or content.get("problem_statement", ""),
        "methodology": generated.get("methodology_refined") or content.get("methodology", ""),
        "results": latex_escape(content.get("results", "")),
        "discussion": generated.get("discussion") or content.get("discussion_notes", ""),
        "conclusion": generated.get("conclusion") or content.get("conclusion_notes", ""),
        "future_work": latex_escape(content.get("future_work", "")),
        "limitations": latex_escape(content.get("limitations", "")),
        "figures": figures_ctx,
        "tables": tables_ctx,
        "bib_style": BIB_STYLE_MAP.get(citation_style, "plain"),
    }
    return context, journal_key

# ============================================================================
# ROUTE ENDPOINTS
# ============================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extract_pdf", methods=["POST"])
def api_extract_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file stream detected"}), 400
    f = request.files["file"]
    try:
        file_bytes = f.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = [page.get_text() for page in doc]
        full_text = "\n".join(text_parts).strip()
        metadata = doc.metadata or {}
        doc.close()
        return jsonify({
            "text": full_text,
            "title": metadata.get("title", "") or "",
            "author": metadata.get("author", "") or ""
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/parse_bibtex", methods=["POST"])
def api_parse_bibtex():
    if "file" not in request.files:
        return jsonify({"error": "No file stream detected"}), 400
    f = request.files["file"]
    try:
        text = f.read().decode("utf-8", errors="ignore")
        bib_db = bibtexparser.loads(text)
        entries = []
        for entry in bib_db.entries:
            entries.append({
                "key": entry.get("ID", f"ref{len(entries) + 1}"),
                "entry_type": entry.get("ENTRYTYPE", "misc"),
                "title": entry.get("title", ""),
                "authors": entry.get("author", ""),
                "year": entry.get("year", ""),
                "venue": entry.get("journal", entry.get("booktitle", "")),
                "doi": entry.get("doi", ""),
                "url": entry.get("url", ""),
                "source_type": "bibtex"
            })
        return jsonify({"entries": entries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/call_gemini_module", methods=["POST"])
def api_call_gemini_module():
    data = request.json or {}
    module = data.get("module")
    state = data.get("state", {})
    custom_key = state.get("api_key")
    
    content = state.get("content", {})
    pi = state.get("paper_info", {})
    gen = state.get("generated", {})

    if module == "research_analysis":
        prompt = f"""Analyze the research notes below and return ONLY valid JSON with keys:
problem, objectives, methodology, findings, missing_sections (array of strings).
Do not invent anything not present below. If a field is missing, set it to an
empty string and list it in missing_sections.

PROBLEM STATEMENT:
{content.get('problem_statement', '')[:1500]}

OBJECTIVES:
{content.get('objectives', '')[:800]}

METHODOLOGY:
{content.get('methodology', '')[:1500]}

RESULTS:
{content.get('results', '')[:1500]}"""
        res, err = call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE, json_mode=True, custom_key=custom_key)

    elif module == "abstract":
        prompt = f"""Write an academic abstract (maximum 250 words) for a paper titled
"{pi.get('title', 'Untitled')}" using ONLY the information below. Do not
add results, numbers, or claims that are not stated. Plain text only.

PROBLEM: {content.get('problem_statement', '')[:800]}
OBJECTIVES: {content.get('objectives', '')[:500]}
METHODOLOGY: {content.get('methodology', '')[:800]}
RESULTS: {content.get('results', '')[:800]}
CONCLUSION NOTES: {content.get('conclusion_notes', '')[:400]}"""
        res, err = call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE, json_mode=False, custom_key=custom_key)

    elif module == "introduction":
        extra = ""
        analysis = gen.get("research_analysis")
        if analysis and analysis.get("missing_sections"):
            extra = f"\nKnown gaps in the notes (do not fabricate to fill them): {analysis.get('missing_sections')}"
        prompt = f"""Write an academic Introduction section for a paper titled
"{pi.get('title', 'Untitled')}" (keywords: {pi.get('keywords', '')}).
Use ONLY the content below; do not fabricate background facts, statistics, or
citations. Plain text only.

PROBLEM STATEMENT: {content.get('problem_statement', '')[:1200]}
OBJECTIVES: {content.get('objectives', '')[:600]}{extra}"""
        res, err = call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE, json_mode=False, custom_key=custom_key)

    elif module == "methodology_refined":
        prompt = f"""Improve the clarity, structure, and academic tone of the
following methodology section. Do NOT add new steps, tools, or data not
already mentioned. Plain text only.

METHODOLOGY:
{content.get('methodology', '')[:2500]}"""
        res, err = call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE, json_mode=False, custom_key=custom_key)

    elif module == "discussion":
        prompt = f"""Write an academic Discussion section based ONLY on the
results and notes below. Do not introduce new findings or numbers. Plain
text only.

RESULTS: {content.get('results', '')[:1500]}
DISCUSSION NOTES: {content.get('discussion_notes', '')[:1200]}"""
        res, err = call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE, json_mode=False, custom_key=custom_key)

    elif module == "conclusion":
        prompt = f"""Write an academic Conclusion section using ONLY the
information below. Do not introduce new claims. Plain text only.

OBJECTIVES: {content.get('objectives', '')[:500]}
RESULTS: {content.get('results', '')[:800]}
CONCLUSION NOTES: {content.get('conclusion_notes', '')[:600]}
FUTURE WORK: {content.get('future_work', '')[:400]}
LIMITATIONS: {content.get('limitations', '')[:400]}"""
        res, err = call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION_BASE, json_mode=False, custom_key=custom_key)
    else:
        return jsonify({"error": "Unknown workflow generation segment specified"}), 400

    if err:
        return jsonify({"error": err}), 500
    return jsonify({"result": res})

@app.route("/api/generate_preview", methods=["POST"])
def api_generate_preview():
    state = request.json or {}
    try:
        context, journal_key = build_latex_context(state)
        template = latex_jinja_env.from_string(TEMPLATE_MAP[journal_key])
        main_tex = template.render(**context)
        references_bib = generate_references_bib(state)
        return jsonify({"main_tex": main_tex, "references_bib": references_bib})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/download_zip", methods=["POST"])
def api_download_zip():
    state = request.json or {}
    try:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            context, journal_key = build_latex_context(state)
            template = latex_jinja_env.from_string(TEMPLATE_MAP[journal_key])
            main_tex = template.render(**context)
            
            zf.writestr("main.tex", main_tex)
            zf.writestr("references.bib", generate_references_bib(state))
            
            # Formulating structure manifest metadata
            metadata_payload = {
                "paper_info": state.get("paper_info", {}),
                "authors": state.get("authors", []),
                "references_count": len(state.get("references", [])),
                "figures_count": len(state.get("figures", [])),
                "tables_count": len(state.get("tables", [])),
                "generated_on": datetime.now().isoformat(),
                "generated_sections": {k: bool(v) for k, v in state.get("generated", {}).items()},
            }
            zf.writestr("metadata.json", json.dumps(metadata_payload, indent=2))
            
            style = state.get("paper_info", {}).get("citation_style", "Generic")
            readme_text = f"""ResearchPaper Builder Export\n=============================\n\nThis package contains:\n- main.tex          LaTeX source for your paper ({style} template)\n- references.bib    Bibliography file\n- figures/          All uploaded figure files\n- metadata.json     Paper metadata snapshot\n\nHOW TO USE WITH OVERLEAF\n1. Go to https://www.overleaf.com and create a New Project.\n2. Choose "Upload Project" and select this ZIP file.\n3. Overleaf detects main.tex automatically -- click Recompile.\n"""
            zf.writestr("README.txt", readme_text)
            
            for fig in state.get("figures", []):
                if "bytes_b64" in fig and fig["bytes_b64"]:
                    fig_bytes = base64.b64decode(fig["bytes_b64"])
                    zf.writestr(f"figures/{fig['filename']}", fig_bytes)

        buffer.seek(0)
        return send_file(
            buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name="research_paper_package.zip"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)