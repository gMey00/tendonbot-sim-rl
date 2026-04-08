# Thesis Section Writing Prompt

Use this prompt with a research-capable AI (e.g., one with web search) to write a specific section of the thesis. Replace the placeholder at the top with the subsection you want written.

---

## Prompt

```
I need you to write a section of my master's thesis. The section I want you to write is:

>>> PASTE THE SUBSECTION FROM THE CHAPTER OUTLINE HERE (including all bullet points) <<<

For further reference I also uploaded the current state of my thesis as a pdf. First read it before doing anything else.

─────────────────────────────────────────────────────────
PHASE 1 — RESEARCH (do this BEFORE writing anything)
─────────────────────────────────────────────────────────

1. Thoroughly search the web for peer-reviewed scientific papers, conference proceedings, authoritative technical documentation, and established textbooks that are directly relevant to every bullet point listed above.
2. Aim to gather at least 8–15 distinct, real sources. Prefer:
   • Peer-reviewed journal articles and top-tier conference papers (IEEE, ACM, Springer, Elsevier).
   • Official technical documentation from NVIDIA (Isaac Sim, PhysX, IsaacLab, Newton) where applicable.
   • Established textbooks (e.g., Siciliano et al. for robotics, Sutton & Barto for RL).
   • Recent survey papers that consolidate the state of the art.
3. For every source, record: author(s), title, year, venue/publisher, DOI or URL. You will need this for the BibTeX entries later.
4. DO NOT fabricate, hallucinate, or invent any citation. Every claim attributed to a source must actually appear in that source. If you cannot find a source for a specific claim, state the claim without a citation and mark it with a LaTeX comment `% TODO: find citation` so the author can verify.
5. DO NOT copy or closely paraphrase sentences from any source. All text must be original formulations that convey the same factual content in the author's own scientific voice.

─────────────────────────────────────────────────────────
PHASE 2 — WRITING
─────────────────────────────────────────────────────────

Write the section in LaTeX. Adhere to ALL of the following rules:

### Scientific tone and style
- Write in formal, impersonal academic English. Use passive voice consistently (e.g., "the robot is modelled as…") stay consistent throughout the section; the existing thesis uses passive voice.
- Be precise and concise. Avoid colloquial language, hedging words ("basically", "kind of"), and filler phrases.
- Every technical term must be introduced before use or accompanied by a brief inline definition on first occurrence.
- Statements of fact must be supported by citations. Statements that are common knowledge in the field (e.g., "gravity acts downward") do not need citations.
- When comparing approaches, be balanced and evidence-based. Do not make unsupported value judgements.
- Use present tense for established facts and general truths ("PBD projects positions to satisfy constraints"), past tense for specific experimental work ("Müller et al. proposed…"), and future tense only when describing planned work.

### LaTeX formatting conventions (MUST follow exactly)

1. **Document class context**: This is a `book` class thesis (12pt, a4paper, oneside). The section will be `\input`-ed into a chapter file. Do NOT include `\documentclass`, `\begin{document}`, or preamble commands.

2. **Section hierarchy** (use exactly these levels as appropriate):
   - `\section{...}` — top-level within the chapter (this is what you write)
   - `\subsection{...}` — major subdivisions
   - `\subsubsection{...}` — further subdivisions (use sparingly)
   - `\paragraph{...}` — named paragraphs for fine-grained structure within a subsubsection (end with a period: `\paragraph{Topic name.}`)

3. **Section headers**: Use this framing style:
   ```latex
   %===================================================================%
   \section{Section Title}
   \label{sec:snake_case_label}
   %===================================================================%
   ```

4. **Citations**: Use `\cite{BibTeXKey}` for all references. Use the following patterns:
   - Parenthetical citation at end of sentence: `...as shown previously~\cite{Author2020}.`
   - Textual citation: `Author \emph{et al.}~\cite{Author2020} demonstrated that...`
   - Multiple citations: `\cite{Author2020,OtherAuthor2021}`
   - The tilde `~` before `\cite` prevents a line break between the preceding word and the citation bracket.

5. **Acronyms**: Use the `acro` package macros:
   - First use in a section: `\ac{ACRONYM}` (expands to "Full Name (ACRONYM)")
   - Subsequent uses: `\ac{ACRONYM}` (the package handles abbreviation automatically)
   - Plural: `\acp{ACRONYM}`
   - Short form only (e.g., in headings or equations): `\acs{ACRONYM}`
   - If you need a NEW acronym, define it in a comment at the top of the file:
     ```latex
     % NEW ACRONYM — add to formatting/Acronyms.tex:
     % \DeclareAcronym{TGS}{short=TGS, long=Temporal Gauss-Seidel, tag=abbrev, sort=TGS}
     ```

6. **Math**:
   - Inline: `$...$`
   - Display (numbered): `\begin{equation} ... \label{eq:name} \end{equation}`
   - Display (unnumbered): `\[ ... \]` or `\begin{equation*} ... \end{equation*}`
   - Vectors and matrices: `\mathbf{x}`, `\mathbf{M}`
   - Operators: `\mathcal{S}`, `\mathbb{R}`, `\text{clip}`, `\arg\min`
   - Use `amsmath` and `amssymb` features (already loaded).

7. **Figures** (use placeholders as the author will add actual images later):
   ```latex
   \begin{figure}[ht!]
   \centering
   \fbox{\parbox{0.8\textwidth}{\centering \vspace{0.1in}Placeholder for \textit{description} image\vspace{0.1in}}}
   \caption[Short LoF caption]{Full caption with explanation.}\label{fig:label}
   \end{figure}
   ```

8. **Tables**:
   ```latex
   \begin{table}[htb]
       \centering
       \caption{Caption text.}\label{tab:label}
       \begin{tabular}{l c c}
       \hline
       \textbf{Column 1} & \textbf{Column 2} & \textbf{Column 3} \\
       \hline
       Row 1 & data & data \\
       \hline
       \end{tabular}
   \end{table}
   ```

9. **Itemize/enumerate**: Use `\begin{itemize}` with `\item` entries. Use `\textbf{Bold label:}` for labeled items:
   ```latex
   \begin{itemize}
     \item \textbf{Label:} Description text~\cite{Source}.
   \end{itemize}
   ```

10. **Cross-references**: Use `\ref{label}` for figures/tables/equations and `\autoref{label}` or `Chapter/Section~\ref{label}` for sections.

11. **Units and numbers**: Use standard LaTeX: `$0.5\,\mathrm{m}$`, `$140^\circ$`, `$\pm 50^\circ$`.

12. **Special characters**: Use `~` for non-breaking spaces before citations and references. Use `--` for en-dashes, `---` for em-dashes. Escape `%`, `&`, `#`, `_` when used literally.

─────────────────────────────────────────────────────────
PHASE 3 — BIBTEX ENTRIES AND SOURCE DOCUMENTATION
─────────────────────────────────────────────────────────

At the END of the LaTeX section file, include a block comment listing:

1. All BibTeX keys used in the section and what they reference:
   ```latex
   % ---------------------------------------------------------------------------
   % BibTeX keys used in this section:
   %  - AuthorYear_short_descriptor    (brief description of source)
   %  - AnotherAuthor2021_topic        (brief description)
   % ---------------------------------------------------------------------------
   ```

2. Then provide the COMPLETE BibTeX entries for ALL new references in a second comment block, ready to be pasted into the bibliography file:
   ```latex
   % ---------------------------------------------------------------------------
   % BibTeX entries — add to bibliography/literature.bib:
   %
   % @article{AuthorYear_short_descriptor,
   %   author    = {First Last and First Last},
   %   title     = {Full Title},
   %   journal   = {Journal Name},
   %   volume    = {10},
   %   number    = {2},
   %   pages     = {100--120},
   %   year      = {2020},
   %   doi       = {10.xxxx/xxxxx},
   % }
   %
   % @inproceedings{AnotherAuthor2021_topic,
   %   ...
   % }
   % ---------------------------------------------------------------------------
   ```

3. For web/documentation sources, use `@online` or `@misc`:
   ```latex
   % @online{nvidia_isaac_sim_doc,
   %   author  = {{NVIDIA}},
   %   title   = {Isaac Sim Documentation},
   %   year    = {2024},
   %   url     = {https://docs.omniverse.nvidia.com/...},
   %   urldate = {2025-03-03},
   % }
   ```

### BibTeX key naming convention
Follow the pattern used in the existing thesis:
- Papers: `AuthornameYearKeyword` (e.g., `Schulman2017PPO`, `muller2007pbd`, `chen2024vbd`)
- Documentation: `descriptive_snake_case` (e.g., `physx_pbd_particlesystem_510`, `omni_physics_particles`)
- Books: `AuthornameYear` or `AuthornameYearKeyword` (e.g., `SuttonBarto2018`)
- Theses: `AuthornameYear` (e.g., `Klein2023`)

─────────────────────────────────────────────────────────
CRITICAL REMINDERS
─────────────────────────────────────────────────────────

• NEVER invent a paper, author, journal, or result. Every citation must be a real, verifiable publication.
• NEVER copy text from sources — rephrase all content in your own words.
• If you are uncertain whether a specific fact is accurate, mark it: `% TODO: verify claim`
• Cover ALL bullet points from the outline above. Do not skip any.
• Target approximately 2–5 pages of dense academic text per section (the full Background chapter is ~20 pages across 7 sections).
• The bibliography file path is `bibliography/literature.bib` and the style is a numeric bracket style `[1]` based on a custom FAPS BST file.
```

---

## How to use

1. Copy the prompt above.
2. Replace the `>>> PASTE THE SUBSECTION... <<<` placeholder with the exact subsection content from your `0_1_ChapterOutline.tex`.
3. Paste into a research-capable AI that can search the web (e.g., ChatGPT with browsing, Perplexity, Claude with web search).
4. Review the output: verify all citations exist, check tone consistency, and paste the BibTeX entries into `bibliography/literature.bib`.
5. Save the LaTeX output into the corresponding `chapters/background/2_X_*.tex` file (or methodology/results file as appropriate).
