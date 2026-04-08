# Mermaid Diagrams → LaTeX PDF Workflow

This directory holds Mermaid diagram source files (`.mmd`) and a build script
that converts them to PDF for inclusion in the LaTeX theses via `\includegraphics`.

## Directory Structure

```
diagrams/
├── build_diagrams.sh       Build script (renders .mmd → PDF)
├── mmdc_config.json        Mermaid CLI theme/font config
├── README.md               This file
├── example_flowchart.mmd   Example diagram
└── output/                 Generated PDFs (git-ignored)
```

## Prerequisites

- **Node.js** — `brew install node`
- **Mermaid CLI** — `npm install -g @mermaid-js/mermaid-cli`

Verify with: `mmdc --version`

## Usage

### Build all diagrams

```bash
cd doc/Theses/shared/figures/diagrams
bash build_diagrams.sh
```

### Build a specific diagram

```bash
bash build_diagrams.sh my_diagram.mmd
```

PDFs are written to `output/`. The filename matches the source: `foo.mmd` → `output/foo.pdf`.

## Creating a New Diagram

1. Create a `.mmd` file in this directory:

    ```bash
    cat > my_diagram.mmd << 'EOF'
    flowchart LR
        A[Input] --> B[Process] --> C[Output]
    EOF
    ```

2. Build it:

    ```bash
    bash build_diagrams.sh my_diagram.mmd
    ```

3. Include in LaTeX:

    ```latex
    \begin{figure}[htbp]
        \centering
        \includegraphics[width=0.8\textwidth]{figures/diagrams/output/my_diagram.pdf}
        \caption{My diagram.}
        \label{fig:my_diagram}
    \end{figure}
    ```

    Since the `figures/` directory is symlinked into each thesis project, the
    path `figures/diagrams/output/my_diagram.pdf` works from any thesis.

## Customisation

Edit `mmdc_config.json` to change the theme, font, or colours:

```json
{
    "theme": "neutral",
    "themeVariables": {
        "fontSize": "14px",
        "fontFamily": "Helvetica, Arial, sans-serif"
    }
}
```

Available themes: `default`, `neutral`, `dark`, `forest`, `base`.

See the [Mermaid theming docs](https://mermaid.js.org/config/theming.html) for
all available `themeVariables`.

## Supported Diagram Types

Mermaid supports all of these — just use the appropriate syntax in your `.mmd` file:

| Type | Keyword |
|------|---------|
| Flowchart | `flowchart` |
| Sequence | `sequenceDiagram` |
| Class | `classDiagram` |
| State | `stateDiagram-v2` |
| ER | `erDiagram` |
| Gantt | `gantt` |
| Pie | `pie` |
| Git Graph | `gitGraph` |
| Mindmap | `mindmap` |
| Timeline | `timeline` |
| Quadrant | `quadrantChart` |
| XY Chart | `xychart-beta` |
| Sankey | `sankey-beta` |
| Block | `block-beta` |

See the full [Mermaid syntax reference](https://mermaid.js.org/intro/syntax-reference.html).
