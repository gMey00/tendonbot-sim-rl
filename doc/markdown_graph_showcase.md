# Markdown Graph & Diagram Showcase

A reference of every diagram type available in GitHub-flavoured Markdown
(rendered via Mermaid, GeoJSON, TopoJSON, and STL).

---

## Table of Contents

- [Mermaid Diagrams](#mermaid-diagrams)
  - [Flowchart](#flowchart)
  - [Sequence Diagram](#sequence-diagram)
  - [Class Diagram](#class-diagram)
  - [State Diagram](#state-diagram)
  - [Entity Relationship Diagram](#entity-relationship-diagram)
  - [Gantt Chart](#gantt-chart)
  - [Pie Chart](#pie-chart)
  - [Git Graph](#git-graph)
  - [Mindmap](#mindmap)
  - [Timeline](#timeline)
  - [Quadrant Chart](#quadrant-chart)
  - [XY Chart (Bar)](#xy-chart-bar)
  - [XY Chart (Line)](#xy-chart-line)
  - [Sankey Diagram](#sankey-diagram)
  - [Block Diagram](#block-diagram)
  - [Packet Diagram](#packet-diagram)
  - [Kanban Board](#kanban-board)
  - [Architecture Diagram](#architecture-diagram)
- [Mathematical Equations (LaTeX)](#mathematical-equations-latex)
- [GeoJSON Map](#geojson-map)
- [TopoJSON Map](#topojson-map)
- [STL 3D Model](#stl-3d-model)

---

## Mermaid Diagrams

All diagrams below use fenced code blocks with the `mermaid` language identifier.

### Flowchart

```mermaid
flowchart TD
    A[Start] --> B{Decision?}
    B -->|Yes| C[Action A]
    B -->|No| D[Action B]
    C --> E[Result]
    D --> E
    E --> F((End))
```

### Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant Database

    User ->> Frontend: Submit form
    Frontend ->> API: POST /data
    API ->> Database: INSERT record
    Database -->> API: OK
    API -->> Frontend: 201 Created
    Frontend -->> User: Show confirmation
```

### Class Diagram

```mermaid
classDiagram
    class Animal {
        +String name
        +int age
        +make_sound() String
    }
    class Dog {
        +String breed
        +fetch() void
    }
    class Cat {
        +bool indoor
        +purr() void
    }
    Animal <|-- Dog
    Animal <|-- Cat
```

### State Diagram

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Running : start()
    Running --> Paused : pause()
    Paused --> Running : resume()
    Running --> Finished : complete()
    Paused --> Idle : reset()
    Finished --> [*]
```

### Entity Relationship Diagram

```mermaid
erDiagram
    STUDENT ||--o{ ENROLMENT : enrols
    COURSE ||--o{ ENROLMENT : includes
    STUDENT {
        int id PK
        string name
        string email
    }
    COURSE {
        int id PK
        string title
        int credits
    }
    ENROLMENT {
        int student_id FK
        int course_id FK
        date enrolled_at
    }
```

### Gantt Chart

```mermaid
gantt
    title Project Timeline
    dateFormat  YYYY-MM-DD
    section Research
        Literature Review       :done,    lr, 2026-01-01, 30d
        Experiment Design       :active,  ed, after lr, 20d
    section Implementation
        Prototype               :         proto, after ed, 25d
        Testing                 :         test, after proto, 15d
    section Writing
        Draft                   :         draft, after proto, 30d
        Review & Revise         :         rev, after draft, 20d
```

### Pie Chart

```mermaid
pie title Reward Composition
    "Goal Reached"          : 42
    "Position Tracking"     : 25
    "Orientation Tracking"  : 15
    "Regularisation"        : 10
    "Action Rate"           : 8
```

### Git Graph

```mermaid
gitGraph
    commit id: "init"
    branch feature/reach
    checkout feature/reach
    commit id: "add reach env"
    commit id: "tune rewards"
    checkout main
    merge feature/reach
    branch feature/place
    checkout feature/place
    commit id: "add place env"
    commit id: "add curriculum"
    checkout main
    merge feature/place
    commit id: "release v1.0"
```

### Mindmap

```mermaid
mindmap
    root((RL Pipeline))
        Environment
            Scene Setup
            Reward Design
            Curriculum
        Training
            PPO
            Hyperparameter Tuning
            Checkpointing
        Evaluation
            Metrics
            Visualisation
            Ablation Studies
        Infrastructure
            Isaac Sim
            Isaac Lab
            skrl
```

### Timeline

```mermaid
timeline
    title Project Milestones
    2025-Q4 : Isaac Sim Setup
            : Robot URDF Import
    2026-Q1 : Reach Task Converged
            : Cube Place Task Converged
            : Tendon Actuation Validated
    2026-Q2 : Cube Sort Task
            : Shirt Simulation
            : Project Thesis Draft
    2026-Q3 : Master Thesis
            : Final Experiments
```

### Quadrant Chart

```mermaid
quadrantChart
    title Task Complexity vs Training Time
    x-axis Low Complexity --> High Complexity
    y-axis Short Training --> Long Training
    quadrant-1 Hard & Slow
    quadrant-2 Easy & Slow
    quadrant-3 Easy & Fast
    quadrant-4 Hard & Fast
    Reach: [0.2, 0.15]
    Cube Place: [0.5, 0.55]
    Cube Sort: [0.75, 0.7]
    Shirt Place: [0.65, 0.8]
    Shirt Sort: [0.9, 0.9]
```

### XY Chart (Bar)

```mermaid
xychart-beta
    title "Training Steps to Convergence"
    x-axis ["Reach", "Cube Place", "Cube Sort", "Shirt Place", "Shirt Sort"]
    y-axis "Steps (×1000)" 0 --> 400
    bar [36, 250, 300, 350, 400]
```

### XY Chart (Line)

```mermaid
xychart-beta
    title "Mean Episode Reward over Training"
    x-axis "Steps (×1000)" [0, 50, 100, 150, 200, 250, 300]
    y-axis "Reward" 0 --> 500
    line [0, 80, 200, 320, 400, 450, 470]
```

### Sankey Diagram

```mermaid
sankey-beta

Reach Reward,Position Tracking,25
Reach Reward,Orientation Tracking,15
Reach Reward,Goal Reached,42
Regularisation,Action Rate,8
Regularisation,Joint Velocity,10
```

### Block Diagram

```mermaid
block-beta
    columns 3
    input["Observations (22-dim)"] :3
    space :3
    block:policy :3
        columns 2
        hidden1["Hidden 256"] hidden2["Hidden 128"]
    end
    space :3
    actions["Actions (5-dim)"] :3
```

### Packet Diagram

```mermaid
packet-beta
    title "Observation Vector (Reach Task)"
    0-4: "Joint Positions (5)"
    5-9: "Joint Velocities (5)"
    10-16: "Pose Command (7)"
    17-21: "Previous Actions (5)"
```

### Kanban Board

```mermaid
kanban
    column1["Done"]
        task1["Reach Task"]
        task2["Cube Place Task"]
    column2["In Progress"]
        task3["Cube Sort Task"]
    column3["Planned"]
        task4["Shirt Place Task"]
        task5["Shirt Sort Task"]
```

### Architecture Diagram

```mermaid
architecture-beta
    group sim(cloud)[Simulation]
    group rl(server)[RL Training]

    service isaac(server)[Isaac Sim] in sim
    service lab(server)[Isaac Lab] in rl
    service skrl(server)[skrl PPO] in rl
    service gpu(disk)[GPU] in sim

    isaac:R -- L:lab
    lab:R -- L:skrl
    isaac:B -- T:gpu
```

---

## Mathematical Equations (LaTeX)

GitHub renders LaTeX math in Markdown using `$` for inline and `$$` for blocks.

Inline: The policy gradient is $\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)\right]$.

Block equation — PPO clipped surrogate objective:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.

Bellman equation:

$$
V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]
$$

---

## GeoJSON Map

GitHub renders GeoJSON maps inline when placed in a `geojson` code block.

```geojson
{
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [11.0275, 49.5732]
            },
            "properties": {
                "name": "FAPS — FAU Erlangen-Nürnberg",
                "description": "Department of Mechanical Engineering"
            }
        }
    ]
}
```

---

## TopoJSON Map

GitHub also renders TopoJSON. This example shows a simple bounding box around Erlangen.

```topojson
{
    "type": "Topology",
    "objects": {
        "erlangen": {
            "type": "GeometryCollection",
            "geometries": [
                {
                    "type": "Polygon",
                    "arcs": [[0]],
                    "properties": { "name": "Erlangen Area" }
                }
            ]
        }
    },
    "arcs": [
        [[10.95, 49.55], [11.10, 49.55], [11.10, 49.60], [10.95, 49.60], [10.95, 49.55]]
    ]
}
```

---

## STL 3D Model

GitHub renders STL files placed in `stl` code blocks as interactive 3D viewers.
Below is a minimal cube for demonstration.

```stl
solid cube
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 1 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 0 1 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 1
      vertex 1 0 0
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 0 0 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 0 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 1
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 0
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 1
      vertex 0 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 0 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 1
      vertex 0 0 1
    endloop
  endfacet
endsolid cube
```
