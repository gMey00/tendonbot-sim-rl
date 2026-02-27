---
applyTo: '**'
---
## General rules:

1. If I tell you to do something, I mean you need to do every part of it. Don't skip steps or give me instructions on what to do. You have tools, use them.
2. If something is wrong with my system, assume I would rather you fix it than find a workaround.
3. I only have a limited number of prompts, so do not waste them asking for silly clarifications. Start doing the task and I will stop you if it is wrong.

## Coding rules:

1. Use proper decomposition instead of comments to explain your code.
2. Use proper variable names and do not shorten words. You can omit words in names, however, if the context is clear.
3. Prefer immutability over mutability.
4. Prefer functional/declarative programming over imperative programming.
5. Always check my coding style first. Pay attention to the finest details:
  - Naming conventions
  - Spacing
  - Formatting
  - Architecture
  - Import Order, Grouping and Style
  - Etc.
6. For Python code, proper static typing is required.
7. If a command related to Python fails, check if the virtual environment is activated. If not, activate it and try again.

## Test writing rules:

1. Always write tests based on the logical behavior that the code should have, not what the implementation is currently doing.
2. The format for tests should always be `test_that_<functionality>[_when_<condition>]` and the structure of the test should follow the arrange-act-assert pattern.
3. Never use mocks, only use stub or real implementations.
4. Pretty test DSL is important.
5. Do not implement code unless you failed the test first.

## After Task Completion:

After completing any task, run (Only if you are the main agent, otherwise hand execution back to the main agent):
```bash
read -p "What shall I do next? " cmd && echo "$cmd"
```

It may take some time for the command to finish executing, but wait patiently and do not interrupt it. Once it finishes, execute the instructions returned in stdout.
