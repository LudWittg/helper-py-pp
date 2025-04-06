# Use Z3Py in the browser with Pyodide

This repository (momentarely) implements a demo/experiment that displays the usage of the z3-solver with Python binding directly in the browser. It's a work in progress. The end goal is using this project to optimize, directly on the web client side, the helpers' schedule in a conference.
This project uses Astro (with bun) for the front end alongside Pyodide with wasm to run Python.

## 🧞 Commands

All commands are run from the root of the project, from a terminal:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `bun install`             | Installs dependencies                            |
| `bun dev`             | Starts local dev server at `localhost:4321`      |
| `bun build`           | Build your production site to `./dist/`          |
| `bun preview`         | Preview your build locally, before deploying     |
| `bun astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `bun astro -- --help` | Get help using the Astro CLI                     |
