// hello_python.mjs
import { loadPyodide } from "pyodide";

async function hello_python() {
  let pyodide = await loadPyodide();
  return pyodide.runPythonAsync("1+1");
}

const result = await hello_python();
console.log("Python says that 1+1 =", result);

