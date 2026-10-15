/* Runs the real evolvemas package in Pyodide. The page talks to it with postMessage. */
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const ready = (async () => {
  const py = await loadPyodide();
  const base = new URL("../py/", self.location.href);
  const manifest = await (await fetch(new URL("manifest.json", base))).json();
  for (const rel of manifest.files) {
    const text = await (await fetch(new URL(rel, base))).text();
    const path = "/py/" + rel;
    py.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    py.FS.writeFile(path, text);
  }
  const driver = await (await fetch(new URL("driver.py", self.location.href))).text();
  py.FS.writeFile("/py/driver.py", driver);
  py.runPython("import sys; sys.path.insert(0, '/py'); import driver");
  postMessage(JSON.stringify({ t: "ready", version: manifest.version }));
  return py.pyimport("driver");
})().catch((e) => {
  postMessage(JSON.stringify({ t: "fatal", msg: String(e) }));
  throw e;
});

onmessage = async (e) => {
  const driver = await ready;
  driver.handle(JSON.stringify(e.data));
};
