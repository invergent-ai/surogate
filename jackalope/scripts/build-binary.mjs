// Compile jackalope to a standalone executable (no Node required at runtime).
// Two bundler patches:
//   • react-devtools-core — Ink imports it but only uses it in DEV; stub it out.
//   • sixel/upng.js — relies on an implicit global `UPNG`; declare it so the
//     module initializes under strict-mode bundling.
const target = process.argv[2] || "bun-linux-x64";
const outfile = process.argv[3] || "dist/jackalope";
const entry = process.argv[4] || "src/cli.tsx";

const patches = {
  name: "jackalope-patches",
  setup(b) {
    b.onResolve({ filter: /^react-devtools-core$/ }, () => ({ path: "react-devtools-core", namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default {};", loader: "js" }));
    b.onLoad({ filter: /sixel[\/\\](lib[\/\\])?upng\.js$/ }, async (args) => {
      const src = await Bun.file(args.path).text();
      return { contents: "var UPNG;\n" + src, loader: "js" };
    });
  },
};

const res = await Bun.build({
  entrypoints: [entry],
  target: "bun",
  plugins: [patches],
  compile: { target, outfile },
});
if (!res.success) { for (const m of res.logs) console.error(m); process.exit(1); }
console.log("compiled →", outfile, "(", target, ")");
