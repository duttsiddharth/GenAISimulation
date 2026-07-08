import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base must match the repo name for GitHub Pages project sites:
// https://duttsiddharth.github.io/genaisimulation/
export default defineConfig({
  plugins: [react()],
  base: "/genaisimulation/",
});
