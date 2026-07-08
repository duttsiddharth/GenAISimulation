import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import GenAIDevSimulator from "./GenAIDevSimulator.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <GenAIDevSimulator />
  </React.StrictMode>
);
