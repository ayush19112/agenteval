import React from "react";
import { createRoot } from "react-dom/client";
import AgentEvalDashboard from "./AgentEvalDashboard.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <AgentEvalDashboard />
  </React.StrictMode>
);
