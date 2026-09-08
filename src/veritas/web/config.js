// Frontend runtime configuration.
//
// VERITAS_API_BASE controls where the UI sends API requests.
//   - Leave "" (empty) when the FastAPI backend serves this UI directly
//     (single-service / same-origin deployment).
//   - For a SPLIT deployment (this frontend on Netlify, backend on a
//     Hugging Face Space), set it to the backend URL, e.g.:
//         window.VERITAS_API_BASE = "https://<user>-veritas.hf.space";
window.VERITAS_API_BASE = "";
