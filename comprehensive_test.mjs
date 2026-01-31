import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

// Color codes for output
const colors = {
  reset: "\x1b[0m",
  green: "\x1b[32m",
  red: "\x1b[31m",
  yellow: "\x1b[33m",
  cyan: "\x1b[36m",
  bold: "\x1b[1m"
};

function log(msg, type = "info") {
  const prefix = {
    info: `[INFO]`,
    pass: `${colors.green}[PASS]${colors.reset}`,
    fail: `${colors.red}[FAIL]${colors.reset}`,
    warn: `${colors.yellow}[WARN]${colors.reset}`,
    section: `\n${colors.cyan}${colors.bold}===`
  };
  const suffix = type === "section" ? ` ===${colors.reset}` : "";
  console.log(`${prefix[type] || ""} ${msg}${suffix}`);
}

function wait(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForUrl(proc, timeoutMs = 15_000) {
  let url = null;
  let buf = "";
  const startedAt = Date.now();

  proc.stdout.setEncoding("utf8");
  proc.stdout.on("data", (chunk) => {
    buf += chunk;
    const m = buf.match(/Trainer backend running on (http:\/\/127\.0\.0\.1:\d+)/);
    if (m) url = m[1];
  });

  while (!url) {
    if (Date.now() - startedAt > timeoutMs) {
      throw new Error("启动超时：未获取到后端 URL");
    }
    await wait(100);
  }
  return url;
}

async function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

const metrics = {
    total: 0,
    passed: 0,
    failed: 0,
    performance: {}
};

async function runTest(name, fn) {
    metrics.total++;
    log(`Testing: ${name}`, "info");
    const start = performance.now();
    try {
        await fn();
        const duration = performance.now() - start;
        metrics.passed++;
        metrics.performance[name] = duration;
        log(`${name} - ${duration.toFixed(2)}ms`, "pass");
    } catch (e) {
        metrics.failed++;
        log(`${name} - ${e.message}`, "fail");
        if (e.response) {
            try {
                const text = await e.response.text();
                console.log("Response body:", text);
            } catch (err) {}
        }
    }
}

async function main() {
  log("Starting Server...", "info");
  const proc = spawn(process.execPath, ["server.js"], {
    cwd: process.cwd(),
    env: { ...process.env, PORT: "0", NO_OPEN: "1" },
    stdio: ["ignore", "pipe", "pipe"]
  });

  // Capture stderr for debugging
  proc.stderr.on('data', (data) => {
      // console.error(`Server stderr: ${data}`);
  });

  let url = null;
  try {
    url = await waitForUrl(proc);
    log(`Server running at ${url}`, "info");

    log("Health Checks", "section");
    
    await runTest("GET /api/ping", async () => {
      const res = await fetch(url + "/api/ping");
      await assert(res.ok, "Status should be 200");
      const j = await res.json();
      await assert(j.ok === true, "Response should be {ok: true}");
    });

    await runTest("POST /api/ping", async () => {
        const res = await fetch(url + "/api/ping", { method: "POST" });
        await assert(res.ok, "Status should be 200");
        const j = await res.json();
        await assert(j.ok === true, "Response should be {ok: true}");
    });

    log("File Upload Tests", "section");

    await runTest("Upload TXT File", async () => {
        const form = new FormData();
        const content = "Test content for upload.";
        const blob = new Blob([content], { type: "text/plain" });
        form.append("file", blob, "test.txt");

        const res = await fetch(url + "/api/upload", { method: "POST", body: form });
        await assert(res.ok, "Upload failed");
        const j = await res.json();
        await assert(j.text === content, "Content mismatch");
        await assert(j.fileName === "test.txt", "Filename mismatch");
    });

    await runTest("Upload Unsupported File (.doc)", async () => {
        const form = new FormData();
        const blob = new Blob(["fake doc"], { type: "application/msword" });
        form.append("file", blob, "test.doc");

        const res = await fetch(url + "/api/upload", { method: "POST", body: form });
        await assert(res.status === 400, "Should return 400 for .doc");
        const text = await res.text();
        await assert(text.includes("不支持"), "Error message should mention unsupported");
    });

    log("Session Management Tests", "section");

    await runTest("Start Session - Missing Notes", async () => {
        const res = await fetch(url + "/api/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ apiKey: "sk-test-key" })
        });
        await assert(res.status === 400, "Should fail without notes");
        const text = await res.text();
        await assert(text.includes("notes"), "Error should mention notes");
    });

    await runTest("Start Session - Missing API Key", async () => {
        const res = await fetch(url + "/api/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ notes: "Some valid notes content here." })
        });
        await assert(res.status === 400, "Should fail without API key");
        const text = await res.text();
        await assert(text.includes("API Key"), "Error should mention API Key");
    });

    // NOTE: We cannot test success of /api/start without a real API key because it calls the external API.
    // However, we can test that it proceeds to the point of calling the API or validates inputs.

    await runTest("Session State - Invalid ID", async () => {
        const res = await fetch(url + "/api/state", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sessionId: "invalid-id" })
        });
        await assert(res.status === 400, "Should fail with invalid session ID");
    });

    log("Note Intent API Tests", "section");

    await runTest("Note Intent Status - Invalid Intent", async () => {
        const res = await fetch(url + "/api/note_intent_status", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ intent: "X", sessionIds: ["s1"] })
        });
        await assert(res.status === 400, "Should reject unsupported intent");
        const text = await res.text();
        await assert(text.includes("A/B/C/D"), "Error should mention allowed intents");
    });

    await runTest("Note Intent Status - Missing sessionIds", async () => {
        const res = await fetch(url + "/api/note_intent_status", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ intent: "A" })
        });
        await assert(res.status === 400, "Should fail without sessionIds");
        const text = await res.text();
        await assert(text.includes("sessionIds"), "Error should mention sessionIds");
    });

    await runTest("Note Intent Status - Not Cached", async () => {
        const res = await fetch(url + "/api/note_intent_status", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ intent: "A", sessionIds: ["nonexistent-session"] })
        });
        await assert(res.ok, "Status should be 200 for valid payload");
        const j = await res.json();
        await assert(j.status === "not_cached", "Should report not_cached for fresh key");
    });

    await runTest("Note Intent Content - Invalid Intent", async () => {
        const res = await fetch(url + "/api/note_intent_content", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ intent: "Z", sessionIds: ["s1"] })
        });
        await assert(res.status === 400, "Should reject unsupported intent");
        const text = await res.text();
        await assert(text.includes("intent"), "Error should mention intent");
    });

    await runTest("Note Intent Content - Missing sessionIds", async () => {
        const res = await fetch(url + "/api/note_intent_content", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ intent: "A" })
        });
        await assert(res.status === 400, "Should fail without sessionIds");
        const text = await res.text();
        await assert(text.includes("sessionIds"), "Error should mention sessionIds");
    });

    log("Performance Metrics", "section");
    console.table(metrics.performance);

    log("Summary", "section");
    console.log(`Total: ${metrics.total}`);
    console.log(`Passed: ${metrics.passed}`);
    console.log(`Failed: ${metrics.failed}`);

  } finally {
    if (proc && !proc.killed) proc.kill();
  }
}

main().catch((e) => {
  console.error("FATAL ERROR:", e);
  process.exitCode = 1;
});
