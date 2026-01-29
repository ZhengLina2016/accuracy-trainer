import { spawn } from "child_process";
import assert from "assert";
import fs from "fs";
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SESSIONS_FILE = path.join(__dirname, "sessions.json");
const BACKUP_FILE = path.join(__dirname, "sessions.json.bak");

// 8 days ago
const OLD_TIMESTAMP = Date.now() - (8 * 24 * 60 * 60 * 1000);
const EXPIRED_SESSION_ID = "expired_test_session_" + Date.now();

async function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

async function main() {
  let proc;
  let hasBackup = false;

  try {
    // 1. Prepare Cleanup Test: Inject expired session
    if (fs.existsSync(SESSIONS_FILE)) {
      try {
        fs.copyFileSync(SESSIONS_FILE, BACKUP_FILE);
        hasBackup = true;
        console.log("[Setup] Backed up sessions.json");
      } catch (e) {
        console.warn("[Setup] Warning: Could not backup sessions.json:", e.message);
      }
    }

    let sessions = [];
    if (fs.existsSync(SESSIONS_FILE)) {
        try {
            const content = fs.readFileSync(SESSIONS_FILE, 'utf8');
            if (content.trim()) sessions = JSON.parse(content);
        } catch (e) { console.log("Error reading sessions.json, starting empty"); }
    }
    
    // Add expired session
    sessions.push({
      id: EXPIRED_SESSION_ID,
      createdAt: new Date(OLD_TIMESTAMP).toISOString(),
      lastActive: OLD_TIMESTAMP,
      notes: "Expired Session",
      apiKey: "dummy"
    });
    fs.writeFileSync(SESSIONS_FILE, JSON.stringify(sessions, null, 2));
    console.log("[Setup] Injected expired session:", EXPIRED_SESSION_ID);

    // 2. Start Server
    console.log("Starting Server in MOCK Mode...");
    // Use ignore for stdin to prevent hanging, pipe for stdout to read port, inherit for stderr to see errors
    proc = spawn(process.execPath, ["server.js"], {
      env: { ...process.env, PORT: "9999", MOCK_LLM: "true" },
      stdio: ["ignore", "pipe", "inherit"]
    });

    let baseUrl = "";
    const portPromise = new Promise((resolve, reject) => {
        proc.stdout.on('data', (data) => {
            const str = data.toString();
            process.stdout.write(str); // Pass through
            const match = str.match(/running on (http:\/\/[\w\.:]+)/);
            if (match) resolve(match[1]);
        });
        setTimeout(() => reject(new Error("Timeout waiting for server start")), 10000);
    });

    baseUrl = await portPromise;
    console.log(`Server started at ${baseUrl}`);

    // 3. Run Functional Tests
    console.log("1. Starting Session...");
    const startRes = await fetch(`${baseUrl}/api/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes: "Mock Notes for Testing", apiKey: "dummy_key_1234567890" })
    });
    
    if (!startRes.ok) throw new Error(`Start failed: ${await startRes.text()}`);
    const startData = await startRes.json();
    assert.ok(startData.sessionId, "Should have sessionId");
    assert.strictEqual(startData.pack.meta.version_hash, "mock", "Should be mock pack");
    console.log("✅ Start Session Passed");

    // 4. Submit Answer
    console.log("2. Submitting Answer...");
    const submitRes = await fetch(`${baseUrl}/api/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        sessionId: startData.sessionId, 
        apiKey: "dummy_key_1234567890",
        answers: { q1: "A" } 
      })
    });

    if (!submitRes.ok) throw new Error(`Submit failed: ${await submitRes.text()}`);
    const submitData = await submitRes.json();
    
    // Verify structure fix
    assert.strictEqual(submitData.summary.status, "continue", "Should continue");
    console.log("✅ Submit Answer Passed");

    // 5. Verify Cleanup
    console.log("3. Verifying Cleanup...");
    // Give server a moment to write to disk
    await wait(2000);
    
    const newSessionsContent = fs.readFileSync(SESSIONS_FILE, 'utf8');
    const newSessions = JSON.parse(newSessionsContent);
    const found = newSessions.find(s => s.id === EXPIRED_SESSION_ID);
    
    if (!found) {
        console.log("✅ Cleanup Passed: Expired session removed");
    } else {
        throw new Error("❌ Cleanup Failed: Expired session still exists");
    }

  } catch (e) {
    console.error("❌ Test Failed:", e);
    process.exit(1);
  } finally {
    if (proc) {
        proc.kill();
        // Wait for process to exit to release file locks?
        // await wait(1000); 
    }
    
    // Restore backup
    if (hasBackup) {
      try {
        // Wait a bit to ensure file is not locked by server
        await wait(1000);
        fs.copyFileSync(BACKUP_FILE, SESSIONS_FILE);
        fs.unlinkSync(BACKUP_FILE);
        console.log("[Teardown] Restored sessions.json");
      } catch (e) {
        console.error("Error restoring backup:", e);
      }
    }
    process.exit(0);
  }
}

main();
