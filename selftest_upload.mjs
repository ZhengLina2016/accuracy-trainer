import { spawn } from "node:child_process";

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

  proc.stderr.setEncoding("utf8");
  proc.stderr.on("data", () => {});

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

async function main() {
  const proc = spawn(process.execPath, ["server.js"], {
    cwd: process.cwd(),
    env: { ...process.env, PORT: "0", NO_OPEN: "1" },
    stdio: ["ignore", "pipe", "pipe"]
  });

  let url = null;
  try {
    url = await waitForUrl(proc);

    {
      const res = await fetch(url + "/api/ping", { method: "POST" });
      await assert(res.ok, "ping 失败");
      const j = await res.json();
      await assert(j && j.ok === true, "ping 返回不符合预期");
    }

    {
      const form = new FormData();
      const content = "hello txt 上传\n第二行";
      const blob = new Blob([content], { type: "text/plain" });
      form.append("file", blob, "sample.txt");

      const res = await fetch(url + "/api/upload", { method: "POST", body: form });
      await assert(res.ok, "txt 上传失败");
      const j = await res.json();
      await assert(j && typeof j.text === "string", "txt 返回缺少 text");
      await assert(j.text.includes("hello txt 上传"), "txt 内容不匹配");
    }

    {
      const form = new FormData();
      const blob = new Blob(["not a doc"], { type: "application/msword" });
      form.append("file", blob, "legacy.doc");
      const res = await fetch(url + "/api/upload", { method: "POST", body: form });
      await assert(!res.ok, ".doc 应该被拒绝");
      const t = await res.text();
      await assert(/不支持/.test(t), ".doc 错误信息不符合预期");
    }

    {
      const form = new FormData();
      const blob = new Blob(["not a real docx"], {
        type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      });
      form.append("file", blob, "broken.docx");
      const res = await fetch(url + "/api/upload", { method: "POST", body: form });
      await assert(!res.ok, "损坏 docx 应该返回 400");
      const t = await res.text();
      await assert(/DOCX 解析失败/.test(t), "docx 错误信息不符合预期");
    }

    console.log("SELFTEST_OK");
  } finally {
    if (proc && !proc.killed) proc.kill();
  }
}

main().catch((e) => {
  console.error("SELFTEST_FAIL:", e?.stack || String(e));
  process.exitCode = 1;
});

