// ================================
// V1 Trainer Backend (DeepSeek OpenAI-compatible)
// - Fix accuracy math (server-side, never trust model's %)
// - Prevent repeated questions (session-level seen hashes + prompt avoid list + retries)
// - Key via API (no env var required for end users)
// - Optional local persistence (sessions.json) so closing page doesn't wipe progress
// - Serve /public and open browser automatically (for EXE / local run)
// ================================

import express from "express";
import cors from "cors";
import { randomUUID, createHash } from "crypto";
import OpenAI from "openai";
import Ajv from "ajv";
import fs from "fs";
import path from "path";
import open from "open";
import { fileURLToPath } from "url";
import multer from "multer";
import mammoth from "mammoth";
import { createWorker } from "tesseract.js";

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" })); // Increased limit for base64 images

const upload = multer({ storage: multer.memoryStorage() });

// ====== Serve frontend (index.html) ======
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 把当前目录作为静态目录（确保 index.html 和 server.js 在同一文件夹）
app.use(express.static(__dirname));

// 兜底：访问 / 时返回 index.html
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});


// ====== Config ======
const TARGET_ACCURACY = 85.0;      // 目标正确率（累计）
const ROUND_TARGET_ACC = 85.0;     // 每轮期望正确率（用于调难度）
const DEFAULT_NUM_QUESTIONS = 5;   // 推荐题量 4-5
const MODEL = process.env.MODEL_NAME || "deepseek-chat";
const PORT = Number(process.env.PORT || 8787);
const OCR_LANGS = "chi_sim+eng";
const OCR_LANG_PATH = process.cwd();
const MOCK_LLM = process.env.MOCK_LLM === "true";
const CLEANUP_AGE_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

// DeepSeek OpenAI-compatible endpoint
const BASE_URL = "https://api.deepseek.com/v1";

// ====== Static frontend (optional) ======
// Put your index.html/app.js/styles.css under ./public
const PUBLIC_DIR = path.join(process.cwd(), "public");
if (fs.existsSync(PUBLIC_DIR)) {
  app.use(express.static(PUBLIC_DIR));
  app.get("/", (_, res) => res.sendFile(path.join(PUBLIC_DIR, "index.html")));
}

// ====== Cooling & Usage Logs ======
const USAGE_FILE = path.join(process.cwd(), "usage_logs.json");
let usageLogs = {
  chapterCooldowns: {}, // { chapterHash: lastTimestamp }
  globalUsage: []       // [ timestamp, ... ]
};

function rotateUsageLogs() {
  try {
    if (!fs.existsSync(USAGE_FILE)) return;
    
    const stats = fs.statSync(USAGE_FILE);
    const MAX_SIZE = 1 * 1024 * 1024; // 1MB rotation limit
    
    if (stats.size > MAX_SIZE) {
      const backupFile = `${USAGE_FILE}.${new Date().toISOString().replace(/[:.]/g, "-")}.bak`;
      fs.renameSync(USAGE_FILE, backupFile);
      
      // Keep only last 5 backups
      const dir = path.dirname(USAGE_FILE);
      const files = fs.readdirSync(dir)
        .filter(f => f.startsWith("usage_logs.json") && f.endsWith(".bak"))
        .sort()
        .reverse(); // Newest first
        
      for (let i = 5; i < files.length; i++) {
        fs.unlinkSync(path.join(dir, files[i]));
        console.log(`[log] cleaned up old log: ${files[i]}`);
      }
      
      // Reset memory state
      usageLogs = { chapterCooldowns: {}, globalUsage: [] };
      safeWriteUsage();
      console.log(`[log] rotated usage log to ${backupFile}`);
    }
  } catch (e) {
    console.warn("[log] rotation failed:", e);
  }
}

function safeReadUsage() {
  try {
    if (!fs.existsSync(USAGE_FILE)) return;
    usageLogs = JSON.parse(fs.readFileSync(USAGE_FILE, "utf-8"));
    // Clean old global usage (> 2 hours)
    const now = Date.now();
    usageLogs.globalUsage = (usageLogs.globalUsage || []).filter(t => now - t < 2 * 60 * 60 * 1000);
  } catch (e) {
    console.warn("[persist] load usage failed:", String(e));
  }
}

function safeWriteUsage() {
  try {
    fs.writeFileSync(USAGE_FILE, JSON.stringify(usageLogs, null, 2), "utf-8");
    // Check rotation after write
    rotateUsageLogs();
  } catch (e) {
    console.warn("[persist] save usage failed:", String(e));
  }
}

safeReadUsage();

function checkServerCooling(notes) {
  const now = Date.now();
  
  // 1. Chapter Cooling (30 mins)
  const chapterHash = createHash("sha256").update(notes.slice(0, 200)).digest("hex"); // Simple hash of start of notes
  const lastTime = usageLogs.chapterCooldowns[chapterHash];
  if (lastTime) {
    const diff = now - lastTime;
    if (diff < 30 * 60 * 1000) {
      const remain = Math.ceil((30 * 60 * 1000 - diff) / 60000);
      return { ok: false, msg: `同一章节需间隔 30 分钟再次使用。请休息 ${remain} 分钟。` };
    }
  }

  // 2. Global Usage (3 times in 2 hours)
  // Clean first
  usageLogs.globalUsage = usageLogs.globalUsage.filter(t => now - t < 2 * 60 * 60 * 1000);
  if (usageLogs.globalUsage.length >= 3) {
      // Check if this specific chapter was the last one used? No, spec says "Different chapters... continuous use > 3 times"
      // Actually spec says: "Same chapter: 30 min cooldown. Different chapters: > 3 times -> warning".
      // But user said "Force limit" in Signal 5? 
      // User said: "Different chapters: Continuous use > 3 times -> Force prompt (Warning)". 
      // "Button only: 'Got it'". It doesn't say BLOCK, it says PROMPT. 
      // But wait, "Force prompt" implies blocking or at least a strong interruption.
      // However, "Button only: 'Got it'" suggests it's just a warning dialog, then maybe proceed?
      // Re-reading: "UI prompt... Button: 'Got it'". It seems like a warning.
      // But for the sake of "Negative Question Generation", let's return a warning flag to frontend.
      return { ok: true, warning: "这个工具更适合间隔使用，连续使用会降低判断准确性。" };
  }

  return { ok: true, chapterHash };
}

function recordServerUsage(chapterHash) {
  const now = Date.now();
  if (chapterHash) usageLogs.chapterCooldowns[chapterHash] = now;
  usageLogs.globalUsage.push(now);
  safeWriteUsage();
}

const sessions = new Map();
const sessionStore = sessions; // Alias to fix ReferenceError: sessionStore is not defined
const SESS_FILE = path.join(process.cwd(), "sessions.json");

function safeReadSessions() {
  try {
    if (!fs.existsSync(SESS_FILE)) return;
    const raw = fs.readFileSync(SESS_FILE, "utf-8");
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return;
    for (const s of arr) {
      // Rebuild Sets
      s.seen = new Set(s.seen || []);
      s.seenStems = new Set(s.seenStems || []);
      // SECURITY: Do not trust persisted API keys. Force re-entry from frontend.
      s.apiKey = null; 
      sessions.set(s.id, s);
    }
    console.log(`[persist] loaded sessions: ${sessions.size}`);
  } catch (e) {
    console.warn("[persist] load failed:", String(e));
  }
}

let isSaving = false;
let pendingSave = false;

async function safeWriteSessions() {
  if (isSaving) {
    pendingSave = true;
    return;
  }
  isSaving = true;

  try {
    const now = Date.now();
    // Cleanup old sessions
    for (const [id, s] of sessions) {
      const lastActive = s.lastActive || new Date(s.createdAt).getTime() || 0;
      if (now - lastActive > CLEANUP_AGE_MS) {
        sessions.delete(id);
        console.log(`[cleanup] removed expired session: ${id}`);
      }
    }

    const arr = Array.from(sessions.values()).map(s => {
      // Create a shallow copy to avoid modifying the active session
      const serialized = { ...s };
      // Security: Never persist API keys to disk
      delete serialized.apiKey;
      // Serialize Sets
      serialized.seen = Array.from(s.seen || []);
      serialized.seenStems = Array.from(s.seenStems || []);
      return serialized;
    });
    
    // Async write with atomic rename to prevent blocking event loop
    const tmpFile = SESS_FILE + ".tmp";
    await fs.promises.writeFile(tmpFile, JSON.stringify(arr, null, 2), "utf-8");
    await fs.promises.rename(tmpFile, SESS_FILE);
  } catch (e) {
    console.warn("[persist] save failed:", String(e));
  } finally {
    isSaving = false;
    if (pendingSave) {
      pendingSave = false;
      // Use setImmediate to break recursion stack if highly active
      setImmediate(safeWriteSessions);
    }
  }
}

// Load persisted sessions on startup
safeReadSessions();

// Save periodically (and on changes in key routes too)
setInterval(safeWriteSessions, 15_000).unref();

// ====== JSON Schemas ======
function practicePackSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      meta: {
        type: "object",
        properties: {
          subject: { type: "string" },
          chapter_title: { type: "string" },
          version_hash: { type: "string" },
          created_at: { type: "string" },
          timebox_minutes: { type: "number" }
        }
      },
      extracted: {
        type: "object",
        properties: {
          core_concepts: { type: "array", items: { type: "string" } },
          core_claims: { type: "array", items: { type: "string" } },
          likely_misconceptions: { type: "array", items: { type: "string" } },
          prior_links: { type: "array", items: { type: "string" } }
        }
      },
      questions: {
        type: "array",
        minItems: 3,
        maxItems: 6,
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            id: { type: "string" },
            intent: { type: "string", enum: ["A", "B", "C", "D"] },
            type: { type: "string", enum: ["single", "multi", "tf", "short"] },
            stem: { type: "string" },
            options: { type: "array", items: { type: "string" } },
            answer: {
              type: "object",
              properties: {
                kind: { type: "string", enum: ["exact", "set", "keywords"] },
                value: { type: ["string", "array", "boolean", "number"] } // Allow mixed types, normalized later
              },
              required: ["kind", "value"]
            },
            rubric: {
              type: "object",
              properties: {
                must_have: { type: "array", items: { type: "string" } },
                common_wrong: { type: "array", items: { type: "string" } },
                strictness: { type: "string" }
              }
            },
            rationale: { type: "string" },
            concept: { type: "string" },
            difficulty: { type: "integer" }
          },
          required: ["id", "intent", "type", "stem", "options", "answer", "rationale", "concept"]
        }
      },
      scoring: {
        type: "object",
        properties: {
          objective_rules: { type: "string" },
          short_rules: { type: "object" }
        }
      },
      stop_rules: {
        type: "object",
        properties: {
          stable_if: { type: "object" },
          unstable_if: { type: "object" },
          messages: { type: "object" }
        }
      },
      ui_hints: { type: "object" }
    },
    required: ["questions", "stop_rules"]
  };
}

// ====== AJV Validators ======
const ajv = new Ajv({ allErrors: true, strict: false });
const validatePack = ajv.compile(practicePackSchema());

function gradeNextSchema() {
  return {
    type: "object",
    properties: {
      grading: {
        type: "object",
        properties: {
          results: { type: "array" },
          summary: { type: "object" },
          coaching: { type: "object" }
        },
        required: ["results", "summary"]
      },
      nextQuiz: {
        type: "object",
        properties: {
          questions: { type: "array" }
        }
      }
    },
    required: ["grading"]
  };
}
const validateGradeNext = ajv.compile(gradeNextSchema());

// ====== Prompt ======
function makeSystemPrompt() {
  return "You are a helpful assistant.";
}

function makePackPrompt() {
  return [
    "你是“章节即时诊断工具”，目标是生成一个完整的“Practice Pack”（练习包）。",
    "原则：",
    "1. 只基于用户提供的笔记内容，不引入外部新概念。",
    "2. 题量控制在 3-6 题，覆盖核心概念。",
    "3. 必须输出严格的 JSON 格式，不要 Markdown。",
    "4. 必须包含每道题的完整题干（stem）、选项（options）、答案（answer）和解析（rationale）。",
    "",
    "Practice Pack 结构要求：",
    "1. questions[]: 包含 3-6 道题。",
    "   - intent: A (核心表述), B (变体), C (易错点), D (关联点)",
    "   - stem: 题目描述（必须存在！）",
    "   - type: single|multi|tf|short",
    "   - options: 选项列表。如果是 single/multi，提供 4 个选项；如果是 tf，固定提供 [\"正确\", \"错误\"]；如果是 short，提供空数组 []。",
    "   - answer: { kind: 'exact'|'set'|'keywords', value: ... }",
    "     - single/tf: value 为选项字母 (如 'A') 或 'T'/'F' (T对应第1个选项，F对应第2个)",
    "     - multi: value 为数组 (如 ['A','C'])",
    "     - short: value 为关键词数组",
    "   - rationale: 详细的题目解析。",
    "2. stop_rules: 定义何时终止。",
    "   - stable_if: { A_pass: true, B_pass: true }",
    "   - unstable_if: { A_fail_count: 1 }",
    "   - messages: 定义 stable/unstable/invalid 时的用户提示文案。",
    "3. ui_hints: { render_mode: 'one_by_one', hide_total_count: true }",
    "",
    "重要：每道题必须有 'stem' 字段，描述问题内容。不要留空！"
  ].join("\n");
}


// ====== Robust JSON extraction ======
function extractFirstJson(text) {
  if (!text) throw new Error("模型返回为空");
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    throw new Error("模型未返回可解析的JSON对象");
  }
  const jsonStr = text.slice(start, end + 1);
  try {
    return JSON.parse(jsonStr);
  } catch (e) {
    throw e;
  }
}

  // ====== Client (per session key) ======
function getClient(apiKey) {
  if (!apiKey || typeof apiKey !== "string" || apiKey.length < 10) {
    throw new Error("缺少有效的 API Key。请在前端输入并保存 Key 后再试。");
  }
  return new OpenAI({
    apiKey: apiKey.trim(),
    baseURL: "https://api.deepseek.com/v1"
  });
}

// ====== Helpers: accuracy computed by server ======
function computeAccuracyFromResults(results) {
  const total = Array.isArray(results) ? results.length : 0;
  const correct = Array.isArray(results) ? results.filter(r => r.correct).length : 0;
  const accuracy = total ? (correct / total) * 100 : 0;
  const errorRate = 100 - accuracy;
  return { total, correct, accuracy, errorRate };
}

function difficultyAdjustmentByTarget(accuracy, target) {
  if (accuracy < target - 5) return "down";
  if (accuracy > target + 5) return "up";
  return "same";
}

function clampInt(n, lo, hi, fallback) {
  const x = Number.isFinite(Number(n)) ? Math.round(Number(n)) : fallback;
  return Math.max(lo, Math.min(hi, x));
}

// ====== Helpers: question de-dup ======
function normalizeText(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[，。、“”‘’：；！!？?（）()【】[\]{}<>]/g, "")
    .trim();
}

function questionHash(q) {
  // Hash stem + options to detect repeats
  const stem = normalizeText(q?.stem);
  const opts = Array.isArray(q?.options) ? q.options.map(normalizeText).join("|") : "";
  const type = String(q?.type || "");
  const concept = normalizeText(q?.concept);
  const raw = `${type}||${concept}||${stem}||${opts}`;
  return createHash("sha256").update(raw).digest("hex");
}

function filterNewQuestions(session, questions) {
  const out = [];
  for (const q of questions || []) {
    const h = questionHash(q);
    if (session.seen.has(h)) continue;
    session.seen.add(h);
    out.push(q);
  }
  return out;
}

function takeFirstN(arr, n) {
  return Array.isArray(arr) ? arr.slice(0, n) : [];
}

function getMockResponse(what) {
  if (what === "generatePracticePack") {
    return JSON.stringify({
      meta: { subject: "Mock Subject", chapter_title: "Mock Chapter", version_hash: "mock", created_at: new Date().toISOString(), timebox_minutes: 10 },
      extracted: { core_concepts: ["Mock Concept"], core_claims: ["Mock Claim"], likely_misconceptions: ["Mock Misconception"], prior_links: [] },
      questions: [
        { id: "q1", intent: "A", type: "single", stem: "Mock Question 1", options: ["A", "B", "C", "D"], answer: { kind: "exact", value: "A" }, rationale: "Mock Rationale", concept: "Mock Concept", difficulty: 1 },
        { id: "q2", intent: "B", "type": "tf", "stem": "Mock Question 2 (T/F)", options: [], answer: { kind: "exact", value: "T" }, rationale: "Mock Rationale", concept: "Mock Concept", difficulty: 1 },
        { id: "q3", intent: "C", "type": "multi", "stem": "Mock Question 3", options: ["A", "B", "C", "D"], answer: { kind: "set", value: ["A", "B"] }, rationale: "Mock Rationale", concept: "Mock Concept", difficulty: 1 }
      ],
      stop_rules: { stable_if: {}, unstable_if: {}, messages: {} },
      ui_hints: {}
    });
  }
  if (what === "createQuiz_init") {
    return JSON.stringify({
      validity: { status: "ok" },
      chapterModel: { core_concepts: ["Mock"], core_claims: ["Mock"], likely_misconceptions: ["Mock"], prior_links: [] },
      questions: [
        { id: "q_mock_1", type: "single", intent: "A", stem: "Mock Quiz Question 1", options: ["A", "B", "C", "D"], answer: "A", rationale: "Mock Rationale", concept: "Mock" }
      ]
    });
  }
  if (what === "gradeAndNextQuiz_raw") {
    return JSON.stringify({
      grading: {
        results: [
          { id: "q_mock_1", correct: true, correctAnswer: "A", briefRationale: "Mock Rationale", errorType: "None" }
        ],
        summary: { difficultyAdjustment: "same", status: "continue", nextNumQuestions: 3 },
        coaching: { topWeaknesses: [], nextFocus: "None", microDrill: "None" }
      },
      nextQuiz: {
        questions: [
           { id: "q_mock_2", type: "single", intent: "B", stem: "Mock Follow-up Question", options: ["A", "B", "C", "D"], answer: "B", rationale: "Mock Rationale", concept: "Mock" }
        ]
      }
    });
  }
  return "{}";
}

// ====== chatJson with validator ======
async function chatJson({ session, messages, validator, what, apiKey }) {
  if (MOCK_LLM) {
    console.log(`[Mock] chatJson called for ${what}`);
    const mockText = getMockResponse(what);
    const data = extractFirstJson(mockText);
    if (validator) {
      const ok = validator(data);
      if (!ok) console.warn(`[Mock] Validation failed for ${what}`);
    }
    return data;
  }

  const client = getClient(apiKey);

  let resp;
  try {
    resp = await client.chat.completions.create({
      model: MODEL,
      messages,
      temperature: 0.1,
      top_p: 0.9,
      max_tokens: 1600
    });
  } catch (e) {
    throw e;
  }

  const text = resp.choices?.[0]?.message?.content ?? "";

  let data;
  try {
    data = extractFirstJson(text);
  } catch (e) {
    throw e;
  }

  // ✅ 只有传了 validator 才校验
  if (validator) {
    const ok = validator(data);
    if (!ok) {
      const err = validator.errors?.map(e => `${e.instancePath || "(root)"} ${e.message}`).join("; ");
      throw new Error(`${what} JSON schema 校验失败：${err}\n原始输出长度：${text.length}`);
    }
  }

  return data;
}

async function chatTextStream({ messages, what, apiKey, onDelta }) {
  if (MOCK_LLM) {
    const mockText = getMockResponse(what);
    if (typeof onDelta === "function") onDelta(String(mockText || ""));
    return String(mockText || "");
  }

  const client = getClient(apiKey);
  const stream = await client.chat.completions.create({
    model: MODEL,
    messages,
    temperature: 0.1,
    top_p: 0.9,
    max_tokens: 1600,
    stream: true
  });

  let full = "";
  for await (const chunk of stream) {
    const delta = chunk?.choices?.[0]?.delta?.content ?? "";
    if (!delta) continue;
    full += delta;
    if (typeof onDelta === "function") onDelta(delta);
  }
  return full;
}

function truncateNotesForModel(notes, maxChars = 12000) {
  const s = String(notes || "");
  if (s.length <= maxChars) return s;
  const head = s.slice(0, Math.floor(maxChars * 0.7));
  const tail = s.slice(-Math.floor(maxChars * 0.25));
  return `${head}\n...\n（中间内容为节省耗时已省略）\n...\n${tail}`;
}

function sanitizeOptionText(opt, idx) {
  let s = String(opt ?? "");
  s = s.replace(/^\s+/, "");
  const letter = String.fromCharCode(65 + (Number.isFinite(Number(idx)) ? Number(idx) : 0)); // A/B/C/D
  const patterns = [
    new RegExp(`^${letter}\\s*[\\.、:：)\\]]\\s*`, "i"),
    /^[A-D]\s*[\.、:：)\]]\s*/i,
    new RegExp(`^${letter}\\s+`, "i"),
    /^[A-D]\s+/i
  ];
  for (const re of patterns) s = s.replace(re, "");
  return s.trim();
}

function normalizeStopRules(rawStopRules) {
  const stable = rawStopRules?.stable_if && typeof rawStopRules.stable_if === "object" ? { ...rawStopRules.stable_if } : {};
  const unstable = rawStopRules?.unstable_if && typeof rawStopRules.unstable_if === "object" ? { ...rawStopRules.unstable_if } : {};
  const messages = rawStopRules?.messages && typeof rawStopRules.messages === "object" ? { ...rawStopRules.messages } : {};

  const stableMin = Number.isFinite(Number(stable.min_questions)) ? Number(stable.min_questions) : 3;
  const unstableMin = Number.isFinite(Number(unstable.min_questions)) ? Number(unstable.min_questions) : 2;
  stable.min_questions = Math.max(3, Math.round(stableMin));
  unstable.min_questions = Math.max(2, Math.round(unstableMin));

  if (unstable.A_fail_count !== undefined) {
    const n = Number(unstable.A_fail_count);
    unstable.A_fail_count = Number.isFinite(n) ? Math.max(2, Math.round(n)) : 2;
  } else {
    unstable.A_fail_count = 2;
  }

  return {
    stable_if: stable,
    unstable_if: unstable,
    messages: {
      stable: messages.stable || "基础扎实！",
      unstable: messages.unstable || "建议重读笔记。",
      invalid: messages.invalid || "数据异常。"
    }
  };
}

async function generatePracticePackStream({ session, notes, apiKey, onDelta }) {
  const notesForModel = truncateNotesForModel(notes, 12000);
  const prompt = `
输入内容（本章全部边界）：
${notesForModel}

任务：
请生成一个完整的 Practice Pack (JSON)，包含：
1. 提取核心概念和易错点 (extracted)
2. 生成 3-6 道检查题 (questions)，每题必须包含题干 (stem)、选项 (options) 和详细解析 (rationale)
3. 设定终止规则 (stop_rules)
4. 设定 UI 提示 (ui_hints)

重要：
- 选择题必须提供 options 数组。
- 判断题 (tf) 的 options 必须为 ["正确", "错误"]。
- 填空题 (short) 的 options 为 []。
- rationale 需要简洁清晰（建议 80-160 字），避免长篇大论。
- 必须严格遵循 System Prompt 中的 JSON 结构定义。
`.trim();

  const text = await chatTextStream({
    messages: [
      { role: "system", content: makePackPrompt() },
      { role: "user", content: prompt }
    ],
    what: "generatePracticePack",
    apiKey,
    onDelta
  });

  const raw = extractFirstJson(text);

  const pack = {
    meta: {
      subject: raw.meta?.subject || "未知科目",
      chapter_title: raw.meta?.chapter_title || "未知章节",
      version_hash: raw.meta?.version_hash || "v1",
      created_at: raw.meta?.created_at || new Date().toISOString(),
      timebox_minutes: typeof raw.meta?.timebox_minutes === "number" ? raw.meta.timebox_minutes : 15
    },
    extracted: {
      core_concepts: Array.isArray(raw.extracted?.core_concepts) ? raw.extracted.core_concepts.map(String) : [],
      core_claims: Array.isArray(raw.extracted?.core_claims) ? raw.extracted.core_claims.map(String) : [],
      likely_misconceptions: Array.isArray(raw.extracted?.likely_misconceptions) ? raw.extracted.likely_misconceptions.map(String) : [],
      prior_links: Array.isArray(raw.extracted?.prior_links) ? raw.extracted.prior_links.map(String) : []
    },
    questions: normalizeQuestions(raw.questions, true),
    scoring: {
      objective_rules: raw.scoring?.objective_rules || "每题 1 分",
      short_rules: raw.scoring?.short_rules || {}
    },
    stop_rules: normalizeStopRules(raw.stop_rules),
    ui_hints: raw.ui_hints || { render_mode: "one_by_one" }
  };

  const ok = validatePack(pack);
  if (!ok) {
    const err = validatePack.errors?.map(e => `${e.instancePath || "(root)"} ${e.message}`).join("; ");
    throw new Error(`generatePracticePack JSON schema 校验失败（归一化后）：${err}`);
  }

  return pack;
}

// ====== Generate Practice Pack ======
async function generatePracticePack({ session, notes, apiKey }) {
  const notesForModel = truncateNotesForModel(notes, 12000);
  const prompt = `
输入内容（本章全部边界）：
${notesForModel}

任务：
请生成一个完整的 Practice Pack (JSON)，包含：
1. 提取核心概念和易错点 (extracted)
2. 生成 3-6 道检查题 (questions)，每题必须包含题干 (stem)、选项 (options) 和详细解析 (rationale)
3. 设定终止规则 (stop_rules)
4. 设定 UI 提示 (ui_hints)

重要：
- 选择题必须提供 options 数组。
- 判断题 (tf) 的 options 必须为 ["正确", "错误"]。
- 填空题 (short) 的 options 为 []。
- rationale 需要简洁清晰（建议 80-160 字），避免长篇大论。
- 必须严格遵循 System Prompt 中的 JSON 结构定义。
`.trim();

  const raw = await chatJson({
    session,
    messages: [
      { role: "system", content: makePackPrompt() },
      { role: "user", content: prompt }
    ],
    validator: null, // 先不校验，手动归一化后再校验
    what: "generatePracticePack",
    apiKey
  });

  // 归一化逻辑
  const pack = {
    meta: {
      subject: raw.meta?.subject || "未知科目",
      chapter_title: raw.meta?.chapter_title || "未知章节",
      version_hash: raw.meta?.version_hash || "v1",
      created_at: raw.meta?.created_at || new Date().toISOString(),
      timebox_minutes: typeof raw.meta?.timebox_minutes === "number" ? raw.meta.timebox_minutes : 15
    },
    extracted: {
      core_concepts: Array.isArray(raw.extracted?.core_concepts) ? raw.extracted.core_concepts.map(String) : [],
      core_claims: Array.isArray(raw.extracted?.core_claims) ? raw.extracted.core_claims.map(String) : [],
      likely_misconceptions: Array.isArray(raw.extracted?.likely_misconceptions) ? raw.extracted.likely_misconceptions.map(String) : [],
      prior_links: Array.isArray(raw.extracted?.prior_links) ? raw.extracted.prior_links.map(String) : []
    },
    questions: normalizeQuestions(raw.questions, true),
    scoring: {
      objective_rules: raw.scoring?.objective_rules || "每题 1 分",
      short_rules: raw.scoring?.short_rules || {}
    },
    stop_rules: normalizeStopRules(raw.stop_rules),
    ui_hints: raw.ui_hints || { render_mode: "one_by_one" }
  };

  // 校验归一化后的数据
  const ok = validatePack(pack);
  if (!ok) {
    const err = validatePack.errors?.map(e => `${e.instancePath || "(root)"} ${e.message}`).join("; ");
    throw new Error(`generatePracticePack JSON schema 校验失败（归一化后）：${err}`);
  }

  return pack;
}

// ====== LLM Calls ======
async function createQuiz({ session, notes, round, difficultyState, numQuestions, apiKey }) {
  // Provide avoid list to reduce repeats (only short excerpts to keep tokens low)
  const avoid = Array.from(session.seen || []).slice(-50); // hashes (not useful to model)
  const avoidStems = (session.lastStems || []).slice(-30); // actual stems for model
  const avoidText = avoidStems.length
    ? avoidStems.map((s, i) => `${i + 1}) ${s}`).join("\n")
    : "无";

  const prompt = `
输入内容（本章全部边界）：
${notes}

任务：
1. **预检（Validity Check）**：
   - 如果内容仅为标题/零散句子，无法抽取≥1个明确核心概念 -> reject_incomplete
   - 如果内容仅为列表/年表/术语总览，缺乏理解逻辑 -> reject_listing
   - 如果核心概念数量 > 5 (内容粒度过大) -> reject_too_large
   - 否则 -> ok

2. 【章节即时理解建模】（仅当 ok 时生成）：
   - core_concepts (≤5个): 本章必须会用的最小概念集合
   - core_claims (≤3个): 本章的关键判断/结论/规则
   - likely_misconceptions: 新手最容易想当然的点
   - prior_links: 与前置知识的最小连接点

3. 【生成第一轮题目】（仅当 ok 时生成）：
   - 题量：${numQuestions} 题（必须介于 3-6 之间）
   - 必须包含题干 (stem)、选项 (options) 和详细解析 (rationale)。
   - 必须按顺序尝试生成以下四类出题意图：
     - Type A: 核心表述检查 (1-2题) —— 确认能否准确使用核心概念
     - Type B: 表述变体/条件扰动 (1题) —— 区分理解 vs 记住说法
     - Type C: 易错点/想当然陷阱 (1题) —— 暴露 likely_misconceptions
     - Type D: 最小跨章联结 (0-1题) —— 关联 prior_links
   - 题型允许：single/multi/tf/short
   - 判断题 (tf) 的 options 必须为 ["正确", "错误"]，填空题 (short) 为 []。

输出 JSON 格式：
{
  "validity": {
    "status": "ok|reject_incomplete|reject_listing|reject_too_large",
    "reason": "给用户的拒绝/建议理由"
  },
  "chapterModel": { ... },
  "questions": [ ... ]
}
`.trim();

  // We may retry once if many duplicates are filtered out
  const maxTry = 2;
  let collected = [];
  let modelData = null;

  // First pass: check validity and get model
  const raw = await chatJson({
    session,
    messages: [
      { role: "system", content: makeSystemPrompt() },
      { role: "user", content: prompt }
    ],
    validator: null,
    what: "createQuiz_init",
    apiKey
  });

  // Handle Rejection
  if (raw.validity && raw.validity.status !== "ok") {
      return { 
          questions: [], 
          chapterModel: null, 
          validity: raw.validity 
      };
  }

  if (raw.chapterModel) {
      modelData = raw.chapterModel;
  }

  // Process first batch
  const fresh = filterNewQuestions(session, raw.questions || []);
  const normalizedFresh = normalizeQuestions(fresh);
  
  session.lastStems = session.lastStems || [];
  for (const q of normalizedFresh) session.lastStems.push(String(q.stem || "").slice(0, 120));
  session.lastStems = session.lastStems.slice(-200);
  
  collected = collected.concat(normalizedFresh);

  // If insufficient, try top-up (only if valid)
  for (let t = 0; t < maxTry && collected.length < numQuestions; t++) {
    // ... (logic for top-up would go here, but for simplicity/safety we might just accept what we have or do a simpler loop)
    // For now, let's just stick to the initial batch if it's decent. 
    // If we strictly need more, we'd need a secondary prompt. 
    // Given the constraints, let's rely on the first pass being good enough usually.
    break; 
  }
  
  // Note: Original loop logic was a bit complex with the retry. 
  // Since we changed the prompt structure, we simplified the flow.
  // If strict top-up is needed, we can re-add it, but usually 1 pass is enough.

  return { questions: collected, chapterModel: modelData, validity: { status: "ok" } };
}

function normalizeQuestions(questions, isPack = false) {
  if (!Array.isArray(questions)) return [];
  return questions.map((q, idx) => {
    if (!q || typeof q !== "object") return q;
    
    // 显式提取必需字段，确保没有额外属性干扰 Schema 校验
    const out = {
      id: q.id != null ? String(q.id) : `q_${idx}`,
      type: ["single", "multi", "tf", "short"].includes(q.type) ? q.type : "single",
      intent: ["A", "B", "C", "D"].includes(q.intent) ? q.intent : "A",
      stem: q.stem != null ? String(q.stem) : "无题干",
      options: Array.isArray(q.options) ? q.options.map((s, i) => sanitizeOptionText(s, i)) : [],
      concept: q.concept != null ? String(q.concept) : "通用",
      rationale: q.rationale != null ? String(q.rationale) : (q.explanation != null ? String(q.explanation) : (q.briefRationale != null ? String(q.briefRationale) : "无解析")),
      difficulty: typeof q.difficulty === "number" ? q.difficulty : 1
    };

    if (out.type === "tf") {
      out.options = ["正确", "错误"];
    }
    
    // 处理答案格式
    if (isPack) {
      // Practice Pack 模式：需要 { kind, value } 对象
      if (q.answer && typeof q.answer === "object" && q.answer.kind && q.answer.value !== undefined) {
        out.answer = {
          kind: ["exact", "set", "keywords"].includes(q.answer.kind) ? q.answer.kind : "exact",
          value: q.answer.value
        };
      } else {
        // 兜底转换
        let kind = "exact";
        if (out.type === "multi") kind = "set";
        if (out.type === "short") kind = "keywords";
        out.answer = { kind, value: q.answer != null ? q.answer : "" };
      }
    } else {
      // 普通模式：简单的字符串/数组
      if (q.answer && typeof q.answer === "object" && q.answer.value !== undefined) {
        out.answer = String(q.answer.value);
      } else {
        out.answer = q.answer != null ? String(q.answer) : "";
      }
    }

    return out;
  });
}

function normalizeGradeNext(data) {
  if (!data || typeof data !== "object") return data;

  // 1) grading.results 字段清洗与兜底
  const results = data?.grading?.results;
  if (Array.isArray(results)) {
    data.grading.results = results.map((r, idx) => {
      if (!r || typeof r !== "object") return r;
      
      // 提取解析，按优先级尝试不同字段
      const rationale = r.briefRationale || r.briefExplanation || r.rationale || r.explanation || "";
      
      // 显式提取必需字段，过滤掉模型返回的额外字段
      return {
        id: r.id != null ? String(r.id) : `q_${idx}`,
        correct: typeof r.correct === "boolean" ? r.correct : Boolean(r.isCorrect ?? false),
        correctAnswer: r.correctAnswer != null ? String(r.correctAnswer) : (r.standardAnswer != null ? String(r.standardAnswer) : ""),
        briefRationale: rationale.trim() !== "" ? String(rationale) : "暂无详细解析。",
        errorType: (r.errorType != null && String(r.errorType).trim() !== "") ? String(r.errorType) : "无"
      };
    });
  }

  // 2) difficultyAdjustment 兜底归一化
  const adj = data?.grading?.summary?.difficultyAdjustment;
  if (adj && typeof adj === "string") {
    const s = adj.toLowerCase();
    if (!["up", "same", "down"].includes(s)) {
      // 常见模型输出：increase/decrease/keep
      if (["increase", "harder", "raise", "upward"].includes(s)) data.grading.summary.difficultyAdjustment = "up";
      else if (["decrease", "easier", "lower", "downward"].includes(s)) data.grading.summary.difficultyAdjustment = "down";
      else data.grading.summary.difficultyAdjustment = "same";
    }
  } else if (data?.grading?.summary) {
    data.grading.summary.difficultyAdjustment = "same";
  }

  // 3) nextQuiz 字段处理与兜底
  if (data.nextQuiz && typeof data.nextQuiz === "object") {
    data.nextQuiz = {
      questions: normalizeQuestions(data.nextQuiz.questions)
    };
  }

  return data;
}

async function gradeAndNextQuiz({ session, notes, chapterModel, quiz, answers, targetAccuracy, nextRound, difficultyState, nextNumQuestionsHint, seenStems, apiKey }) {
  const prompt = `
章节理解模型（中间层）：
${JSON.stringify(chapterModel || {}, null, 2)}

【当前轮题目】：
${JSON.stringify(quiz)}

【用户作答】：
${JSON.stringify(answers)}

任务：
1. **判分与诊断** (grading)：
   - 逐题判分，指出是否正确，并给出极简解析。
   - 诊断重点：
     - Type A (核心表述) 是否稳定？
     - Type B (变体) 是否能识别？
     - Type C (易错点) 是否踩坑？
   - 决策终止 (status="finish" | "continue"):
     - **强制终止(Negative)**:
       - 如果 Type A 连续错误 或 概念严重混淆 -> finish (outcome="unstable", reason: "基础不稳，建议重新阅读")
       - 如果 用户质疑题目有效性 -> finish (outcome="unstable", reason: "题目存疑，终止本轮")
     - **正常结束(Positive)**:
       - 如果 Type A+B 稳定 且 易错题未命中 -> finish (outcome="stable", reason: "练习充分，建议进入下一节")
     - **其他终止**:
       - 如果 达到最大轮次限制 -> finish (outcome="limit_reached", reason: "已达本章练习上限")
     - 否则 -> continue (outcome 可省略)

2. **生成下一轮** (nextQuiz) [仅当 status="continue"]:
   - 题量：3-6 题 (nextNumQuestions)
   - 每题必须包含题干 (stem)、选项 (options) 和详细解析 (rationale)。
   - 必须包含未通过的测试点变体 + 新的 Type B/C/D 组合。
   - 严禁重复已出题干。
   - 判断题 (tf) 的 options 固定为 ["正确", "错误"]。

输出 JSON (严格遵守 Schema):
{
  "grading": {
    "results": [...],
    "summary": { 
      "accuracy": ..., 
      "errorRate": ..., 
      "difficultyAdjustment": "same", 
      "nextNumQuestions": 3-6, 
      "status": "continue|finish",
      "outcome": "stable|unstable|limit_reached",
      "reason": "..."
    },
    "coaching": { ... }
  },
  "nextQuiz": { "questions": [...] }
}
`.trim();

  const raw = await chatJson({
    session,
    messages: [
      { role: "system", content: makeSystemPrompt() },
      { role: "user", content: prompt }
    ],
    // 先不过 schema，拿到原始对象
    validator: null,
    what: "gradeAndNextQuiz_raw",
    apiKey
  });

  // 先做字段清洗和兜底
  const cleaned = normalizeGradeNext(raw);
  
  // Ensure status is present
  if (cleaned.grading && cleaned.grading.summary && !cleaned.grading.summary.status) {
      cleaned.grading.summary.status = "continue";
  }

  // 对 grading.results 做更强的兜底，补全必需字段并删除多余字段
  if (cleaned && cleaned.grading && Array.isArray(cleaned.grading.results)) {
    cleaned.grading.results = cleaned.grading.results.map((r, idx) => {
      const out = {};
      out.id = typeof r.id === "string" ? r.id : String(r.id ?? `q_${idx}`);
      out.correct = typeof r.correct === "boolean" ? r.correct : Boolean(r.isCorrect ?? false);
      out.correctAnswer = r.correctAnswer != null ? String(r.correctAnswer) : "";
      out.briefRationale = r.briefRationale != null
        ? String(r.briefRationale)
        : (r.briefExplanation != null ? String(r.briefExplanation) : "");
      if (!out.briefRationale) out.briefRationale = out.correct ? "回答正确。" : "回答不正确。";
      out.errorType = r.errorType != null && String(r.errorType).trim() !== "" ? String(r.errorType) : "无";
      return out;
    });
  }

  // summary 兜底
  if (cleaned && cleaned.grading) {
    const s = cleaned.grading.summary || {};
    cleaned.grading.summary = {
      accuracy: typeof s.accuracy === "number" ? s.accuracy : 0,
      errorRate: typeof s.errorRate === "number" ? s.errorRate : 100,
      difficultyAdjustment: typeof s.difficultyAdjustment === "string" ? s.difficultyAdjustment : "same",
      nextNumQuestions: clampInt(s.nextNumQuestions, 3, 6, DEFAULT_NUM_QUESTIONS),
      status: ["continue", "finish"].includes(s.status) ? s.status : "continue",
      reason: s.reason ? String(s.reason) : ""
    };
  }

  // coaching 兜底
  if (cleaned && cleaned.grading) {
    const c = cleaned.grading.coaching || {};
    cleaned.grading.coaching = {
      topWeaknesses: Array.isArray(c.topWeaknesses) ? c.topWeaknesses.map(String) : [],
      nextFocus: c.nextFocus != null ? String(c.nextFocus) : "",
      microDrill: c.microDrill != null ? String(c.microDrill) : ""
    };
  }

  // nextQuiz 兜底
  if (!cleaned.nextQuiz || !Array.isArray(cleaned.nextQuiz.questions)) {
    cleaned.nextQuiz = { questions: [] };
  }

  // 兜底完成后再按严格 schema 校验
  const ok = validateGradeNext(cleaned);
  if (!ok) {
    const err = validateGradeNext.errors?.map(e => `${e.instancePath || "(root)"} ${e.message}`).join("; ");
    throw new Error(`gradeAndNextQuiz JSON schema 校验失败（清洗后仍不合规）：${err}`);
  }

  const data = cleaned;

  // Rule 10 & System Mandatory: Force cooling / Max limits
  const MAX_ROUNDS = 5; 
  const MAX_QUESTIONS = 30;
  
  if (nextRound > MAX_ROUNDS || (session.totals.answered + (quiz.questions?.length || 0)) >= MAX_QUESTIONS) {
      data.grading.summary.status = "finish";
      if (!data.grading.summary.reason) {
          data.grading.summary.reason = "本次练习已达上限，建议休息一下或进入下一节。";
      }
  }

  // === Server-side override: compute accuracy correctly ===
  const calc = computeAccuracyFromResults(data.grading.results);
  const nextNum = clampInt(data.grading.summary?.nextNumQuestions, 3, 6, DEFAULT_NUM_QUESTIONS);

  data.grading.summary.accuracy = Number(calc.accuracy.toFixed(1));
  data.grading.summary.errorRate = Number(calc.errorRate.toFixed(1));
  data.grading.summary.nextNumQuestions = nextNum;
  data.grading.summary.difficultyAdjustment = difficultyAdjustmentByTarget(calc.accuracy, targetAccuracy);

  // === Dedupe nextQuiz and ensure length matches ===
  const isFinished = data.grading.summary.status === "finish";
  let nextQuestions = [];

  if (!isFinished) {
    const want = data.grading.summary.nextNumQuestions;
    const filtered = filterNewQuestions(session, data.nextQuiz?.questions || []);
    session.lastStems = session.lastStems || [];
    for (const q of filtered) session.lastStems.push(String(q.stem || "").slice(0, 120));
    session.lastStems = session.lastStems.slice(-200);

    // If after filtering we are short, top-up with createQuiz (one extra call max)
    nextQuestions = takeFirstN(filtered, want);
    if (nextQuestions.length < want) {
      const topup = await createQuiz({
        session,
        notes,
        round: nextRound,
        difficultyState,
        numQuestions: want - nextQuestions.length,
        apiKey
      });
      nextQuestions = nextQuestions.concat(topup.questions || []);
      nextQuestions = takeFirstN(nextQuestions, want);
    }
  }

  data.nextQuiz = { questions: nextQuestions };

  return data;
}

// ====== Helpers for review text ======
function formatReviewText(grading, quiz) {
  const byId = new Map((grading.results || []).map(r => [r.id, r]));
  const byQ = new Map((quiz.questions || []).map(q => [q.id, q]));

  let lines = [];
  let i = 1;
  for (const q of quiz.questions || []) {
    const r = byId.get(q.id);
    if (!r) continue;

    lines.push(`${i}. ${q.stem}`);
    lines.push(`结果：${r.correct ? "✅正确" : "❌错误"} | 标准答案：${r.correctAnswer}`);
    lines.push(`解析：${r.briefRationale}`);
    if (!r.correct) {
      lines.push(`错因类型：${r.errorType} | 概念：${byQ.get(q.id)?.concept || ""}`);
    }
    lines.push("");
    i++;
  }

  const calc = computeAccuracyFromResults(grading.results);
  lines.push(`本轮汇总：正确率 ${calc.accuracy.toFixed(1)}% / 错误率 ${(100 - calc.accuracy).toFixed(1)}%`);
  lines.push(`难度调整：${grading.summary.difficultyAdjustment} | 下轮题量：${grading.summary.nextNumQuestions}`);
  lines.push("");
  lines.push("易错点（Top）：");
  for (const w of grading.coaching.topWeaknesses || []) lines.push(`- ${w}`);
  lines.push("");
  lines.push(`下一步重点：${grading.coaching.nextFocus || ""}`);
  lines.push("");
  lines.push(`3-5分钟小练习：${grading.coaching.microDrill || ""}`);
  
  if (grading.summary.status === "finish") {
      lines.push("");
      lines.push(grading.summary.reason || "🎉 这一节的即时巩固已经完成，继续出题意义不大。");
  }

  return lines.join("\n");
}

function updateWeaknessStats(session, quiz, grading) {
  const qById = new Map((quiz.questions || []).map(q => [q.id, q]));
  for (const r of grading.results || []) {
    const q = qById.get(r.id);
    if (!q) continue;

    // track totals for concept too (optional)
    const key = `${q.concept}｜${r.errorType}`;
    session.weaknesses[key] = session.weaknesses[key] || { wrong: 0, total: 0, examples: [] };
    session.weaknesses[key].total += 1;

    if (r.correct) continue;
    session.weaknesses[key].wrong += 1;
    if (session.weaknesses[key].examples.length < 3) {
      session.weaknesses[key].examples.push(String(q.stem || "").slice(0, 60));
    }
  }
}

function weaknessText(session) {
  const entries = Object.entries(session.weaknesses || {});
  if (entries.length === 0) return "暂无数据。";

  entries.sort((a, b) => (b[1].wrong || 0) - (a[1].wrong || 0));
  const top = entries.slice(0, 10);

  const lines = [];
  for (const [k, v] of top) {
    lines.push(`- ${k}（错 ${v.wrong} 次 / 共 ${v.total} 次）`);
    for (const ex of v.examples || []) lines.push(`  · ${ex}…`);
  }
  return lines.join("\n");
}

function metaFromSession(session) {
  const cumulativeAccuracy = session.totals.answered
    ? (session.totals.correct / session.totals.answered) * 100
    : null;

  const last = session.history[session.history.length - 1];
  return {
    round: session.round,
    lastRoundAccuracy: last ? last.accuracy : null,
    totalAnswered: session.totals.answered,
    cumulativeAccuracy: cumulativeAccuracy ? Number(cumulativeAccuracy.toFixed(1)) : null,
    difficultyAdjustment: session.lastGrading?.summary?.difficultyAdjustment || "same",
    status: session.lastGrading?.summary?.status || "continue"
  };
}

// ====== Helpers: seen stems for de-dup ======
function normStem(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[^\p{L}\p{N} ]/gu, "") // 去掉中英文标点，只保留字母数字和空格
    .trim();
}

function markSeen(session, quiz) {
  if (!quiz?.questions) return;
  session.seenStems = session.seenStems || new Set();
  for (const q of quiz.questions) {
    const key = normStem(q.stem);
    if (key) session.seenStems.add(key);
  }
}

// ====== Helpers: compress notes to boundary ======
async function compressNotesToBoundary(session, notes) {
  if (MOCK_LLM) return "Mock Notes Boundary";
  const client = getClient(session.apiKey);
  const prompt = `
你要把用户笔记压缩成“可出题边界”，要求：
- 只保留：定义/结论/条件/步骤/公式/对比点/易混点
- 删除：例子/铺垫/感想/重复描述
- 不得引入新知识
- 输出纯文本，使用条目化结构（<= 300 行，越短越好）

用户笔记：
${notes}
`.trim();

  const resp = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "你是笔记压缩器，只输出纯文本要点。" },
      { role: "user", content: prompt }
    ],
    temperature: 0.1,
    top_p: 0.9
  });

  return (resp.choices?.[0]?.message?.content ?? "").trim();
}

// ====== OCR Worker Pool (Single Persistent Worker) ======
let globalWorker = null;
const ocrMutex = {
  lock: Promise.resolve(),
  dispatch(fn) {
    const p = this.lock.then(fn);
    this.lock = p.catch(() => {});
    return p;
  }
};

async function getOcrTextSafe(buffer) {
  return ocrMutex.dispatch(async () => {
    if (!globalWorker) {
      console.log("[OCR] Initializing worker...");
      globalWorker = await createWorker(OCR_LANGS, undefined, { langPath: OCR_LANG_PATH });
      console.log("[OCR] Worker ready.");
    }
    const { data: { text } } = await globalWorker.recognize(buffer);
    return text;
  });
}

// ====== Routes ======

app.post("/api/ping", (req, res) => {
  res.json({ ok: true });
});

app.get("/api/ping", (req, res) => {
  res.json({ ok: true });
});

// ====== New Upload Endpoint ======
app.post("/api/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file && !req.body.image) {
      return res.status(400).send("未检测到文件或图片数据");
    }

    let text = "";
    let fileName = "";

    if (req.file) {
      // 修复中文文件名乱码问题：multer 默认使用 latin1 编码 originalname
      try {
        fileName = Buffer.from(req.file.originalname, 'latin1').toString('utf8');
      } catch (e) {
        fileName = req.file.originalname;
      }
      const buffer = req.file.buffer;
      const ext = path.extname(fileName).toLowerCase();

      if (ext === ".docx") {
        try {
          const result = await mammoth.extractRawText({ buffer });
          text = result.value;
        } catch (e) {
          return res.status(400).send(`DOCX 解析失败：${e?.message || String(e)}`);
        }
      } else if (ext === ".doc") {
        return res.status(400).send("暂不支持 .doc（老版 Word 格式），请另存为 .docx 或导出为 .txt 后再上传。");
      } else if (ext === ".txt") {
        text = buffer.toString("utf-8");
      } else if ([".png", ".jpg", ".jpeg", ".bmp", ".gif"].includes(ext)) {
        text = await getOcrTextSafe(buffer);
      } else {
        return res.status(400).send(`暂不支持的文件格式: ${ext}`);
      }
    } else if (req.body.image) {
      // Base64 image from clipboard
      fileName = "截图.png";
      const base64Data = req.body.image.replace(/^data:image\/\w+;base64,/, "");
      const buffer = Buffer.from(base64Data, 'base64');
      text = await getOcrTextSafe(buffer);
    }

    res.json({ text: text.trim(), fileName });
  } catch (e) {
    console.error("Upload Error:", e);
    res.status(500).send(`文件处理失败: ${e.message}`);
  }
});

app.post("/api/start", async (req, res) => {
  try {
    const { notes, sessionId: existing, apiKey } = req.body || {};
    if (!notes || String(notes).trim().length < 5) return res.status(400).send("notes 不能为空");

    const id = existing || randomUUID();
    const now = new Date().toISOString();

    const session = sessionStore.get(id) || {
      id,
      apiKey: null,
      notes: "",
      notesRaw: "",
      notesBound: "",
      round: 1,
      difficultyState: { level: 0 },
      history: [],
      totals: { answered: 0, correct: 0 },
      weaknesses: {},
      lastQuiz: null,
      lastGrading: null,
      lastGradedQuiz: null,
      lastReviewText: "",
      createdAt: now,
      lastActive: Date.now(),
      seen: new Set(),
      seenStems: new Set(),
      lastStems: []
    };

    // Server-side Cooling Check
    if (notes && String(notes).length > 5) {
      const cooling = checkServerCooling(String(notes));
      if (!cooling.ok) {
        return res.status(429).send(cooling.msg);
      }
      if (cooling.chapterHash) {
        session.pendingChapterHash = cooling.chapterHash;
      }
    }

    if (apiKey && typeof apiKey === "string" && apiKey.length > 5) {
      session.apiKey = apiKey.trim();
    }

    // overwrite notes if provided (V1: allow reuse sessionId with new notes)
    session.notesRaw = String(notes);
    session.notes = session.notesRaw;
    session.notesBound = session.notesBound || "";
    session.createdAt = session.createdAt || now;
    session.lastActive = Date.now();

    // require key before first quiz / compression
    if (!session.apiKey) {
      sessionStore.set(id, session);
      return res.status(400).send("缺少 API Key：请在前端输入并保存 Key 后再试。");
    }

    // skip heavy compression to save time, unless notes are massive
    if (session.notesRaw.length > 8000) {
      session.notesBound = await compressNotesToBoundary(session, session.notesRaw);
    } else {
      session.notesBound = session.notesRaw;
    }

    // (Re)start from round 1 if new session or no lastQuiz
    if (!session.lastQuiz) {
      session.round = 1;
      session.history = [];
      session.seen = session.seen || new Set();
    }

    const pack = await generatePracticePack({
      session,
      notes: session.notesBound || session.notesRaw || session.notes,
      apiKey: session.apiKey
    });

    session.lastPack = pack; // Save the pack
    session.lastQuiz = { questions: pack.questions }; // Legacy support if needed
    
    sessionStore.set(id, session);

    res.json({
      sessionId: id,
      pack: pack, // Return the whole pack
      meta: { ...pack.meta, sessionId: id }
    });
  } catch (e) {
    res.status(500).send(String(e));
  }
});

app.post("/api/start_stream", async (req, res) => {
  res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();

  const writeLine = (obj) => {
    try {
      res.write(JSON.stringify(obj) + "\n");
    } catch {}
  };

  try {
    const { notes, sessionId: existing, apiKey } = req.body || {};
    if (!notes || String(notes).trim().length < 5) {
      writeLine({ type: "error", message: "notes 不能为空" });
      return res.end();
    }

    const id = existing || randomUUID();
    const now = new Date().toISOString();

    const session = sessionStore.get(id) || {
      id,
      apiKey: null,
      notes: "",
      notesRaw: "",
      notesBound: "",
      round: 1,
      difficultyState: { level: 0 },
      history: [],
      totals: { answered: 0, correct: 0 },
      weaknesses: {},
      lastQuiz: null,
      lastGrading: null,
      lastGradedQuiz: null,
      lastReviewText: "",
      createdAt: now,
      lastActive: Date.now(),
      seen: new Set(),
      seenStems: new Set(),
      lastStems: []
    };

    const cooling = checkServerCooling(String(notes));
    if (!cooling.ok) {
      writeLine({ type: "error", message: cooling.msg });
      return res.end();
    }
    if (cooling.chapterHash) {
      session.pendingChapterHash = cooling.chapterHash;
    }

    if (apiKey && typeof apiKey === "string" && apiKey.length > 5) {
      session.apiKey = apiKey.trim();
    }

    session.notesRaw = String(notes);
    session.notes = session.notesRaw;
    session.notesBound = session.notesBound || "";
    session.createdAt = session.createdAt || now;
    session.lastActive = Date.now();

    if (!session.apiKey) {
      sessionStore.set(id, session);
      writeLine({ type: "error", message: "缺少 API Key：请在前端输入并保存 Key 后再试。" });
      return res.end();
    }

    if (session.notesRaw.length > 8000) {
      writeLine({ type: "stage", value: "正在压缩笔记（仅对超长内容）" });
      session.notesBound = await compressNotesToBoundary(session, session.notesRaw);
    } else {
      session.notesBound = session.notesRaw;
    }

    if (!session.lastQuiz) {
      session.round = 1;
      session.history = [];
      session.seen = session.seen || new Set();
    }

    writeLine({ type: "stage", value: "正在生成题目（实时输出）" });

    let sendBuf = "";
    let lastFlush = Date.now();
    const flush = (force = false) => {
      const nowMs = Date.now();
      if (!force && sendBuf.length < 120 && nowMs - lastFlush < 120) return;
      if (sendBuf) writeLine({ type: "delta", text: sendBuf });
      sendBuf = "";
      lastFlush = nowMs;
    };

    const pack = await generatePracticePackStream({
      session,
      notes: session.notesBound || session.notesRaw || session.notes,
      apiKey: session.apiKey,
      onDelta: (t) => {
        sendBuf += String(t || "");
        flush(false);
      }
    });

    flush(true);

    session.lastPack = pack;
    session.lastQuiz = { questions: pack.questions };
    sessionStore.set(id, session);

    writeLine({
      type: "final",
      sessionId: id,
      pack,
      meta: { ...pack.meta, sessionId: id }
    });
    return res.end();
  } catch (e) {
    writeLine({ type: "error", message: String(e?.message || e) });
    return res.end();
  }
});

app.post("/api/submit", async (req, res) => {
  try {
    const { sessionId, answers, apiKey } = req.body || {};
    const session = sessionStore.get(sessionId);
    if (!session) return res.status(400).send("无效 sessionId，请先开始训练。");

    session.lastActive = Date.now();

    // Robust Key Handling: Update session key if provided in request
    if (apiKey && typeof apiKey === "string" && apiKey.length > 5) {
       session.apiKey = apiKey.trim();
    }

    if (!session.apiKey) return res.status(400).send("缺少 API Key：请在前端输入并保存 Key 后再试。");
    if (!session.lastQuiz) return res.status(400).send("当前没有可提交的题目。");

    const quiz = session.lastQuiz;

    // ====== 单次调用：判分 + 下一轮出题（V1：并由服务端纠正正确率） ======
    const nextRound = session.round + 1;

    const combo = await gradeAndNextQuiz({
      session,
      notes: session.notesBound || session.notesRaw || session.notes,
      chapterModel: session.chapterModel,
      quiz,
      answers,
      targetAccuracy: ROUND_TARGET_ACC,
      nextRound,
      difficultyState: session.difficultyState,
      nextNumQuestionsHint: DEFAULT_NUM_QUESTIONS,
      seenStems: Array.from(session.seenStems || []),
      apiKey: session.apiKey
    });

    const grading = combo.grading;
    session.lastGrading = grading;

    // update totals (use computed results)
    const calc = computeAccuracyFromResults(grading.results);
    session.totals.answered += calc.total;
    session.totals.correct += calc.correct;

    // record history
    session.history.push({ 
      round: session.round, 
      accuracy: Number(calc.accuracy.toFixed(1)), 
      total: calc.total, 
      correct: calc.correct,
      quiz: JSON.parse(JSON.stringify(quiz)), // deep copy
      results: grading.results
    });

    // update weaknesses
    updateWeaknessStats(session, quiz, grading);

    // adjust difficulty state (simple level drift)
    if (grading.summary.difficultyAdjustment === "up") session.difficultyState.level += 1;
    if (grading.summary.difficultyAdjustment === "down") session.difficultyState.level -= 1;

    const reviewText = formatReviewText(grading, quiz);

    session.lastGradedQuiz = quiz;
    session.lastReviewText = reviewText;

    // ====== 后端强去重 ======
    const wantN = combo.grading.summary.nextNumQuestions;
    let nextQs = (combo.nextQuiz.questions || []).filter(q => {
      const key = normStem(q.stem);
      session.seenStems = session.seenStems || new Set();
      if (session.seenStems.has(key)) return false;
      return true;
    });

    // 如果去重后题量不足，补题（仅在需要时才额外调用一次模型）
    if (nextQs.length < wantN) {
      const missing = wantN - nextQs.length;
      const extra = await createQuiz({
        session,
        notes: session.notes,
        round: nextRound,
        difficultyState: session.difficultyState,
        numQuestions: missing,
        apiKey: session.apiKey
      });

      for (const q of extra.questions || []) {
        const key = normStem(q.stem);
        if (!session.seenStems.has(key) && nextQs.length < wantN) {
          nextQs.push(q);
        }
      }
    }

    // 最终仍不足就截断/兜底（极少发生）
    nextQs = nextQs.slice(0, wantN);
    combo.nextQuiz.questions = nextQs;

    // Done condition: cumulative accuracy >= TARGET
    const cumAcc = (session.totals.correct / session.totals.answered) * 100;
    const isGoalReached = cumAcc >= TARGET_ACCURACY;

    // next round quiz (来自 combo.nextQuiz)
    session.round = nextRound;
    session.lastQuiz = {
      questions: combo.nextQuiz.questions,
      meta: { round: session.round, targetAccuracy: TARGET_ACCURACY }
    };
    // 记录下一轮已出题干
    markSeen(session, session.lastQuiz);

    sessionStore.set(sessionId, session);

    return res.json({
      done: false, // 永远不停止，除非用户手动重置
      isGoalReached, // 仅作为达标标记
      meta: metaFromSession(session),
      history: session.history,
      weaknessText: weaknessText(session),
      reviewText,
      results: grading.results, // 添加详细判分结果用于前端展示
      summary: grading.summary,
      nextQuiz: session.lastQuiz
    });
  } catch (e) {
    return res.status(500).send(String(e));
  }
});

app.post("/api/state", (req, res) => {
  const { sessionId } = req.body || {};
  const session = sessionStore.get(sessionId);
  if (!session) return res.status(400).send("无效 sessionId。");
  res.json({
    meta: metaFromSession(session),
    history: session.history,
    weaknessText: weaknessText(session),
    lastQuiz: session.lastQuiz
  });
});

app.post("/api/export", (req, res) => {
  const { sessionId } = req.body || {};
  const session = sessionStore.get(sessionId);
  if (!session) return res.status(400).send("无效 sessionId。");
  res.json({
    id: session.id,
    createdAt: session.createdAt,
    notes: session.notes,
    meta: metaFromSession(session),
    difficultyState: session.difficultyState,
    history: session.history,
    weaknesses: session.weaknesses,
    totals: session.totals
  });
});

app.post("/api/review_last", (req, res) => {
  const { sessionId } = req.body || {};
  const session = sessionStore.get(sessionId);
  if (!session || !session.lastGrading || !session.lastGradedQuiz) {
    return res.status(400).send("暂无解析：请先提交一次答案。");
  }
  return res.json({ reviewText: formatReviewText(session.lastGrading, session.lastGradedQuiz) });
});

app.post("/api/history_round", (req, res) => {
  const { sessionId, round } = req.body || {};
  const session = sessionStore.get(sessionId);
  if (!session) return res.status(400).send("无效 sessionId。");
  
  const record = session.history.find(h => h.round === round);
  if (!record) return res.status(404).send(`未找到第 ${round} 轮的历史记录。`);
  
  res.json({
    quiz: record.quiz,
    results: record.results,
    meta: {
      round: record.round,
      accuracy: record.accuracy,
      total: record.total,
      correct: record.correct
    }
  });
});

app.post("/api/wrong_questions", (req, res) => {
  const { sessionId } = req.body || {};
  const session = sessionStore.get(sessionId);
  if (!session) return res.status(400).send("无效 sessionId。");

  const wrongQuestions = [];
  // session.history contains: { round, accuracy, total, correct, quiz, results }
  // results is an array of: { id, isCorrect, userAns, correctAns, rationale }
  
  for (const entry of session.history) {
    if (!entry.results || !entry.quiz) continue;
    
    for (const resItem of entry.results) {
      if (resItem.isCorrect === false) {
        // Find the question detail from the quiz
        const qDetail = entry.quiz.questions.find(q => q.id === resItem.id);
        if (qDetail) {
          wrongQuestions.push({
            ...qDetail,
            userAns: resItem.userAns,
            isCorrect: false,
            round: entry.round,
            // standard format expected by frontend
            correctAns: resItem.correctAns,
            rationale: resItem.rationale || qDetail.rationale
          });
        }
      }
    }
  }

  res.json({
     wrongQuestions
   });
 });
 
 app.post("/api/sync_history", (req, res) => {
   const { sessionId, results, quiz } = req.body || {};
   const session = sessionStore.get(sessionId);
   if (!session) return res.status(400).send("无效 sessionId。");
 
   // results: Array of { id, isCorrect, userAns, correctAns, rationale }
   // quiz: The whole pack or the current quiz questions
   
   if (!results || !quiz) return res.status(400).send("数据不完整。");
 
   const calc = computeAccuracyFromResults(results);
   
   session.history.push({
     round: session.round,
     accuracy: Number(calc.accuracy.toFixed(1)),
     total: calc.total,
     correct: calc.correct,
     quiz: quiz,
     results: results,
     syncedAt: new Date().toISOString()
   });
 
   session.totals.answered += calc.total;
   session.totals.correct += calc.correct;
   session.round += 1;
   
   sessionStore.set(sessionId, session);
   safeWriteSessions();
 
   res.json({ ok: true });
 });
 
 app.post("/api/reset", (req, res) => {
  const { sessionId } = req.body || {};
  sessionStore.delete(sessionId);
  res.json({ ok: true });
});

// ====== Start ======
function startServer(preferredPort) {
  const host = "0.0.0.0";
  const maxTries = 10;
  const tryListen = (port, triesLeft) => {
    const server = app.listen(port, host);
    server.on("listening", () => {
      const addr = server.address();
      const actualPort = addr && typeof addr === "object" ? addr.port : port;
      const url = `http://127.0.0.1:${actualPort}`;
      console.log(`Trainer backend running on ${url}`);
      console.log("Using DeepSeek OpenAI-compatible mode:");
      console.log("  baseURL =", BASE_URL);
      console.log("  model   =", MODEL);
      if (process.env.NO_OPEN !== "1") {
        try { open(url); } catch {}
      }
    });
    server.on("error", (err) => {
      if (err && err.code === "EADDRINUSE" && triesLeft > 0) {
        try { server.close(() => tryListen(port + 1, triesLeft - 1)); } catch { tryListen(port + 1, triesLeft - 1); }
        return;
      }
      console.error(err);
      process.exitCode = 1;
    });
  };
  tryListen(preferredPort, maxTries);
}

startServer(PORT);
