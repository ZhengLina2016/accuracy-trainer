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
const DEFAULT_NUM_QUESTIONS = 8;
const MODEL = process.env.MODEL_NAME || "deepseek-chat";
const PORT = Number(process.env.PORT || 8787);
const OCR_LANGS = "chi_sim+eng";
const OCR_LANG_PATH = process.cwd();

// DeepSeek OpenAI-compatible endpoint
const BASE_URL = "https://api.deepseek.com/v1";

// ====== Static frontend (optional) ======
// Put your index.html/app.js/styles.css under ./public
const PUBLIC_DIR = path.join(process.cwd(), "public");
if (fs.existsSync(PUBLIC_DIR)) {
  app.use(express.static(PUBLIC_DIR));
  app.get("/", (_, res) => res.sendFile(path.join(PUBLIC_DIR, "index.html")));
}

// ====== In-memory sessions + persistence ======
const sessions = new Map();
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

function safeWriteSessions() {
  try {
    const arr = Array.from(sessions.values()).map(s => ({
      ...s,
      // serialize Set
      seen: Array.from(s.seen || []),
      seenStems: Array.from(s.seenStems || []),
    }));
    fs.writeFileSync(SESS_FILE, JSON.stringify(arr, null, 2), "utf-8");
  } catch (e) {
    console.warn("[persist] save failed:", String(e));
  }
}

// Load persisted sessions on startup
safeReadSessions();

// Save periodically (and on changes in key routes too)
setInterval(safeWriteSessions, 15_000).unref();

// ====== JSON Schemas ======
function quizSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      questions: {
        type: "array",
        minItems: 8,
        maxItems: 20,
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            id: { type: "string" },
            type: { type: "string", enum: ["single", "multi", "tf", "short"] },
            stem: { type: "string" },
            options: { type: ["array", "null"], items: { type: "string" } },
            concept: { type: "string" },
            answer: { type: "string" },
            rationale: { type: "string" }
          },
          required: ["id", "type", "stem", "options", "concept", "answer", "rationale"]
        }
      }
    },
    required: ["questions"]
  };
}

function gradingSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      results: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            id: { type: "string" },
            correct: { type: "boolean" },
            correctAnswer: { type: "string" },
            briefRationale: { type: "string" },
            errorType: { type: "string" }
          },
          required: ["id", "correct", "correctAnswer", "briefRationale", "errorType"]
        }
      },
      summary: {
        type: "object",
        additionalProperties: false,
        properties: {
          accuracy: { type: "number" },
          errorRate: { type: "number" },
          difficultyAdjustment: { type: "string", enum: ["up", "same", "down"] },
          nextNumQuestions: { type: "integer", minimum: 8, maximum: 20 }
        },
        required: ["accuracy", "errorRate", "difficultyAdjustment", "nextNumQuestions"]
      },
      coaching: {
        type: "object",
        additionalProperties: false,
        properties: {
          topWeaknesses: { type: "array", items: { type: "string" }, maxItems: 8 },
          nextFocus: { type: "string" },
          microDrill: { type: "string" }
        },
        required: ["topWeaknesses", "nextFocus", "microDrill"]
      }
    },
    required: ["results", "summary", "coaching"]
  };
}

function gradeNextSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      grading: gradingSchema(),
      nextQuiz: quizSchema()
    },
    required: ["grading", "nextQuiz"]
  };
}

// ====== AJV Validators ======
const ajv = new Ajv({ allErrors: true, strict: false });
const validateQuiz = ajv.compile(quizSchema());
const validateGrading = ajv.compile(gradingSchema());
const validateGradeNext = ajv.compile(gradeNextSchema());

// ====== Prompt ======
function makeSystemPrompt() {
  return [
    "你是“自适应难度学习测评专家”，目标是让用户保持在“心流”状态。",
    "必须严格遵守用户给定的‘已学内容边界’，不得引入新概念/新术语/新结论。",
    "出题策略：根据用户当前的正确率动态调整难度。",
    "- 如果正确率高于85%，增加题目深度、增加多选题比例、引入更细微的易错点。",
    "- 如果正确率低于85%，回归基础定义、增加单选题和判断题比例、提供更清晰的提示。",
    "理想目标是让用户每一轮的正确率都维持在 85% 左右，从而实现最高效的长期记忆。",
    "题目构成比例：60%稳对、25%思考、15%边界易错（不得在题干暴露分类）。",
    "题型允许：single(单选)/multi(多选)/tf(判断)/short(简答)。至少50%可客观判分。",
    "关键输出约束：你必须只输出一个 JSON 对象，不得输出任何解释文字、前后缀、markdown、代码块。"
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

// ====== chatJson with validator ======
async function chatJson({ session, messages, validator, what, apiKey }) {
  const client = getClient(apiKey);

  let resp;
  try {
    resp = await client.chat.completions.create({
      model: MODEL,
      messages,
      temperature: 0.1,
      top_p: 0.9
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

// ====== LLM Calls ======
async function createQuiz({ session, notes, round, difficultyState, numQuestions, apiKey }) {
  // Provide avoid list to reduce repeats (only short excerpts to keep tokens low)
  const avoid = Array.from(session.seen || []).slice(-50); // hashes (not useful to model)
  const avoidStems = (session.lastStems || []).slice(-30); // actual stems for model
  const avoidText = avoidStems.length
    ? avoidStems.map((s, i) => `${i + 1}) ${s}`).join("\n")
    : "无";

  const prompt = `
已学内容边界（只允许在此范围内出题）：
${notes}

当前轮次：${round}
目标正确率（每轮期望）：${ROUND_TARGET_ACC}%
当前难度状态（仅供自适应）：${JSON.stringify(difficultyState)}
本轮题量：${numQuestions}

强约束：严禁出与“已出题目列表”重复或高度近似（同一问法/同一选项组合/同一判断陈述）。
已出题目列表（只用于去重参考）：
${avoidText}

要求：
- 整体难度控制在使用户正确率约 ${ROUND_TARGET_ACC}%（即错约 ${100 - ROUND_TARGET_ACC}%）附近
- 题目比例：约60%稳对、25%思考、15%边界易错（不在题干暴露）
- 每题给一个 concept（简短），用于统计易错点
- 必须为每道题生成标准答案（answer）和极简解析（rationale，<=2句）
- **标准答案格式说明**：
  - single/multi/tf 类型：必须使用选项对应的字母（如 "A", "AB", "ACD", "T", "F"），严禁使用选项文本全称。
  - short 类型：使用简短的要点。
- 只输出一个 JSON 对象：
  {"questions":[{"id":"...","type":"single|multi|tf|short","stem":"...","options":[...或null],"concept":"...","answer":"...","rationale":"..."}]}
`.trim();

  // We may retry once if many duplicates are filtered out
  const maxTry = 2;
  let collected = [];
  for (let t = 0; t < maxTry && collected.length < numQuestions; t++) {
    const raw = await chatJson({
      session,
      messages: [
        { role: "system", content: makeSystemPrompt() },
        { role: "user", content: prompt }
      ],
      validator: null,                 // ✅ 先不校验
      what: "createQuiz_raw",
      apiKey
    });

    // filter duplicates
    const fresh = filterNewQuestions(session, raw.questions || []);
    const normalizedFresh = normalizeQuestions(fresh);

    // save stems for future avoid list
    session.lastStems = session.lastStems || [];
    for (const q of normalizedFresh) session.lastStems.push(String(q.stem || "").slice(0, 120));
    session.lastStems = session.lastStems.slice(-200);

    collected = collected.concat(normalizedFresh);
  }

  // if still insufficient (rare), allow remaining from last batch to avoid hard crash
  collected = takeFirstN(collected, numQuestions);
  if (collected.length < numQuestions) {
    // As a last resort, do not block user; just proceed
    console.warn(`[dedupe] only got ${collected.length}/${numQuestions} unique questions`);
  }

  return { questions: collected };
}

function normalizeQuestions(questions) {
  if (!Array.isArray(questions)) return [];
  return questions.map((q, idx) => {
    if (!q || typeof q !== "object") return q;
    
    // 显式提取必需字段，确保没有额外属性干扰 Schema 校验
    const out = {
      id: q.id != null ? String(q.id) : `q_${idx}`,
      type: ["single", "multi", "tf", "short"].includes(q.type) ? q.type : "single",
      stem: q.stem != null ? String(q.stem) : "无题干",
      options: null,
      concept: q.concept != null ? String(q.concept) : "通用",
      answer: q.answer != null ? String(q.answer) : "",
      rationale: q.rationale != null ? String(q.rationale) : (q.explanation != null ? String(q.explanation) : (q.briefRationale != null ? String(q.briefRationale) : "无解析"))
    };
    
    // 仅在单选/多选时保留 options 数组
    if (out.type === "single" || out.type === "multi") {
      out.options = Array.isArray(q.options) ? q.options.map(String) : [];
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

async function gradeAndNextQuiz({ session, notes, quiz, answers, targetAccuracy, nextRound, difficultyState, nextNumQuestionsHint, seenStems, apiKey }) {
  const prompt = `
已学内容边界（严格限制：不得引入任何新概念/新术语/新结论）：
${notes}

【当前轮题目 JSON】：
${JSON.stringify(quiz)}

【用户作答 JSON】：
${JSON.stringify(answers)}

你需要一次性输出一个 JSON 对象，包含两部分：
1) grading：对当前轮判分（结构必须满足 gradingSchema）
2) nextQuiz：生成下一轮题目（结构必须满足 quizSchema）

判分要求（grading）：
- results：逐题对/错、标准答案、极简解析（<=2句）
- **重要**：对于 single/multi/tf 类型，correctAnswer 必须是选项字母（如 "A", "ACD", "T", "F"），严禁使用选项内容全称。
- **重要**：briefRationale 必须包含对正确答案的解释或对错误原因的分析，严禁返回空字符串或无意义内容。
- 错题标注 errorType（从以下挑最贴近的，或用同级别短语）：
  1) 概念混淆 2) 条件遗漏 3) 定义不清 4) 推理跳步 5) 审题偏差 6) 表达不精确 7) 计算/符号失误 8) 记忆缺口
- summary：accuracy、errorRate、difficultyAdjustment、nextNumQuestions（8-20）
  nextNumQuestions 优先参考 nextNumQuestionsHint=${nextNumQuestionsHint ?? "null"}（建议 8-12 更快）
- coaching：topWeaknesses / nextFocus / microDrill 按原 schema 输出

【已出过题的题干（禁止重复或近似改写）】：
${(seenStems || []).slice(-120).join("\n")}

出题要求（nextQuiz）：
- 下一轮轮次：${nextRound}
- 目标正确率（每轮期望）：${targetAccuracy}%
- 当前难度状态（仅供自适应）：${JSON.stringify(difficultyState)}
- 题量：必须等于 grading.summary.nextNumQuestions
- nextQuiz 的每道题 stem 必须与上述已出题干明显不同（不得同义改写/换序/只改数字）
- 严禁出与本会话历史题目重复或高度近似（同一问法/同一选项组合/同一判断陈述）
- 题目比例：约60%稳对、25%思考、15%边界易错（不在题干暴露）
- 每题给一个 concept（简短），用于统计易错点
- 必须为每道题生成标准答案（answer）和极简解析（rationale，<=2句）
- **标准答案格式说明**：必须使用选项字母（如 "A", "ACD"），严禁使用文本全称。
- 只在已学边界内出题
- 必须严格遵守 quizSchema，包含 answer 和 rationale 字段

输出格式必须严格是：
{
  "grading": { ...gradingSchema... },
  "nextQuiz": { ...quizSchema... }
}
注意：你必须只输出这一份 JSON，不得输出任何解释文字、markdown、代码块。
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
      nextNumQuestions: clampInt(s.nextNumQuestions, 8, 20, DEFAULT_NUM_QUESTIONS)
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

  // === Server-side override: compute accuracy correctly ===
  const calc = computeAccuracyFromResults(data.grading.results);
  const nextNum = clampInt(data.grading.summary?.nextNumQuestions, 8, 20, DEFAULT_NUM_QUESTIONS);

  data.grading.summary.accuracy = Number(calc.accuracy.toFixed(1));
  data.grading.summary.errorRate = Number(calc.errorRate.toFixed(1));
  data.grading.summary.nextNumQuestions = nextNum;
  data.grading.summary.difficultyAdjustment = difficultyAdjustmentByTarget(calc.accuracy, targetAccuracy);

  // === Dedupe nextQuiz and ensure length matches ===
  const want = data.grading.summary.nextNumQuestions;
  const filtered = filterNewQuestions(session, data.nextQuiz?.questions || []);
  session.lastStems = session.lastStems || [];
  for (const q of filtered) session.lastStems.push(String(q.stem || "").slice(0, 120));
  session.lastStems = session.lastStems.slice(-200);

  // If after filtering we are short, top-up with createQuiz (one extra call max)
  let nextQuestions = takeFirstN(filtered, want);
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
    difficultyAdjustment: session.lastGrading?.summary?.difficultyAdjustment || "same"
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
        const worker = await createWorker(OCR_LANGS, undefined, { langPath: OCR_LANG_PATH });
        try {
          const { data: { text: ocrText } } = await worker.recognize(buffer);
          text = ocrText;
        } finally {
          await worker.terminate();
        }
      } else {
        return res.status(400).send(`暂不支持的文件格式: ${ext}`);
      }
    } else if (req.body.image) {
      // Base64 image from clipboard
      fileName = "截图.png";
      const base64Data = req.body.image.replace(/^data:image\/\w+;base64,/, "");
      const buffer = Buffer.from(base64Data, 'base64');
      const worker = await createWorker(OCR_LANGS, undefined, { langPath: OCR_LANG_PATH });
      try {
        const { data: { text: ocrText } } = await worker.recognize(buffer);
        text = ocrText;
      } finally {
        await worker.terminate();
      }
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

    const session = sessions.get(id) || {
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
      seen: new Set(),
      seenStems: new Set(),
      lastStems: []
    };

    if (apiKey && typeof apiKey === "string" && apiKey.length > 5) {
      session.apiKey = apiKey.trim();
    }

    // overwrite notes if provided (V1: allow reuse sessionId with new notes)
    session.notesRaw = String(notes);
    session.notes = session.notesRaw;
    session.notesBound = session.notesBound || "";
    session.createdAt = session.createdAt || now;

    // require key before first quiz / compression
    if (!session.apiKey) {
      sessions.set(id, session);
      safeWriteSessions();
      return res.status(400).send("缺少 API Key：请在前端输入并保存 Key 后再试。");
    }

    // compress notes to boundary once per (re)start
    session.notesBound = await compressNotesToBoundary(session, session.notesRaw);

    // (Re)start from round 1 if new session or no lastQuiz
    if (!session.lastQuiz) {
      session.round = 1;
      session.difficultyState = { level: 0 };
      session.history = [];
      session.totals = { answered: 0, correct: 0 };
      session.weaknesses = {};
      session.lastGrading = null;
      session.lastGradedQuiz = null;
      session.lastReviewText = "";
      session.seen = session.seen || new Set();
      session.seenStems = session.seenStems || new Set();
      session.lastStems = session.lastStems || [];
    }

    const quiz = await createQuiz({
      session,
      notes: session.notesBound || session.notesRaw || session.notes,
      round: session.round,
      difficultyState: session.difficultyState,
      numQuestions: DEFAULT_NUM_QUESTIONS,
      apiKey: session.apiKey
    });

    session.lastQuiz = { questions: quiz.questions, meta: { round: session.round, targetAccuracy: TARGET_ACCURACY } };
    // 记录已出题干
    markSeen(session, session.lastQuiz);
    sessions.set(id, session);
    safeWriteSessions();

    res.json({
      sessionId: id,
      quiz: session.lastQuiz,
      meta: metaFromSession(session),
      history: session.history,
      weaknessText: weaknessText(session)
    });
  } catch (e) {
    res.status(500).send(String(e));
  }
});

app.post("/api/submit", async (req, res) => {
  try {
    const { sessionId, answers, apiKey } = req.body || {};
    const session = sessions.get(sessionId);
    if (!session) return res.status(400).send("无效 sessionId，请先开始训练。");

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
      quiz,
      answers,
      targetAccuracy: ROUND_TARGET_ACC,
      nextRound,
      difficultyState: session.difficultyState,
      nextNumQuestionsHint: 8,
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

    sessions.set(sessionId, session);
    safeWriteSessions();

    return res.json({
      done: false, // 永远不停止，除非用户手动重置
      isGoalReached, // 仅作为达标标记
      meta: metaFromSession(session),
      history: session.history,
      weaknessText: weaknessText(session),
      reviewText,
      results: grading.results, // 添加详细判分结果用于前端展示
      nextQuiz: session.lastQuiz
    });
  } catch (e) {
    return res.status(500).send(String(e));
  }
});

app.post("/api/state", (req, res) => {
  const { sessionId } = req.body || {};
  const session = sessions.get(sessionId);
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
  const session = sessions.get(sessionId);
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
  const session = sessions.get(sessionId);
  if (!session || !session.lastGrading || !session.lastGradedQuiz) {
    return res.status(400).send("暂无解析：请先提交一次答案。");
  }
  return res.json({ reviewText: formatReviewText(session.lastGrading, session.lastGradedQuiz) });
});

app.post("/api/history_round", (req, res) => {
  const { sessionId, round } = req.body || {};
  const session = sessions.get(sessionId);
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

app.post("/api/reset", (req, res) => {
  const { sessionId } = req.body || {};
  sessions.delete(sessionId);
  safeWriteSessions();
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
