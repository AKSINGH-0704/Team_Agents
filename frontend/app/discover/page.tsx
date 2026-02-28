"use client";
import { useState, useRef, useEffect } from "react";
import { ArrowLeft, Send, Loader2, Shield, BarChart2, History, Plus, Trash2, X, BookOpen, AlertTriangle } from "lucide-react";
import Link from "next/link";
import {
  createChatSession,
  listChatSessions,
  getChatSession,
  sendChatMessage,
  deleteChatSession,
  discoverChat,
  comparePolicies,
} from "@/lib/api";
import PolicyCard from "@/components/PolicyCard";
import ComparisonTable from "@/components/ComparisonTable";

const SESSION_KEY = "policyai_session_id";

interface Message {
  role: "user" | "assistant";
  content: string;
  type?: "question" | "results" | "no_results" | "explanation" | "chat";
  policies?: any[];
  extracted?: any;
  // explanation fields
  example?: string;
  citation?: string;
  policy_name?: string;
  found?: boolean;
}

interface Session {
  id: string;
  session_name?: string;
  created_at: string;
  updated_at: string;
}

const STARTER_PROMPTS = [
  "I need maternity coverage, I have diabetes, budget ₹18,000/year, family of 3",
  "Best family floater plan under ₹12,000/year with OPD coverage",
  "Senior citizen plan with no room rent limit and mental health coverage",
  "I'm 28, healthy, want a basic individual plan under ₹8,000/year",
];

const GREETING: Message = {
  role: "assistant",
  content:
    "Hi! I'm your AI health insurance advisor. Tell me about your health needs, budget, and family size — I'll find and rank the best policies for you.\n\nFor example: \"I need maternity coverage, have diabetes, budget ₹18k/year, family of 3\"",
  type: "question",
};

export default function DiscoverPage() {
  const [messages, setMessages] = useState<Message[]>([GREETING]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionLoading, setSessionLoading] = useState(true);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [sessionPolicyIds, setSessionPolicyIds] = useState<string[]>([]);
  const [selected, setSelected] = useState<{ id: string; name: string }[]>([]);
  const [comparing, setComparing] = useState(false);
  const [compareResult, setCompareResult] = useState<any>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const historyRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Close history dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (historyRef.current && !historyRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  // On mount: restore or create session
  useEffect(() => {
    async function initSession() {
      setSessionLoading(true);
      try {
        const stored = localStorage.getItem(SESSION_KEY);
        if (stored) {
          try {
            const data = await getChatSession(stored);
            if (data?.messages?.length > 0) {
              // Reconstruct messages from DB
              const restored: Message[] = data.messages.map((m: any) => ({
                role: m.role as "user" | "assistant",
                content: m.content,
                type: m.metadata?.type,
                policies: m.metadata?.policies,
                extracted: m.metadata?.extracted_requirements,
              }));
              setMessages(restored);
            }
            setSessionId(stored);
            setSessionLoading(false);
            return;
          } catch {
            // Session expired or deleted — create new one
            localStorage.removeItem(SESSION_KEY);
          }
        }
        // Create new session
        const s = await createChatSession();
        localStorage.setItem(SESSION_KEY, s.session_id);
        setSessionId(s.session_id);
      } catch {
        // Backend unavailable — operate stateless
      } finally {
        setSessionLoading(false);
      }
    }
    initSession();
  }, []);

  async function loadSessionHistory() {
    try {
      const data = await listChatSessions();
      setSessions(data.sessions || []);
    } catch {
      setSessions([]);
    }
  }

  async function switchSession(sid: string) {
    setShowHistory(false);
    setLoading(false);
    try {
      const data = await getChatSession(sid);
      if (data?.messages?.length > 0) {
        const restored: Message[] = data.messages.map((m: any) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
          type: m.metadata?.type,
          policies: m.metadata?.policies,
          extracted: m.metadata?.extracted_requirements,
        }));
        setMessages(restored);
      } else {
        setMessages([GREETING]);
      }
      setSessionId(sid);
      localStorage.setItem(SESSION_KEY, sid);
      setSelected([]);
      setCompareResult(null);
    } catch {
      // ignore
    }
  }

  async function startNewSession() {
    setShowHistory(false);
    try {
      const s = await createChatSession();
      localStorage.setItem(SESSION_KEY, s.session_id);
      setSessionId(s.session_id);
      setMessages([GREETING]);
      setSelected([]);
      setCompareResult(null);
    } catch {
      setMessages([GREETING]);
    }
  }

  async function handleDeleteSession(sid: string, e: React.MouseEvent) {
    e.stopPropagation();
    try {
      await deleteChatSession(sid);
      setSessions((prev) => prev.filter((s) => s.id !== sid));
      if (sid === sessionId) {
        await startNewSession();
      }
    } catch {
      // ignore
    }
  }

  async function sendMessage(text?: string) {
    const userText = (text ?? input).trim();
    if (!userText || loading) return;
    setInput("");
    const userMsg: Message = { role: "user", content: userText };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      let data: any;
      if (sessionId) {
        // Persistent path — backend stores messages in DB
        data = await sendChatMessage(sessionId, userText);
      } else {
        // Stateless fallback (session creation failed)
        const apiMessages = [...messages, userMsg].map((m) => ({
          role: m.role,
          content: m.content,
        }));
        data = await discoverChat(apiMessages, sessionPolicyIds);
      }

      // Track uploaded policy IDs from results for future term lookups
      if (data.uploaded_policy_ids?.length) {
        setSessionPolicyIds(data.uploaded_policy_ids);
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.message ?? "",
          type: data.type,
          policies: data.policies,
          extracted: data.extracted_requirements,
          example: data.example,
          citation: data.citation,
          policy_name: data.policy_name,
          found: data.found,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, something went wrong. Please try again.",
          type: "question",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function toggleCompare(id: string, name: string) {
    setSelected((prev) => {
      const exists = prev.find((p) => p.id === id);
      if (exists) return prev.filter((p) => p.id !== id);
      if (prev.length >= 3) return prev;
      return [...prev, { id, name }];
    });
    setCompareResult(null);
  }

  async function handleCompare() {
    if (selected.length < 2) return;
    setComparing(true);
    try {
      const data = await comparePolicies(selected.map((p) => p.id));
      setCompareResult(data);
    } catch {
      console.error("Compare failed");
    } finally {
      setComparing(false);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex flex-col">
      <div className="max-w-3xl mx-auto w-full px-4 flex flex-col" style={{ height: "100vh" }}>

        {/* Header */}
        <div className="pt-6 pb-3 flex-shrink-0">
          <Link
            href="/"
            className="flex items-center gap-2 text-gray-500 hover:text-gray-800 mb-4 text-sm transition w-fit"
          >
            <ArrowLeft className="w-4 h-4" /> Back
          </Link>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-blue-600" />
              <h1 className="text-xl font-black text-gray-900">Policy Advisor</h1>
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-semibold">AI</span>
            </div>

            {/* Session controls */}
            <div className="flex items-center gap-2" ref={historyRef}>
              <button
                onClick={startNewSession}
                title="New conversation"
                className="p-1.5 rounded-lg text-gray-400 hover:text-blue-600 hover:bg-blue-50 transition"
              >
                <Plus className="w-4 h-4" />
              </button>
              <div className="relative">
                <button
                  onClick={() => { setShowHistory((v) => !v); loadSessionHistory(); }}
                  title="Conversation history"
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-500 hover:text-blue-600 hover:bg-blue-50 border border-gray-200 rounded-lg transition"
                >
                  <History className="w-3.5 h-3.5" /> History
                </button>

                {showHistory && (
                  <div className="absolute right-0 top-9 w-72 bg-white border border-gray-200 rounded-xl shadow-lg z-50 overflow-hidden">
                    <div className="px-3 py-2 border-b border-gray-100 flex items-center justify-between">
                      <span className="text-xs font-semibold text-gray-700">Recent conversations</span>
                      <button onClick={() => setShowHistory(false)}>
                        <X className="w-3.5 h-3.5 text-gray-400" />
                      </button>
                    </div>
                    {sessions.length === 0 ? (
                      <div className="px-3 py-4 text-xs text-gray-400 text-center">No previous sessions</div>
                    ) : (
                      <div className="max-h-60 overflow-y-auto">
                        {sessions.map((s) => (
                          <button
                            key={s.id}
                            onClick={() => switchSession(s.id)}
                            className={`w-full flex items-center justify-between px-3 py-2.5 text-left hover:bg-gray-50 transition group ${s.id === sessionId ? "bg-blue-50" : ""}`}
                          >
                            <div className="min-w-0">
                              <div className="text-xs font-medium text-gray-800 truncate">
                                {s.session_name || `Session ${s.id.slice(0, 8)}`}
                              </div>
                              <div className="text-xs text-gray-400">
                                {new Date(s.updated_at).toLocaleDateString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}
                              </div>
                            </div>
                            <button
                              onClick={(e) => handleDeleteSession(s.id, e)}
                              className="opacity-0 group-hover:opacity-100 p-1 text-red-400 hover:text-red-600 transition flex-shrink-0"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </button>
                        ))}
                      </div>
                    )}
                    <div className="border-t border-gray-100 px-3 py-2">
                      <button
                        onClick={startNewSession}
                        className="w-full text-xs text-blue-600 hover:text-blue-800 font-medium flex items-center gap-1.5"
                      >
                        <Plus className="w-3.5 h-3.5" /> New conversation
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
          <p className="text-xs text-gray-400 mt-0.5">Ask in plain English · Get ranked matches · Compare side-by-side · Sessions saved automatically</p>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto py-2 space-y-4 min-h-0">
          {sessionLoading ? (
            <div className="flex justify-center pt-8">
              <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
            </div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`${msg.role === "user" ? "max-w-[80%]" : "w-full"}`}>
                  {/* Bubble */}
                  <div
                    className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                      msg.role === "user"
                        ? "bg-blue-600 text-white rounded-br-md"
                        : "bg-white border border-gray-200 text-gray-800 shadow-sm rounded-bl-md"
                    }`}
                  >
                    {msg.role === "assistant" && (
                      <div className="flex items-center gap-1.5 mb-1.5">
                        <Shield className="w-3.5 h-3.5 text-blue-500" />
                        <span className="text-xs font-bold text-blue-600">PolicyAI</span>
                      </div>
                    )}
                    {msg.content}
                  </div>

                  {/* Results block */}
                  {msg.role === "assistant" && msg.type === "results" && msg.policies && (
                    <div className="mt-3 space-y-3">
                      {/* Extracted requirements chips */}
                      {msg.extracted && (
                        <div className="flex flex-wrap gap-1.5 px-1">
                          {(msg.extracted.needs ?? []).map((n: string) => (
                            <span key={n} className="text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full font-medium">
                              {n}
                            </span>
                          ))}
                          {msg.extracted.budget_max && (
                            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full font-medium">
                              Budget ₹{Number(msg.extracted.budget_max).toLocaleString("en-IN")}/yr
                            </span>
                          )}
                          {msg.extracted.members && (
                            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full font-medium">
                              {msg.extracted.members} members
                            </span>
                          )}
                          {msg.extracted.preferred_type && (
                            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full font-medium">
                              {msg.extracted.preferred_type.replace("_", " ")}
                            </span>
                          )}
                        </div>
                      )}

                      {/* Policy cards + RAG insights */}
                      <div className="grid gap-3">
                        {msg.policies.map((p: any) => (
                          <div key={p.id}>
                            <PolicyCard
                              {...p}
                              onCompare={toggleCompare}
                              isSelected={!!selected.find((s) => s.id === p.id)}
                            />
                            {/* RAG insights from actual policy PDF */}
                            {p.rag_insights?.available && p.rag_insights.hidden_traps?.length > 0 && (
                              <div className="mt-1 border border-orange-200 bg-orange-50 rounded-xl px-3 py-2.5">
                                <div className="text-xs font-bold text-orange-700 mb-1.5 flex items-center gap-1">
                                  <AlertTriangle className="w-3 h-3" /> Hidden Conditions Found in Policy Document
                                </div>
                                {p.rag_insights.hidden_traps.map((trap: any, ti: number) => (
                                  <div key={ti} className="text-xs text-orange-800 mb-1">
                                    <span className="font-semibold">{trap.type.replace(/_/g, " ")}: </span>
                                    {trap.plain_english}
                                    {trap.impact && <span className="text-orange-600"> — {trap.impact}</span>}
                                  </div>
                                ))}
                                {p.rag_insights.key_fact && (
                                  <div className="text-xs text-orange-700 mt-1.5 border-t border-orange-200 pt-1.5">
                                    <span className="font-semibold">Key fact: </span>{p.rag_insights.key_fact}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>

                      {/* Compare bar */}
                      {selected.length >= 2 && (
                        <div className="bg-white border border-blue-200 rounded-xl p-3 flex items-center justify-between gap-3">
                          <div className="text-sm text-gray-700 flex flex-wrap items-center gap-1.5">
                            <span className="font-semibold">{selected.length} selected:</span>
                            {selected.map((p) => (
                              <span key={p.id} className="text-xs bg-gray-100 px-2 py-0.5 rounded-full">
                                {p.name.split(" ").slice(0, 3).join(" ")}
                              </span>
                            ))}
                          </div>
                          <button
                            onClick={handleCompare}
                            disabled={comparing}
                            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-sm font-semibold px-4 py-2 rounded-xl transition flex-shrink-0"
                          >
                            {comparing ? <Loader2 className="w-4 h-4 animate-spin" /> : <BarChart2 className="w-4 h-4" />}
                            Compare
                          </button>
                        </div>
                      )}

                      {/* Comparison table */}
                      {compareResult && (
                        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5">
                          <h2 className="text-base font-bold text-gray-900 mb-3">Side-by-Side Comparison</h2>
                          <ComparisonTable {...compareResult} />
                        </div>
                      )}

                      {/* Continue nudge */}
                      <div className="bg-blue-50 border border-blue-100 rounded-xl px-4 py-2.5 text-xs text-blue-700">
                        Want to refine? Try: &quot;Show me only plans under ₹10k&quot; or &quot;Which covers critical illness?&quot;
                      </div>
                    </div>
                  )}

                  {/* No results block */}
                  {msg.role === "assistant" && msg.type === "no_results" && (
                    <div className="mt-2 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-xs text-amber-800">
                      Try relaxing one requirement — e.g. increase budget slightly or remove a specific coverage need.
                    </div>
                  )}

                  {/* Term explanation block */}
                  {msg.role === "assistant" && msg.type === "explanation" && (
                    <div className="mt-2 space-y-2">
                      {msg.found === false ? (
                        <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-800">
                          {msg.content}
                        </div>
                      ) : (
                        <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-4 space-y-2">
                          <div className="flex items-center gap-1.5 text-xs font-bold text-indigo-700 mb-1">
                            <BookOpen className="w-3.5 h-3.5" /> Plain English Explanation
                            {msg.policy_name && (
                              <span className="font-normal text-indigo-500">from {msg.policy_name}</span>
                            )}
                          </div>
                          {msg.example && (
                            <div className="bg-white rounded-lg px-3 py-2 text-xs text-gray-700 border border-indigo-100">
                              <span className="font-semibold text-indigo-700">Example: </span>{msg.example}
                            </div>
                          )}
                          {msg.citation && (
                            <div className="text-xs text-gray-500 border-l-2 border-indigo-300 pl-3 italic">
                              &quot;{msg.citation}&quot;
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {/* Typing indicator */}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
                <div className="flex items-center gap-2 text-sm text-gray-400">
                  <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                  <span>Searching policies for you…</span>
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Starter prompts — only on first load */}
        {messages.length === 1 && !loading && (
          <div className="flex-shrink-0 flex flex-wrap gap-2 pt-2 pb-1">
            {STARTER_PROMPTS.map((p) => (
              <button
                key={p}
                onClick={() => sendMessage(p)}
                className="text-xs bg-white border border-gray-200 hover:border-blue-300 hover:text-blue-700 text-gray-600 px-3 py-1.5 rounded-full transition"
              >
                {p.length > 48 ? p.slice(0, 48) + "…" : p}
              </button>
            ))}
          </div>
        )}

        {/* Input box */}
        <div className="flex-shrink-0 pb-4 pt-2">
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm flex items-end gap-2 p-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="e.g. maternity + diabetes, ₹15k/year, family of 4…"
              className="flex-1 resize-none text-sm text-gray-800 outline-none min-h-[40px] max-h-[120px] placeholder:text-gray-400 leading-relaxed"
              rows={1}
            />
            <button
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              className="w-9 h-9 flex items-center justify-center bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl transition flex-shrink-0"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
          <p className="text-xs text-center text-gray-400 mt-1.5">Enter to send · Shift+Enter for new line · Sessions saved automatically</p>
        </div>
      </div>
    </main>
  );
}
