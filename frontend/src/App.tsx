import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Loader2, Square, Copy, CopyCheck, AlertCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function App() {
  const [input, setInput] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 使用相对路径，让 API 请求通过 Vite 代理到后端
  // 开发环境: Vite dev server 代理到 localhost:47569
  // 生产环境: 通过 Nginx 或其他反向代理
  const apiUrl = import.meta.env.VITE_API_URL || undefined;  // undefined = 使用相对路径（同源）

  const thread = useStream<{ messages: Message[] }>({
    apiUrl: apiUrl,
    assistantId: "agent",
    messagesKey: "messages",
    onError: (error) => {
      console.error("Stream error:", error);
      const displayUrl = apiUrl || "当前页面 (通过 Vite 代理)";
      setErrorMsg(`连接后端失败，请检查服务是否启动 (地址: ${displayUrl})`);
    },
  });

  // 自动滚动到底部
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thread.messages]);

  // 清除错误状态
  useEffect(() => {
    if (thread.isLoading) {
      setErrorMsg(null);
    }
  }, [thread.isLoading]);

  const handleSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      if (!input.trim() || thread.isLoading) return;

      setErrorMsg(null);
      const newMessages: Message[] = [
        ...(thread.messages || []),
        { type: "human", content: input, id: Date.now().toString() },
      ];

      thread.submit({ messages: newMessages } as any);
      setInput("");
    },
    [input, thread]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleCopy = async (text: string, id: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto">
      {/* 头部 */}
      <header className="px-4 py-3 border-b border-border flex items-center justify-between shrink-0">
        <h1 className="text-lg font-semibold">RAG 教学</h1>
        <span className="text-xs text-text-2">基于 LangGraph 的检索增强生成</span>
      </header>

      {/* 错误提示 */}
      {errorMsg && (
        <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/20 flex items-center gap-2 text-red-400 text-sm">
          <AlertCircle size={16} />
          <span>{errorMsg}</span>
          <button 
            onClick={() => setErrorMsg(null)}
            className="ml-auto text-red-400 hover:text-red-300"
          >
            ✕
          </button>
        </div>
      )}

      {/* 消息区域 */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {thread.messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-text-2">
            <p className="text-2xl mb-2">👋 你好！</p>
            <p>输入问题，我将检索知识库并生成回答</p>
            <p className="text-xs mt-2 text-text-2/60">后端地址: localhost:47569</p>
          </div>
        )}

        {thread.messages.map((msg, idx) => {
          const isHuman = msg.type === "human";
          const id = msg.id || `msg-${idx}`;
          const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);

          return (
            <div key={id} className={`flex ${isHuman ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  isHuman
                    ? "bg-accent text-white rounded-br-md"
                    : "bg-surface-2 text-text rounded-bl-md"
                }`}
              >
                {isHuman ? (
                  <p>{content}</p>
                ) : (
                  <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {content}
                    </ReactMarkdown>
                  </div>
                )}

                {!isHuman && content.length > 0 && (
                  <button
                    onClick={() => handleCopy(content, id)}
                    className="mt-2 flex items-center gap-1 text-xs text-text-2 hover:text-text transition-colors"
                  >
                    {copiedId === id ? <CopyCheck size={14} /> : <Copy size={14} />}
                    {copiedId === id ? "已复制" : "复制"}
                  </button>
                )}
              </div>
            </div>
          );
        })}

        {/* 加载状态 */}
        {thread.isLoading && (
          <div className="flex justify-start">
            <div className="bg-surface-2 rounded-2xl rounded-bl-md px-4 py-3 flex items-center gap-2 text-text-2 text-sm">
              <Loader2 size={16} className="animate-spin" />
              <span>检索知识库中...</span>
            </div>
          </div>
        )}
      </div>

      {/* 输入区域 */}
      <div className="px-4 py-3 border-t border-border shrink-0">
        <form onSubmit={handleSubmit} className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入问题...（Shift+Enter 换行）"
            rows={1}
            className="flex-1 bg-surface-2 rounded-xl px-4 py-3 text-sm text-text placeholder:text-text-2 resize-none outline-none focus:ring-2 focus:ring-accent min-h-[44px] max-h-[120px]"
          />
          {thread.isLoading ? (
            <button
              type="button"
              onClick={() => thread.stop()}
              className="p-3 rounded-xl bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
            >
              <Square size={18} />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="p-3 rounded-xl bg-accent text-white hover:bg-accent-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Send size={18} />
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
