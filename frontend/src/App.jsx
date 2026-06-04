import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

const DEFAULT_CONFIG = {
  duration: 24,
  language: 'zh-CN',
  ratio: '16:9',
  style_name: '奇幻动画电影',
  generate_character_images: true,
  audit_character_images: true,
  character_image_size: '2K',
  character_image_watermark: false,
  generate_videos: true,
  video_resolution: '540p',
  video_model: 'viduq3',
  video_audio: true,
  wait_for_completion: true,
  poll_interval: 10,
}

const EVENT_LABEL = {
  agent_start: 'Supervisor 启动',
  agent_call: 'Supervisor 输出',
  tool_call: 'Supervisor 调用子 agent',
  tool_result: '子 agent 返回结果',
  subagent_start: '子 agent 启动',
  subagent_end: '子 agent 完成',
  subagent_tool_call: '子 agent 调用工具',
  subagent_tool_result: '子 agent 工具结果',
  final_result: '最终结果',
  done: '生成完成',
  error: '错误',
}

const SUBAGENT_LABEL = {
  character_subagent: '主体子 agent',
  shot_subagent: '分镜子 agent',
}

function parseSseFrames(buffer) {
  const frames = []
  let remaining = buffer
  while (remaining.includes('\n\n')) {
    const boundary = remaining.indexOf('\n\n')
    const raw = remaining.slice(0, boundary)
    remaining = remaining.slice(boundary + 2)
    const dataLines = []
    let eventName = 'message'
    for (const line of raw.split(/\r?\n/)) {
      if (line.startsWith('event:')) eventName = line.slice(6).trim()
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
    }
    if (!dataLines.length) continue
    try {
      frames.push({ event: eventName, data: JSON.parse(dataLines.join('\n')) })
    } catch {
      frames.push({ event: 'error', data: { event: 'error', payload: '无法解析 SSE 数据帧' } })
    }
  }
  return { frames, remaining }
}

function formatJSON(value) {
  if (value == null) return ''
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function getCharacterImage(c) {
  return c?.final_image_url || c?.reconstructed_image_url || c?.image_url || ''
}

function MediaArtifact({ artifact }) {
  if (artifact.kind === 'video') {
    return (
      <figure className="media-tile media-tile--video">
        <video src={artifact.url} controls preload="metadata" />
        <figcaption>{artifact.title}</figcaption>
      </figure>
    )
  }
  return (
    <figure className="media-tile media-tile--image">
      <img src={artifact.url} alt={artifact.title} loading="lazy" />
      <figcaption>{artifact.title}</figcaption>
    </figure>
  )
}

function ThinkingStep({ step }) {
  const label = EVENT_LABEL[step.event] || step.event
  const agentLabel = step.agent ? SUBAGENT_LABEL[step.agent] || step.agent : null
  const showPayload =
    step.payload != null &&
    !(typeof step.payload === 'string' && step.payload.length === 0) &&
    step.event !== 'done'

  return (
    <li className={`thinking-step thinking-step--${step.event}`}>
      <div className="thinking-step__head">
        <span className={`thinking-tag thinking-tag--${step.event}`}>{label}</span>
        {step.name ? <span className="thinking-name">{step.name}</span> : null}
        {agentLabel ? <span className="thinking-sub">{agentLabel}</span> : null}
      </div>
      {showPayload ? (
        <pre className="thinking-payload">{formatJSON(step.payload)}</pre>
      ) : null}
      {step.artifacts && step.artifacts.length ? (
        <div className="thinking-media">
          {step.artifacts.map((artifact, idx) => (
            <MediaArtifact key={`${artifact.kind}-${idx}-${artifact.url}`} artifact={artifact} />
          ))}
        </div>
      ) : null}
    </li>
  )
}

function ThinkingPanel({ steps, streaming, collapsed, onToggle }) {
  const label = streaming ? '正在思考…' : '已完成思考'
  return (
    <section className={`thinking-panel ${collapsed ? 'is-collapsed' : ''}`}>
      <button type="button" className="thinking-panel__toggle" onClick={onToggle}>
        <span className={`thinking-panel__indicator ${streaming ? 'is-active' : ''}`} />
        <span className="thinking-panel__label">{label}</span>
        <span className="thinking-panel__count">{steps.length} 步</span>
        <span className="thinking-panel__chevron">{collapsed ? '展开' : '收起'}</span>
      </button>
      {!collapsed ? (
        <ol className="thinking-list">
          {steps.length === 0 ? (
            <li className="thinking-empty">等待第一个事件…</li>
          ) : (
            steps.map((step, idx) => <ThinkingStep key={idx} step={step} />)
          )}
        </ol>
      ) : null}
    </section>
  )
}

function CharacterBlock({ character }) {
  const imageUrl = getCharacterImage(character)
  return (
    <article className="final-card final-card--character">
      <header className="final-card__header">
        <h4>{character.name}</h4>
        <span className="final-tag">type {character.type}</span>
      </header>
      <dl className="final-card__props">
        <div><dt>UID</dt><dd>{character.uid || '-'}</dd></div>
        <div><dt>设定</dt><dd>{character.setting || '-'}</dd></div>
        <div><dt>外观</dt><dd>{character.appearance || '-'}</dd></div>
        <div><dt>音色</dt><dd>{character.voice_id || '-'}</dd></div>
      </dl>
      {imageUrl ? (
        <img className="final-card__media" src={imageUrl} alt={character.name} loading="lazy" />
      ) : (
        <div className="final-card__placeholder">无主体图片</div>
      )}
    </article>
  )
}

function ShotBlock({ shot, video }) {
  return (
    <article className="final-card final-card--shot">
      <header className="final-card__header">
        <h4>镜头 {shot.sort + 1} · {shot.theme}</h4>
        <span className="final-tag">{shot.duration}s</span>
      </header>
      <dl className="final-card__props">
        <div><dt>描述</dt><dd>{shot.description}</dd></div>
        <div><dt>视频提示词</dt><dd>{video?.prompt || '-'}</dd></div>
        <div><dt>状态</dt><dd>{video?.status || 'pending'}</dd></div>
      </dl>
      {video?.video_url ? (
        <video className="final-card__media" src={video.video_url} controls preload="metadata" />
      ) : (
        <div className="final-card__placeholder">{video?.error || '无视频结果'}</div>
      )}
    </article>
  )
}

function FinalResult({ result }) {
  if (!result || typeof result !== 'object') {
    return <pre className="final-raw">{formatJSON(result)}</pre>
  }
  const { characters = [], shots = [], videos = [], plan } = result
  return (
    <div className="final-wrap">
      <header className="final-summary">
        <h3>故事生成完成</h3>
        <div className="final-summary__pills">
          <span>{characters.length} 个主体</span>
          <span>{shots.length} 个分镜</span>
          <span>{videos.filter((v) => v.video_url).length} 条视频</span>
          {plan?.duration_list ? (
            <span>总时长 {plan.duration_list.reduce((a, b) => a + b, 0)}s</span>
          ) : null}
        </div>
      </header>
      {characters.length ? (
        <section className="final-section">
          <h3 className="final-section__title">主体提示词 & 主体图</h3>
          <div className="final-grid final-grid--characters">
            {characters.map((c) => (
              <CharacterBlock key={c.uid || c.name} character={c} />
            ))}
          </div>
        </section>
      ) : null}
      {shots.length ? (
        <section className="final-section">
          <h3 className="final-section__title">分镜提示词 & 分镜视频</h3>
          <div className="final-grid final-grid--shots">
            {shots.map((s) => {
              const video = videos.find((v) => v.sort === s.sort)
              return <ShotBlock key={s.sort} shot={s} video={video} />
            })}
          </div>
        </section>
      ) : null}
    </div>
  )
}

function AssistantMessage({ message, onToggleThinking }) {
  return (
    <div className="message message--assistant">
      <div className="message__avatar">SG</div>
      <div className="message__body">
        <ThinkingPanel
          steps={message.steps}
          streaming={message.status === 'streaming'}
          collapsed={message.thinkingCollapsed}
          onToggle={() => onToggleThinking(message.id)}
        />
        {message.error ? <div className="message-error">{message.error}</div> : null}
        {message.final ? <FinalResult result={message.final} /> : null}
      </div>
    </div>
  )
}

function UserMessage({ message }) {
  return (
    <div className="message message--user">
      <div className="message__body">
        <pre className="message__text">{message.text}</pre>
      </div>
      <div className="message__avatar message__avatar--user">你</div>
    </div>
  )
}

function ConfigDrawer({ config, onChange, open, onToggle }) {
  function update(field, value) {
    onChange({ ...config, [field]: value })
  }

  return (
    <div className={`config-drawer ${open ? 'is-open' : ''}`}>
      <button type="button" className="config-drawer__toggle" onClick={onToggle}>
        <span>视频生成配置</span>
        <span className="config-drawer__chevron">{open ? '收起' : '展开'}</span>
      </button>
      {open ? (
        <div className="config-grid">
          <label className="config-field">
            <span>总时长 (s)</span>
            <input type="number" min="4" value={config.duration}
              onChange={(e) => update('duration', Number(e.target.value))} />
          </label>
          <label className="config-field">
            <span>语言</span>
            <select value={config.language} onChange={(e) => update('language', e.target.value)}>
              <option value="zh-CN">zh-CN</option>
              <option value="en-US">en-US</option>
              <option value="ja-JP">ja-JP</option>
            </select>
          </label>
          <label className="config-field">
            <span>画幅</span>
            <select value={config.ratio} onChange={(e) => update('ratio', e.target.value)}>
              <option value="16:9">16:9</option>
              <option value="9:16">9:16</option>
              <option value="1:1">1:1</option>
            </select>
          </label>
          <label className="config-field">
            <span>主体风格</span>
            <input value={config.style_name} onChange={(e) => update('style_name', e.target.value)} />
          </label>
          <label className="config-field">
            <span>主体图尺寸</span>
            <select value={config.character_image_size}
              onChange={(e) => update('character_image_size', e.target.value)}>
              <option value="1024x1024">1024x1024</option>
              <option value="2K">2K</option>
            </select>
          </label>
          <label className="config-field">
            <span>视频分辨率</span>
            <select value={config.video_resolution}
              onChange={(e) => update('video_resolution', e.target.value)}>
              <option value="540p">540p</option>
              <option value="720p">720p</option>
              <option value="1080p">1080p</option>
            </select>
          </label>
          <label className="config-field">
            <span>视频模型</span>
            <input value={config.video_model} onChange={(e) => update('video_model', e.target.value)} />
          </label>
          <label className="config-field">
            <span>轮询间隔 (s)</span>
            <input type="number" min="1" value={config.poll_interval}
              onChange={(e) => update('poll_interval', Number(e.target.value))} />
          </label>

          <label className="config-toggle">
            <input type="checkbox" checked={config.generate_character_images}
              onChange={(e) => update('generate_character_images', e.target.checked)} />
            <span>生成主体图片</span>
          </label>
          <label className="config-toggle">
            <input type="checkbox" checked={config.audit_character_images}
              onChange={(e) => update('audit_character_images', e.target.checked)} />
            <span>审核并重构图片</span>
          </label>
          <label className="config-toggle">
            <input type="checkbox" checked={config.character_image_watermark}
              onChange={(e) => update('character_image_watermark', e.target.checked)} />
            <span>主体图加水印</span>
          </label>
          <label className="config-toggle">
            <input type="checkbox" checked={config.generate_videos}
              onChange={(e) => update('generate_videos', e.target.checked)} />
            <span>生成分镜视频</span>
          </label>
          <label className="config-toggle">
            <input type="checkbox" checked={config.video_audio}
              onChange={(e) => update('video_audio', e.target.checked)} />
            <span>生成音频</span>
          </label>
          <label className="config-toggle">
            <input type="checkbox" checked={config.wait_for_completion}
              onChange={(e) => update('wait_for_completion', e.target.checked)} />
            <span>等待视频完成</span>
          </label>
        </div>
      ) : null}
    </div>
  )
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [config, setConfig] = useState(DEFAULT_CONFIG)
  const [configOpen, setConfigOpen] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const readerRef = useRef(null)
  const scrollRef = useRef(null)

  const activeAssistantId = useMemo(() => {
    const last = messages[messages.length - 1]
    return last && last.role === 'assistant' && last.status === 'streaming' ? last.id : null
  }, [messages])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  function updateAssistant(id, updater) {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === id ? { ...msg, ...updater(msg) } : msg)),
    )
  }

  function toggleThinking(id) {
    updateAssistant(id, (msg) => ({ thinkingCollapsed: !msg.thinkingCollapsed }))
  }

  async function stopStreaming() {
    if (readerRef.current) {
      try {
        await readerRef.current.cancel()
      } catch {
        /* ignore */
      }
      readerRef.current = null
    }
    setIsStreaming(false)
  }

  async function handleSubmit(event) {
    event?.preventDefault?.()
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return

    const userMessage = { id: `u-${Date.now()}`, role: 'user', text: trimmed }
    const assistantId = `a-${Date.now()}`
    const assistantMessage = {
      id: assistantId,
      role: 'assistant',
      steps: [],
      final: null,
      error: '',
      status: 'streaming',
      thinkingCollapsed: false,
    }

    setMessages((prev) => [...prev, userMessage, assistantMessage])
    setInput('')
    setIsStreaming(true)

    try {
      const response = await fetch(`${API_BASE_URL}/api/story_grid/sse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ ...config, theme: trimmed }),
      })

      if (!response.ok || !response.body) {
        throw new Error(`请求失败：${response.status}`)
      }

      const reader = response.body.getReader()
      readerRef.current = reader
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parsed = parseSseFrames(buffer)
        buffer = parsed.remaining

        for (const frame of parsed.frames) {
          const payload = frame.data
          if (!payload) continue

          if (payload.event === 'error') {
            updateAssistant(assistantId, () => ({
              error: typeof payload.payload === 'string' ? payload.payload : '生成失败',
              status: 'error',
            }))
            continue
          }

          if (payload.event === 'final_result') {
            updateAssistant(assistantId, () => ({ final: payload.payload }))
            continue
          }

          if (payload.event === 'done') {
            updateAssistant(assistantId, () => ({
              status: 'completed',
              thinkingCollapsed: true,
            }))
            continue
          }

          updateAssistant(assistantId, (msg) => ({ steps: [...msg.steps, payload] }))
        }
      }
    } catch (err) {
      updateAssistant(assistantId, () => ({
        error: err instanceof Error ? err.message : '请求失败',
        status: 'error',
      }))
    } finally {
      readerRef.current = null
      setIsStreaming(false)
      updateAssistant(assistantId, (msg) =>
        msg.status === 'streaming' ? { status: 'completed', thinkingCollapsed: true } : {},
      )
    }
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="chat-app">
      <header className="chat-header">
        <h1>StoryGrid Studio</h1>
        <p>输入剧本大纲，自动生成主体图与分镜视频</p>
      </header>

      <main className="chat-main" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="chat-empty">
            <h2>从一段剧本大纲开始</h2>
            <p>在底部输入框写下故事大纲，调整配置后回车发送。</p>
          </div>
        ) : (
          <div className="chat-thread">
            {messages.map((msg) =>
              msg.role === 'user' ? (
                <UserMessage key={msg.id} message={msg} />
              ) : (
                <AssistantMessage
                  key={msg.id}
                  message={msg}
                  onToggleThinking={toggleThinking}
                />
              ),
            )}
          </div>
        )}
      </main>

      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          className="composer__input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入剧本大纲，Enter 发送，Shift+Enter 换行"
          rows={3}
          disabled={isStreaming}
        />
        <div className="composer__actions">
          <ConfigDrawer
            config={config}
            onChange={setConfig}
            open={configOpen}
            onToggle={() => setConfigOpen((v) => !v)}
          />
          <div className="composer__buttons">
            {isStreaming ? (
              <button type="button" className="btn btn--ghost" onClick={stopStreaming}>
                停止
              </button>
            ) : null}
            <button
              type="submit"
              className="btn btn--primary"
              disabled={isStreaming || !input.trim()}
            >
              {isStreaming ? '生成中…' : '发送'}
            </button>
          </div>
        </div>
      </form>
    </div>
  )
}

export default App
