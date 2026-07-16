import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

const audioInput = document.getElementById('audioInput');
const transcribeBtn = document.getElementById('transcribeBtn');
const statusEl = document.getElementById('status');
const debugInfo = document.getElementById('debugInfo');
const transcriptEl = document.getElementById('transcript');
const clearBtn = document.getElementById('clearBtn');
const downloadBtn = document.getElementById('downloadBtn');
const audioPlayer = document.getElementById('audioPlayer');
const progressContainer = document.getElementById('progressContainer');
const progressLabel = document.getElementById('progressLabel');
const progressState = document.getElementById('progressState');
const progressBar = document.getElementById('progressBar');

let asrPipeline = null;
let currentFile = null;
let audioContext = null;
let currentTranscript = '';
let currentAudioDataPromise = null;
let currentRawAudio = null;
let preloadPipelinePromise = null;
let transcriptionStartTime = null;
let lastTranscriptionElapsed = null;
let lastAudioDuration = null;

const errorModal = document.getElementById('errorModal');
const errorMessage = document.getElementById('errorMessage');
const closeErrorModal = document.getElementById('closeErrorModal');
const retryErrorBtn = document.getElementById('retryErrorBtn');

const showProgress = (label, state, percent = 0) => {
  progressContainer.classList.remove('hidden');
  progressLabel.textContent = label;
  progressState.textContent = state;
  progressBar.style.width = `${Math.min(Math.max(percent, 0), 100)}%`;
};

const hideProgress = () => {
  progressContainer.classList.add('hidden');
  progressBar.style.width = '0%';
  progressLabel.textContent = 'Waiting to start...';
  progressState.textContent = 'Preparing';
};

const formatTime = (seconds) => {
  if (!Number.isFinite(seconds) || seconds < 0) return 'unknown';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = (seconds % 60).toFixed(2);
  const parts = [];
  if (hours) parts.push(`${hours}h`);
  if (minutes) parts.push(`${minutes}m`);
  parts.push(`${secs}s`);
  return parts.join(' ');
};

const updateDebugInfo = () => {
  if (!debugInfo) return;
  const elapsedText = lastTranscriptionElapsed != null ? formatTime(lastTranscriptionElapsed / 1000) : 'pending';
  const audioText = Number.isFinite(lastAudioDuration) ? formatTime(lastAudioDuration) : 'unknown';
  debugInfo.textContent = `Transcribe elapsed: ${elapsedText} · Audio duration: ${audioText}`;
};

// Shifts the highlight earlier/later relative to audio.currentTime.
// Positive ms = highlighter lags behind (slower); negative ms = highlighter
// leads ahead (faster). Mutable so the debug slider (added near the bottom
// of this file) can adjust it live.
let highlightSyncOffsetMs = -300;

const LARGE_FILE_STREAM_THRESHOLD = 90 * 1024 * 1024;
const ASR_CHUNK_SECONDS = 30;
const ASR_CHUNK_SIZE = 16000 * ASR_CHUNK_SECONDS;
const STREAM_WORKLET_BATCH_SIZE = 65536;
const STREAM_RESAMPLE_SECONDS = 8;
let streamProcessorUrl = null;

const enableDownload = (text) => {
  currentTranscript = text;
  downloadBtn.disabled = !text;
};

const appendFloat32 = (left, right) => {
  if (!left.length) return right;
  if (!right.length) return left;
  const result = new Float32Array(left.length + right.length);
  result.set(left, 0);
  result.set(right, left.length);
  return result;
};

const mergeToMono = (buffer) => {
  const channelCount = buffer.numberOfChannels;
  if (channelCount === 1) return buffer.getChannelData(0);
  const length = buffer.length;
  const mono = new Float32Array(length);
  for (let c = 0; c < channelCount; c += 1) {
    const channel = buffer.getChannelData(c);
    for (let i = 0; i < length; i += 1) {
      mono[i] += channel[i];
    }
  }
  for (let i = 0; i < length; i += 1) {
    mono[i] /= channelCount;
  }
  return mono;
};

const decodeAudioDataWithRetry = async (context, audioBuffer) => {
  try {
    return await context.decodeAudioData(audioBuffer);
  } catch (error) {
    console.warn('decodeAudioData failed; retrying with a fresh AudioContext:', error);
    await closeAudioContext();
    const retryContext = await ensureAudioContext();
    return await retryContext.decodeAudioData(audioBuffer.slice(0));
  }
};

let transcriptWords = [];
let currentTranscriptText = '';
let activeWordIndex = -1;
let currentTranscriptWordTimings = [];

const escapeHtml = (value) => {
  const div = document.createElement('div');
  div.textContent = value;
  return div.innerHTML;
};

const toMilliseconds = (value) => {
  if (value == null || value === '') return null;
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return null;
  if (numericValue > 100000) return Math.round(numericValue);
  return Math.round(numericValue * 1000);
};

const normalizeWordTiming = (entry) => {
  if (!entry || typeof entry !== 'object') return null;
  const textValue = typeof entry?.text === 'string'
    ? entry.text.trim()
    : (typeof entry?.word === 'string' ? entry.word.trim() : '');
  if (!textValue) return null;

  const timestampCandidate = entry?.timestamp ?? entry?.time ?? entry?.times ?? entry?.timings ?? null;
  let startTimeMs = null;
  let endTimeMs = null;

  if (Array.isArray(timestampCandidate) && timestampCandidate.length >= 2) {
    startTimeMs = toMilliseconds(timestampCandidate[0]);
    endTimeMs = toMilliseconds(timestampCandidate[1]);
  } else if (timestampCandidate && typeof timestampCandidate === 'object') {
    startTimeMs = toMilliseconds(timestampCandidate.start ?? timestampCandidate.startTime ?? timestampCandidate.from ?? timestampCandidate.begin);
    endTimeMs = toMilliseconds(timestampCandidate.end ?? timestampCandidate.endTime ?? timestampCandidate.to ?? timestampCandidate.finish);
  } else {
    startTimeMs = toMilliseconds(entry?.start ?? entry?.startTime ?? entry?.from ?? entry?.begin);
    endTimeMs = toMilliseconds(entry?.end ?? entry?.endTime ?? entry?.to ?? entry?.finish);
  }

  if (startTimeMs == null && endTimeMs != null) {
    startTimeMs = Math.max(0, endTimeMs - 1000);
  }
  if (endTimeMs == null && startTimeMs != null && entry?.duration != null) {
    endTimeMs = startTimeMs + Math.round(toMilliseconds(entry.duration));
  }

  if (startTimeMs == null || endTimeMs == null || endTimeMs <= startTimeMs) {
    return null;
  }

  return {
    text: textValue,
    startTimeMs,
    endTimeMs,
  };
};

const extractWordTimingsFromResult = (result) => {
  const collected = [];
  const addEntry = (entry) => {
    const normalized = normalizeWordTiming(entry);
    if (normalized) collected.push(normalized);
  };

  if (Array.isArray(result?.words)) {
    result.words.forEach(addEntry);
  }

  if (Array.isArray(result?.chunks)) {
    result.chunks.forEach((chunk) => {
      if (Array.isArray(chunk?.words)) {
        chunk.words.forEach(addEntry);
      } else if (chunk?.timestamp || chunk?.time) {
        addEntry(chunk);
      }
      if (Array.isArray(chunk?.segments)) {
        chunk.segments.forEach(addEntry);
      }
    });
  }

  if (!collected.length && Array.isArray(result?.segments)) {
    result.segments.forEach(addEntry);
  }

  return collected;
};

const normalizeTranscriptionPayload = (result, offsetMs = 0) => {
  const text = Array.isArray(result)
    ? result.map((item) => (item?.text ? String(item.text) : '')).filter(Boolean).join(' ')
    : (result?.text ? String(result.text) : '');

  const words = [];
  const addPayloadWords = (payload, payloadOffsetMs = 0) => {
    const timings = extractWordTimingsFromResult(payload);
    timings.forEach((timing) => {
      const startTimeMs = (timing.startTimeMs ?? 0) + payloadOffsetMs;
      const endTimeMs = (timing.endTimeMs ?? startTimeMs + 1000) + payloadOffsetMs;
      words.push({
        ...timing,
        startTimeMs,
        endTimeMs,
        startTime: startTimeMs / 1000,
        endTime: endTimeMs / 1000,
      });
    });
  };

  if (Array.isArray(result)) {
    result.forEach((item) => addPayloadWords(item, offsetMs));
  } else {
    addPayloadWords(result, offsetMs);
  }

  return { text, words };
};

const setTranscriptWordsTiming = (duration, explicitTimings = []) => {
  if (!duration || !transcriptWords.length || !Number.isFinite(duration)) return;
  const totalWords = transcriptWords.length;
  const durationMs = duration * 1000;
  transcriptWords.forEach((entry, idx) => {
    const explicitTiming = explicitTimings[idx] ?? explicitTimings[explicitTimings.length - 1] ?? null;
    if (explicitTiming?.startTimeMs != null && explicitTiming?.endTimeMs != null) {
      entry.startTimeMs = explicitTiming.startTimeMs;
      entry.endTimeMs = explicitTiming.endTimeMs;
      entry.startTime = entry.startTimeMs / 1000;
      entry.endTime = entry.endTimeMs / 1000;
      return;
    }

    entry.startTimeMs = Math.round((durationMs * idx) / totalWords);
    entry.endTimeMs = Math.round((durationMs * (idx + 1)) / totalWords);
    entry.startTime = entry.startTimeMs / 1000;
    entry.endTime = entry.endTimeMs / 1000;
  });
};

const isElementVisible = (el) => {
  const rect = el.getBoundingClientRect();
  return rect.top >= 0 && rect.bottom <= (window.innerHeight || document.documentElement.clientHeight);
};

const findActiveWordIndex = (currentTime) => {
  if (!transcriptWords.length) return -1;
  const currentTimeMs = currentTime * 1000;
  if (activeWordIndex >= 0 && activeWordIndex < transcriptWords.length) {
    const current = transcriptWords[activeWordIndex];
    if (currentTimeMs >= current.startTimeMs && currentTimeMs < current.endTimeMs) {
      return activeWordIndex;
    }
    if (currentTimeMs >= current.endTimeMs) {
      for (let idx = activeWordIndex + 1; idx < transcriptWords.length; idx += 1) {
        const word = transcriptWords[idx];
        if (currentTimeMs >= word.startTimeMs && currentTimeMs < word.endTimeMs) return idx;
      }
    } else {
      for (let idx = activeWordIndex - 1; idx >= 0; idx -= 1) {
        const word = transcriptWords[idx];
        if (currentTimeMs >= word.startTimeMs && currentTimeMs < word.endTimeMs) return idx;
      }
    }
  }
  for (let idx = 0; idx < transcriptWords.length; idx += 1) {
    const word = transcriptWords[idx];
    if (currentTimeMs >= word.startTimeMs && currentTimeMs < word.endTimeMs) return idx;
  }
  if (currentTimeMs >= transcriptWords[transcriptWords.length - 1].startTimeMs) {
    return transcriptWords.length - 1;
  }
  return -1;
};

let highlightFrame = null;

const updateTranscriptHighlights = () => {
  if (!audioPlayer || !transcriptWords.length) return;
  const currentTime = Math.max(0, audioPlayer.currentTime - (highlightSyncOffsetMs / 1000));
  const nextIndex = findActiveWordIndex(currentTime);
  if (nextIndex === activeWordIndex) return;
  if (activeWordIndex >= 0 && transcriptWords[activeWordIndex]) {
    transcriptWords[activeWordIndex].element.classList.remove('highlight-current');
  }
  activeWordIndex = nextIndex;
  if (activeWordIndex >= 0 && transcriptWords[activeWordIndex]) {
    const element = transcriptWords[activeWordIndex].element;
    element.classList.add('highlight-current');
    if (!isElementVisible(element)) {
      element.scrollIntoView({ behavior: 'auto', block: 'nearest' });
    }
  }
};

const queueTranscriptHighlightUpdate = () => {
  if (highlightFrame) return;
  highlightFrame = requestAnimationFrame(() => {
    highlightFrame = null;
    updateTranscriptHighlights();
  });
};

let highlightSyncLoopActive = false;

const runHighlightSyncLoop = () => {
  if (!highlightSyncLoopActive) return;
  updateTranscriptHighlights();
  requestAnimationFrame(runHighlightSyncLoop);
};

const startHighlightSyncLoop = () => {
  if (highlightSyncLoopActive) return;
  highlightSyncLoopActive = true;
  requestAnimationFrame(runHighlightSyncLoop);
};

const stopHighlightSyncLoop = () => {
  highlightSyncLoopActive = false;
};

const renderTranscript = (text, duration, wordTimings = []) => {
  currentTranscriptText = text;
  currentTranscriptWordTimings = Array.isArray(wordTimings) ? wordTimings : [];
  transcriptWords = [];
  activeWordIndex = -1;
  const tokens = text.match(/(\s+|[^\s]+)/g) || [];
  const fragment = document.createDocumentFragment();
  let wordIndex = 0;

  tokens.forEach((token) => {
    if (/^\s+$/.test(token)) {
      const textNode = document.createTextNode(token.replace(/\n/g, '\n'));
      fragment.appendChild(textNode);
      return;
    }

    const looksLikeWord = /[A-Za-z0-9]/.test(token);
    if (!looksLikeWord) {
      const textNode = document.createTextNode(token);
      fragment.appendChild(textNode);
      return;
    }

    const span = document.createElement('span');
    span.className = 'transcript-word';
    span.dataset.wordIndex = String(wordIndex);
    span.textContent = token;
    fragment.appendChild(span);

    const explicitTiming = currentTranscriptWordTimings[wordIndex] ?? null;
    transcriptWords.push({
      element: span,
      text: token,
      startTime: explicitTiming?.startTimeMs != null ? explicitTiming.startTimeMs / 1000 : null,
      endTime: explicitTiming?.endTimeMs != null ? explicitTiming.endTimeMs / 1000 : null,
      startTimeMs: explicitTiming?.startTimeMs ?? null,
      endTimeMs: explicitTiming?.endTimeMs ?? null,
    });
    span.addEventListener('click', () => {
      const entry = transcriptWords[wordIndex];
      if (entry?.startTimeMs != null) {
        audioPlayer.currentTime = entry.startTimeMs / 1000;
        audioPlayer.play();
      }
    });
    wordIndex += 1;
  });

  if (!transcriptWords.length) {
    const emptySpan = document.createElement('span');
    emptySpan.className = 'text-slate-500';
    emptySpan.textContent = 'No transcription returned.';
    fragment.appendChild(emptySpan);
  }

  transcriptEl.innerHTML = '';
  transcriptEl.appendChild(fragment);
  setTranscriptWordsTiming(duration, currentTranscriptWordTimings);
  updateTranscriptHighlights();
};

const clearTranscript = () => {
  transcriptWords = [];
  currentTranscriptText = '';
  currentTranscriptWordTimings = [];
  activeWordIndex = -1;
  transcriptEl.innerHTML = '';
};

const isMp3File = (file) => {
  return /\/mpeg$/i.test(file.type) || /\.mp3$/i.test(file.name);
};

const skipId3v2Tag = (view) => {
  if (view[0] === 0x49 && view[1] === 0x44 && view[2] === 0x33) {
    const size = ((view[6] & 0x7f) << 21) |
                 ((view[7] & 0x7f) << 14) |
                 ((view[8] & 0x7f) << 7) |
                 (view[9] & 0x7f);
    return 10 + size;
  }
  return 0;
};

const isMp3Sync = (view, idx) => {
  return idx + 1 < view.length && view[idx] === 0xff && (view[idx + 1] & 0xe0) === 0xe0;
};

const findPrevMp3Sync = (view, end, minStart) => {
  for (let i = end; i > minStart; i--) {
    if (isMp3Sync(view, i)) return i;
  }
  return -1;
};

const findNextMp3Sync = (view, start, maxSearch) => {
  for (let i = start; i < Math.min(view.length - 1, maxSearch); i++) {
    if (isMp3Sync(view, i)) return i;
  }
  return -1;
};

const mergeDecodedAudioBuffers = (buffers) => {
  const converted = [];
  let totalLength = 0;
  for (const buffer of buffers) {
    const monoData = mergeToMono(buffer);
    const resampled = resampleAudio(monoData, buffer.sampleRate, 16000);
    converted.push(resampled);
    totalLength += resampled.length;
  }
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of converted) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
};

const decodeLargeMp3File = async (file) => {
  const arrayBuffer = await file.arrayBuffer();
  const view = new Uint8Array(arrayBuffer);
  let offset = skipId3v2Tag(view);
  const sliceSize = 32 * 1024 * 1024;
  const decodedBuffers = [];
  let context = await ensureAudioContext();

  while (offset < view.length) {
    let end = Math.min(offset + sliceSize, view.length);
    if (end < view.length) {
      const sync = findPrevMp3Sync(view, end, offset + 2);
      if (sync > offset) {
        end = sync;
      } else {
        const nextSync = findNextMp3Sync(view, end, end + 65536);
        if (nextSync > offset) end = nextSync;
      }
    }
    if (end <= offset) end = view.length;
    const slice = arrayBuffer.slice(offset, end);
    offset = end;

    let decoded;
    try {
      decoded = await decodeAudioDataWithRetry(context, slice);
    } catch (error) {
      throw new Error(`MP3 slice decode failed on segment ${decodedBuffers.length + 1} / ${Math.ceil(view.length / sliceSize)}: ${error?.message || error}`);
    }
    decodedBuffers.push(decoded);
    context = audioContext;
  }

  return mergeDecodedAudioBuffers(decodedBuffers);
};

const getStreamCaptureProcessorUrl = () => {
  if (streamProcessorUrl) return streamProcessorUrl;
  const processorCode = `class StreamCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.buffer = new Float32Array(0);
      this.threshold = ${STREAM_WORKLET_BATCH_SIZE};
    }

    process(inputs) {
      const input = inputs[0];
      if (input && input[0] && input[0].length) {
        const channelData = input[0];
        const nextBuffer = new Float32Array(this.buffer.length + channelData.length);
        nextBuffer.set(this.buffer);
        nextBuffer.set(channelData, this.buffer.length);
        this.buffer = nextBuffer;

        while (this.buffer.length >= this.threshold) {
          const chunk = new Float32Array(this.buffer.subarray(0, this.threshold));
          this.port.postMessage(chunk, [chunk.buffer]);
          this.buffer = this.buffer.subarray(this.threshold);
        }
      }
      return true;
    }
  }
  registerProcessor('stream-capture-processor', StreamCaptureProcessor);`;
  streamProcessorUrl = URL.createObjectURL(new Blob([processorCode], { type: 'application/javascript' }));
  return streamProcessorUrl;
};

const streamTranscribeLargeFile = async (file, pipelineInstance, updateProgress = () => {}) => {
  const audio = document.createElement('audio');
  audio.src = URL.createObjectURL(file);
  audio.preload = 'auto';
  audio.muted = true;
  audio.crossOrigin = 'anonymous';
  audio.style.display = 'none';
  document.body.appendChild(audio);

  const ctx = await ensureAudioContext();
  await ctx.resume();
  await ctx.audioWorklet.addModule(getStreamCaptureProcessorUrl());

  const source = ctx.createMediaElementSource(audio);
  const worklet = new AudioWorkletNode(ctx, 'stream-capture-processor');
  source.connect(worklet).connect(ctx.destination);

  let rawBuffer = new Float32Array(0);
  let resampledBuffer = new Float32Array(0);
  const chunkTexts = [];
  const chunkWordTimings = [];
  let chunkCounter = 0;
  let queue = Promise.resolve();
  let decodeError = null;
  let streamedOffsetMs = 0;

  const enqueueChunk = (chunk) => {
    queue = queue.then(async () => {
      const chunkIndex = ++chunkCounter;
      const chunkDurationMs = Math.round((chunk.length / 16000) * 1000);
      updateProgress(`Transcribing chunk ${chunkIndex}`, `Processed ${chunkIndex} chunks`, 55 + Math.min(35, chunkIndex * 2));
      const result = await pipelineInstance(chunk, { return_timestamps: 'word' });
      const normalized = normalizeTranscriptionPayload(result, streamedOffsetMs);
      if (normalized.text) {
        chunkTexts.push(normalized.text);
      }
      if (normalized.words.length) {
        chunkWordTimings.push(...normalized.words);
      }
      streamedOffsetMs += chunkDurationMs;
    });
  };

  const flushResampledBuffer = () => {
    while (resampledBuffer.length >= ASR_CHUNK_SIZE) {
      const chunk = resampledBuffer.subarray(0, ASR_CHUNK_SIZE);
      resampledBuffer = resampledBuffer.subarray(ASR_CHUNK_SIZE);
      enqueueChunk(chunk);
    }
  };

  worklet.port.onmessage = (event) => {
    rawBuffer = appendFloat32(rawBuffer, event.data);
    if (rawBuffer.length >= ctx.sampleRate * STREAM_RESAMPLE_SECONDS) {
      const resampled = resampleAudio(rawBuffer, ctx.sampleRate, 16000);
      resampledBuffer = appendFloat32(resampledBuffer, resampled);
      rawBuffer = new Float32Array(0);
      flushResampledBuffer();
    }
  };

  audio.onerror = (event) => {
    decodeError = new Error('Audio playback failed during stream decode.');
  };

  const finish = async () => {
    if (decodeError) throw decodeError;
    if (rawBuffer.length) {
      const resampled = resampleAudio(rawBuffer, ctx.sampleRate, 16000);
      resampledBuffer = appendFloat32(resampledBuffer, resampled);
      rawBuffer = new Float32Array(0);
    }
    flushResampledBuffer();
    if (resampledBuffer.length) {
      enqueueChunk(resampledBuffer);
      resampledBuffer = new Float32Array(0);
    }
    await queue;
    return { text: chunkTexts.join(' '), words: chunkWordTimings };
  };

  try {
    await audio.play();
    await new Promise((resolve, reject) => {
      audio.addEventListener('ended', resolve, { once: true });
      audio.addEventListener('error', () => reject(new Error('Audio playback error during stream decode.')), { once: true });
    });
    const result = await finish();
    return result;
  } finally {
    source.disconnect();
    worklet.disconnect();
    audio.pause();
    audio.src = '';
    audio.remove();
  }
};

const resampleAudio = (input, inputRate, outputRate) => {
  if (inputRate === outputRate) return input;
  const ratio = inputRate / outputRate;
  const outputLength = Math.round(input.length / ratio);
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const t = i * ratio;
    const index = Math.floor(t);
    const nextIndex = Math.min(index + 1, input.length - 1);
    const mix = t - index;
    output[i] = input[index] * (1 - mix) + input[nextIndex] * mix;
  }
  return output;
};

const closeAudioContext = async () => {
  if (!audioContext) return;
  try {
    await audioContext.close();
  } catch (error) {
    console.warn('Failed to close existing AudioContext:', error);
  }
  audioContext = null;
};

const ensureAudioContext = async () => {
  if (audioContext) return audioContext;
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  return audioContext;
};

const preloadPipeline = async (device) => {
  if (asrPipeline) return asrPipeline;
  if (!preloadPipelinePromise) {
    preloadPipelinePromise = pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny', { device })
      .then((loadedPipeline) => {
        asrPipeline = loadedPipeline;
        return loadedPipeline;
      })
      .catch((error) => {
        preloadPipelinePromise = null;
        throw error;
      });
  }
  return preloadPipelinePromise;
};

const prepareAudio = async (file) => {
  if (!file) {
    currentAudioDataPromise = null;
    currentRawAudio = null;
    return null;
  }
  if (currentRawAudio && currentFile === file) return currentRawAudio;
  if (currentAudioDataPromise && currentFile === file) return currentAudioDataPromise;
  const decodeFunction = file.size > LARGE_FILE_STREAM_THRESHOLD && isMp3File(file)
    ? decodeLargeMp3File
    : decodeAudioFile;
  currentAudioDataPromise = decodeFunction(file)
    .then((rawAudio) => {
      currentRawAudio = rawAudio;
      return rawAudio;
    })
    .catch((error) => {
      currentAudioDataPromise = null;
      currentRawAudio = null;
      throw error;
    });
  return currentAudioDataPromise;
};

const transcribeLargeMp3File = async (file, pipelineInstance, updateProgress = () => {}) => {
  const arrayBuffer = await file.arrayBuffer();
  const view = new Uint8Array(arrayBuffer);
  let offset = skipId3v2Tag(view);
  const sliceSize = 32 * 1024 * 1024;
  let pendingBuffer = new Float32Array(0);
  const texts = [];
  const wordTimings = [];
  let sliceIndex = 0;
  let totalSlices = Math.ceil((view.length - offset) / sliceSize);

  while (offset < view.length) {
    let end = Math.min(offset + sliceSize, view.length);
    if (end < view.length) {
      const sync = findPrevMp3Sync(view, end, offset + 2);
      if (sync > offset) {
        end = sync;
      } else {
        const nextSync = findNextMp3Sync(view, end, end + 65536);
        if (nextSync > offset) end = nextSync;
      }
    }
    if (end <= offset) end = view.length;

    sliceIndex += 1;
    updateProgress(`Decoding slice ${sliceIndex}/${totalSlices}`, `Preparing audio slice ${sliceIndex}`, 20 + Math.round((sliceIndex / totalSlices) * 20));
    const sliceBuffer = arrayBuffer.slice(offset, end);
    await closeAudioContext();
    const context = await ensureAudioContext();
    let decoded;
    try {
      decoded = await context.decodeAudioData(sliceBuffer);
    } catch (error) {
      throw new Error(`MP3 slice decode failed on segment ${sliceIndex}/${totalSlices}: ${error?.message || error}`);
    }

    const monoData = mergeToMono(decoded);
    const resampled = resampleAudio(monoData, decoded.sampleRate, 16000);
    pendingBuffer = appendFloat32(pendingBuffer, resampled);

    while (pendingBuffer.length >= ASR_CHUNK_SIZE) {
      const chunk = pendingBuffer.subarray(0, ASR_CHUNK_SIZE);
      pendingBuffer = pendingBuffer.subarray(ASR_CHUNK_SIZE);
      const chunkResult = await pipelineInstance(chunk, { return_timestamps: 'word' });
      const normalized = normalizeTranscriptionPayload(chunkResult, Math.round((sliceIndex - 1) * sliceSize / 16000 * 1000));
      if (normalized.text) texts.push(normalized.text);
      if (normalized.words.length) wordTimings.push(...normalized.words);
    }

    offset = end;
  }

  if (pendingBuffer.length > 0) {
    const chunkResult = await pipelineInstance(pendingBuffer, { return_timestamps: 'word' });
    const normalized = normalizeTranscriptionPayload(chunkResult, Math.round((sliceIndex - 1) * sliceSize / 16000 * 1000));
    if (normalized.text) texts.push(normalized.text);
    if (normalized.words.length) wordTimings.push(...normalized.words);
  }
  return { text: texts.join(' '), words: wordTimings };
};

const formatTranscript = (text) => {
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (!normalized) return '';
  const sentences = normalized.match(/[^.!?]+[.!?]?/g) || [normalized];
  const paragraphs = [];

  const wrapSentence = (sentence) => {
    const words = sentence.trim().split(' ');
    const lines = [];
    let current = [];

    for (const word of words) {
      current.push(word);
      if (current.length >= 12 || /[.!?]$/.test(word)) {
        lines.push(current.join(' '));
        current = [];
      }
    }
    if (current.length) lines.push(current.join(' '));
    return lines.join('\n');
  };

  for (const sentence of sentences) {
    const trimmed = sentence.trim();
    if (!trimmed) continue;
    const withEnding = /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
    const capitalized = withEnding.charAt(0).toUpperCase() + withEnding.slice(1);
    paragraphs.push(wrapSentence(capitalized));
  }

  return paragraphs.join('\n\n');
};

const decodeAudioFile = async (file) => {
  const arrayBuffer = await file.arrayBuffer();
  await closeAudioContext();
  const context = await ensureAudioContext();
  const buffer = await decodeAudioDataWithRetry(context, arrayBuffer);
  const monoData = mergeToMono(buffer);
  return resampleAudio(monoData, buffer.sampleRate, 16000);
};

audioInput.addEventListener('change', () => {
  currentFile = audioInput.files?.[0] ?? null;
  currentRawAudio = null;
  currentAudioDataPromise = null;
  transcribeBtn.disabled = !currentFile;
  if (currentFile) {
    const isLargeFile = currentFile.size > LARGE_FILE_STREAM_THRESHOLD;
    const isLargeMp3 = isLargeFile && isMp3File(currentFile);
    statusEl.textContent = `Selected file: ${currentFile.name} (${Math.round(currentFile.size / 1024)} KB)`;
    audioPlayer.classList.add('hidden');
    audioPlayer.src = '';
    clearTranscript();
    downloadBtn.disabled = true;
    if (!isLargeFile || isLargeMp3) {
      prepareAudio(currentFile).catch((error) => {
        statusEl.textContent = 'Error preparing audio: ' + (error?.message ?? error?.toString());
        transcribeBtn.disabled = true;
        showErrorModal(error);
      });
    } else {
      statusEl.textContent = `Large non-MP3 file selected; streaming decode will be used during transcription.`;
    }
    preloadPipeline(getPreferredDevice()).catch(() => {});
  } else {
    statusEl.textContent = 'Please select an audio file to transcribe.';
  }
});

clearBtn.addEventListener('click', () => {
  clearTranscript();
  statusEl.textContent = 'Transcript cleared. Choose a new audio file to transcribe.';
  audioInput.value = '';
  currentFile = null;
  transcribeBtn.disabled = true;
  downloadBtn.disabled = true;
  audioPlayer.classList.add('hidden');
  audioPlayer.src = '';
  hideProgress();
});

downloadBtn.addEventListener('click', () => {
  if (!currentTranscript) return;
  const blob = new Blob([currentTranscript], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'transcript.txt';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
});

const getPreferredDevice = () => {
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    return 'webgpu';
  }
  return 'wasm';
};

audioPlayer.addEventListener('timeupdate', queueTranscriptHighlightUpdate);
audioPlayer.addEventListener('seeked', queueTranscriptHighlightUpdate);
audioPlayer.addEventListener('play', startHighlightSyncLoop);
audioPlayer.addEventListener('pause', stopHighlightSyncLoop);
audioPlayer.addEventListener('ended', stopHighlightSyncLoop);
audioPlayer.addEventListener('loadedmetadata', () => {
  if (currentTranscriptText) {
    setTranscriptWordsTiming(audioPlayer.duration, currentTranscriptWordTimings);
    updateTranscriptHighlights();
  }
  lastAudioDuration = Number.isFinite(audioPlayer.duration) ? audioPlayer.duration : lastAudioDuration;
  updateDebugInfo();
});

const showErrorModal = (error) => {
  const message = error?.message || error?.toString() || 'Unknown error';
  const fileInfo = currentFile ? `File: ${currentFile.name} (${Math.round(currentFile.size / 1024 / 1024)} MB)\n\n` : '';
  const details = error?.stack ? `${fileInfo}${message}\n\n${error.stack}` : `${fileInfo}${message}`;
  errorMessage.textContent = details;
  errorModal.classList.remove('hidden');
};

closeErrorModal.addEventListener('click', () => {
  errorModal.classList.add('hidden');
});

retryErrorBtn.addEventListener('click', async () => {
  errorModal.classList.add('hidden');
  if (currentFile) {
    transcribeBtn.click();
  }
});

transcribeBtn.addEventListener('click', async () => {
  if (!currentFile) {
    statusEl.textContent = 'No audio file selected.';
    return;
  }

  transcriptionStartTime = performance.now();
  lastTranscriptionElapsed = null;
  lastAudioDuration = null;
  updateDebugInfo();

  transcribeBtn.disabled = true;
  audioInput.disabled = true;
  const device = getPreferredDevice();
  statusEl.textContent = `Loading the Whisper model on ${device.toUpperCase()} and preparing transcription...`;
  showProgress('Preparing audio', 'Decoding file…', 10);

  try {
    showProgress('Preparing', 'Decoding audio and loading Whisper Tiny…', 25);
    const isLargeFile = currentFile.size > LARGE_FILE_STREAM_THRESHOLD;
    const isLargeMp3 = isLargeFile && isMp3File(currentFile);
    const useStreamDecode = isLargeFile && !isLargeMp3;
    const audioPromise = useStreamDecode
      ? Promise.resolve(null)
      : currentRawAudio ? Promise.resolve(currentRawAudio) : (currentAudioDataPromise || prepareAudio(currentFile));
    const modelPromise = asrPipeline ? Promise.resolve(asrPipeline) : preloadPipeline(device);
    const [rawAudio, pipelineInstance] = await Promise.all([audioPromise, modelPromise]);
    asrPipeline = pipelineInstance;
    showProgress('Transcribing', 'Running ASR pipeline…', 55);

    let result;
    if (useStreamDecode) {
      statusEl.textContent = 'Large file detected. Streaming decode and transcribing...';
      showProgress('Transcribing', 'Large file decode path in progress...', 60);
      if (isMp3File(currentFile)) {
        result = await transcribeLargeMp3File(currentFile, asrPipeline, (label, state, percent) => showProgress(label, state, percent));
      } else {
        result = await streamTranscribeLargeFile(currentFile, asrPipeline, (label, state, percent) => showProgress(label, state, percent));
      }
    } else {
      const durationSeconds = rawAudio.length / 16000;
      const chunkSeconds = Math.min(90, Math.max(30, Math.ceil(durationSeconds / 8)));
      const chunkSize = chunkSeconds * 16000;
      const transcribeChunks = async (audioArray) => {
        const totalChunks = Math.ceil(audioArray.length / chunkSize);
        const chunks = [];
        for (let offset = 0; offset < audioArray.length; offset += chunkSize) {
          chunks.push(audioArray.subarray(offset, Math.min(offset + chunkSize, audioArray.length)));
        }

        try {
          await asrPipeline(new Float32Array(16000));
        } catch (e) {
          // ignore warmup errors
        }

        const concurrency = Math.min(3, Math.max(1, Math.ceil(navigator.hardwareConcurrency ? navigator.hardwareConcurrency / 4 : 2)));
        const results = new Array(chunks.length);
        let cursor = 0;

        const worker = async () => {
          while (true) {
            const idx = cursor++;
            if (idx >= chunks.length) break;
            const chunk = chunks[idx];
            const chunkOffsetMs = Math.round((idx * chunkSize) / 16000 * 1000);
            showProgress('Transcribing', `Chunk ${idx + 1} / ${chunks.length}...`, 55 + Math.round((idx / chunks.length) * 35));
            const chunkResult = await asrPipeline(chunk, { return_timestamps: 'word' });
            const normalized = normalizeTranscriptionPayload(chunkResult, chunkOffsetMs);
            results[idx] = normalized;
          }
        };

        const workers = [];
        for (let i = 0; i < concurrency; i++) workers.push(worker());
        await Promise.all(workers);

        const textParts = [];
        const wordTimings = [];
        results.filter(Boolean).forEach((payload) => {
          if (payload?.text) textParts.push(payload.text);
          if (payload?.words?.length) wordTimings.push(...payload.words);
        });
        return { text: textParts.join(' '), words: wordTimings };
      };

      result = await transcribeChunks(rawAudio);
    }

    let text = '';
    let recognizedWordTimings = [];
    if (Array.isArray(result)) {
      text = result.map(item => item?.text ?? JSON.stringify(item)).join(' ');
    } else if (result?.text) {
      text = result.text;
    } else {
      text = JSON.stringify(result, null, 2);
    }

    if (result?.words?.length) {
      recognizedWordTimings = result.words;
    } else {
      recognizedWordTimings = extractWordTimingsFromResult(result);
    }

    const formattedText = formatTranscript(text || '');
    currentTranscript = formattedText;
    enableDownload(formattedText);
    if (currentFile) {
      audioPlayer.src = URL.createObjectURL(currentFile);
      audioPlayer.classList.remove('hidden');
    }
    renderTranscript(formattedText, audioPlayer.duration, recognizedWordTimings);
    lastTranscriptionElapsed = performance.now() - transcriptionStartTime;
    lastAudioDuration = Number.isFinite(audioPlayer.duration) ? audioPlayer.duration : null;
    updateDebugInfo();
    statusEl.textContent = 'Transcription complete.';
    showProgress('Complete', 'Done', 100);
  } catch (error) {
    console.error(error);
    statusEl.textContent = 'Error: ' + (error?.message ?? error?.toString() ?? 'Unknown error');
    transcriptEl.textContent = '';
    showProgress('Error', 'Failed to transcribe', 100);
    showErrorModal(error);
  } finally {
    transcribeBtn.disabled = !currentFile;
    audioInput.disabled = false;
  }
});


statusEl.textContent = 'Choose an audio file to begin transcription.';