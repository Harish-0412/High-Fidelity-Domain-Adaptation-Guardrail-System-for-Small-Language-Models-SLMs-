import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ConversationProvider,
  useConversation,
} from '@elevenlabs/react';

const FALLBACK_AGENT_CONFIG = {
  agent_id: 'agent_4501ktybdpspf4hav9egx5xzpyng',
  name: 'Victoria Neuman',
  first_message:
    "[warmly] Hello, I'm Harish's Girlfriend, your AI medical assistant. Please tell me, what brings you in today?",
};

function ElevenLabsVoiceAgent() {
  const [agentConfig, setAgentConfig] = useState(FALLBACK_AGENT_CONFIG);

  useEffect(() => {
    let isMounted = true;

    fetch('/voice-agent')
      .then((response) => {
        if (!response.ok) {
          throw new Error('Voice agent unavailable');
        }
        return response.json();
      })
      .then((config) => {
        if (isMounted && config?.agent_id) {
          setAgentConfig(config);
        }
      })
      .catch(() => {
        if (isMounted) {
          setAgentConfig(FALLBACK_AGENT_CONFIG);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <ConversationProvider agentId={agentConfig.agent_id}>
      <VoiceAgentFlyout agentConfig={agentConfig} />
    </ConversationProvider>
  );
}

function VoiceAgentFlyout({ agentConfig }) {
  const [isOpen, setIsOpen] = useState(false);
  const [error, setError] = useState('');
  const [conversationId, setConversationId] = useState('');
  const {
    endSession,
    isListening,
    isSpeaking,
    startSession,
    status,
  } = useConversation();

  const assistantName = agentConfig?.name ?? FALLBACK_AGENT_CONFIG.name;
  const firstMessage = agentConfig?.first_message ?? FALLBACK_AGENT_CONFIG.first_message;

  const endVoiceSession = useCallback(() => {
    endSession();
    setIsOpen(false);
    setConversationId('');
  }, [endSession]);

  const startVoiceSession = useCallback(async () => {
    setError('');
    setIsOpen(true);

    try {
      const permissionStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      permissionStream.getTracks().forEach((track) => track.stop());
      startSession({
        agentId: agentConfig.agent_id,
        onConnect: ({ conversationId: activeConversationId }) => {
          setConversationId(activeConversationId);
        },
        onDisconnect: () => {
          setConversationId('');
        },
        onError: (message) => {
          setError(typeof message === 'string' ? message : 'Voice session failed');
        },
      });
    } catch (sessionError) {
      const denied = sessionError?.name === 'NotAllowedError';
      setError(
        denied
          ? 'Microphone permission is needed to start the voice assistant.'
          : 'Unable to start the voice assistant.'
      );
    }
  }, [agentConfig.agent_id, startSession]);

  useEffect(() => {
    return () => {
      endSession();
    };
  }, [endSession]);

  const isLive = status === 'connected';
  const isStarting = status === 'connecting';
  const statusText = useMemo(() => {
    if (error) return error;
    if (isStarting) return 'Connecting to voice';
    if (isLive && isSpeaking) return 'Speaking back';
    if (isLive && isListening) return 'Listening now';
    if (isLive) return 'Voice call active';
    return 'Tap to start voice';
  }, [error, isListening, isLive, isSpeaking, isStarting]);

  const handleLauncherClick = () => {
    if (isOpen || isLive || isStarting) {
      setIsOpen(true);
      return;
    }

    startVoiceSession();
  };

  return (
    <div className={`voice-agent-flyout ${isOpen ? 'is-open' : ''}`}>
      {isOpen ? (
        <section className="voice-agent-panel" aria-label={`${assistantName} voice assistant`}>
          <div className="voice-agent-panel-header">
            <div>
              <span>ElevenLabs voice assistant</span>
              <strong>{assistantName}</strong>
            </div>
            <button
              className="voice-agent-close"
              type="button"
              aria-label="Stop and close voice assistant"
              onClick={endVoiceSession}
            >
              x
            </button>
          </div>

          <div className="voice-agent-session">
            <div className={`voice-agent-live-orb ${isLive ? 'is-live' : ''}`} aria-hidden="true">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <div className="voice-agent-session-copy">
              <span>{isLive ? 'Live voice session' : isStarting ? 'Starting session' : 'Voice assistant'}</span>
              <strong>{statusText}</strong>
              <p>{firstMessage}</p>
            </div>
          </div>

          <div className="voice-agent-controls">
            {isLive || isStarting ? (
              <button className="voice-agent-end" type="button" onClick={endVoiceSession}>
                Stop voice
              </button>
            ) : (
              <button className="voice-agent-start" type="button" onClick={startVoiceSession}>
                Start voice
              </button>
            )}
          </div>

          {conversationId ? (
            <span className="voice-agent-conversation-id">Session {conversationId}</span>
          ) : null}
        </section>
      ) : null}

      <button
        className="voice-agent-launcher"
        type="button"
        aria-label={isOpen ? 'Show voice assistant' : 'Start ElevenLabs voice assistant'}
        aria-expanded={isOpen}
        onClick={handleLauncherClick}
      >
        <span className={`voice-agent-orb ${isLive ? 'is-live' : ''}`} aria-hidden="true">
          <span></span>
        </span>
        <span className="voice-agent-text">
          <strong>{assistantName}</strong>
          <small>{statusText}</small>
        </span>
      </button>
    </div>
  );
}

export default ElevenLabsVoiceAgent;
