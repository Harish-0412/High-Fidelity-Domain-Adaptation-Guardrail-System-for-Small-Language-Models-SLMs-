
import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import DotGrid from '../components/ui/DotGrid';

function Assistant() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hi! I'm your medical assistant. Ask me any questions about medications, prescriptions, or general medical information, and I'll help with evidence-backed answers.",
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { role: 'user', content: inputValue.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          domain: 'medical_prescription',
          query: userMessage.content,
          top_k: 3,
        }),
      });
      const data = await response.json();
      setMessages((prev) => [...prev, { role: 'assistant', content: data.answer, citations: data.citations }]);
    } catch (error) {
      console.error('Error fetching response:', error);
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again later.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const userProfile = JSON.parse(localStorage.getItem('userProfile') || '{}');

  return (
    <div className="assistant-page">
      <div className="dot-background" aria-hidden="true">
        <DotGrid
          dotSize={5}
          gap={15}
          baseColor="#000000"
          activeColor="#000000"
          proximity={120}
          shockRadius={250}
          shockStrength={5}
          resistance={750}
          returnDuration={1.5}
        />
      </div>
      <div className="assistant-container">
        <div className="assistant-header">
          <div className="header-content">
            <h1>Medical Assistant</h1>
            <p>Ask questions, get evidence-backed answers</p>
          </div>
          <div className="header-actions">
            {userProfile.fullName && (
              <div className="profile-chip">
                {userProfile.fullName.split(' ').map((n) => n[0]).join('').toUpperCase().slice(0, 2)}
              </div>
            )}
            <button onClick={() => navigate('/')} className="secondary-action small">
              Home
            </button>
          </div>
        </div>

        <div className="chat-container">
          <div className="chat-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-content">
                  <p>{msg.content}</p>
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="citations">
                      <h4>Sources:</h4>
                      <ul>
                        {msg.citations.map((cite, cIdx) => (
                          <li key={cIdx}>
                            <strong>{cite.citation_id}</strong>: {cite.text}
                            {cite.source_id && ` (Source: ${cite.source_id})`}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message assistant">
                <div className="typing-indicator">
                  <span></span><span></span><span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-container">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about medications, prescriptions, or medical conditions..."
              disabled={isLoading}
              className="chat-input"
              rows={1}
            />
            <button
              onClick={handleSendMessage}
              disabled={isLoading || !inputValue.trim()}
              className="primary-action send-button"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Assistant;
