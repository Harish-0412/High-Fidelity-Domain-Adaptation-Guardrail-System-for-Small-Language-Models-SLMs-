import { useEffect, useState } from 'react';

import DotGrid from './components/ui/DotGrid.jsx';
import CardNav from './components/ui/CardNav.jsx';
import ElevenLabsVoiceAgent from './components/ElevenLabsVoiceAgent.jsx';
import ScrollStack, { ScrollStackItem } from './components/ui/ScrollStack.jsx';
import logo from './assets/logo.svg';

const rotatingWords = ['Assistant', 'Advisor', 'Navigator', 'Companion'];

const navItems = [
  {
    label: 'Assisstant',
    bgColor: '#111111',
    textColor: '#ffffff',
    links: [
      {
        label: 'Ask prescriptions',
        href: '#assistant',
        ariaLabel: 'Open assistant prescription guidance',
      },
      {
        label: 'Check interactions',
        href: '#features',
        ariaLabel: 'View interaction checking feature',
      },
    ],
  },
  {
    label: 'Personal Tracker',
    bgColor: '#2a2a2a',
    textColor: '#ffffff',
    links: [
      {
        label: 'Medication timeline',
        href: '#features',
        ariaLabel: 'View medication timeline feature',
      },
      {
        label: 'Safety reminders',
        href: '#features',
        ariaLabel: 'View safety reminders feature',
      },
    ],
  },
  {
    label: 'Profile',
    bgColor: '#3a3a3a',
    textColor: '#ffffff',
    links: [
      {
        label: 'Health context',
        href: '#features',
        ariaLabel: 'View health context profile feature',
      },
      {
        label: 'Privacy controls',
        href: '#safety',
        ariaLabel: 'View privacy controls',
      },
    ],
  },
];

const features = [
  {
    kicker: 'Assistant',
    title: 'Prescription questions answered with source-backed context.',
    copy:
      'The assistant retrieves relevant drug labels and domain documents first, then shapes responses around the evidence instead of guessing from memory.',
    points: ['Hybrid retrieval', 'Citation-bearing responses', 'Fallback when evidence is weak'],
    stat: '8',
    statLabel: 'citations ready per answer',
  },
  {
    kicker: 'Personal Tracker',
    title: 'A clearer medication journey from reminders to risk signals.',
    copy:
      'The tracker concept keeps dosage schedules, medication history, and follow-up prompts organized so important health context is easier to review.',
    points: ['Medication timeline', 'Adherence-friendly reminders', 'Interaction watchlist'],
    stat: '24/7',
    statLabel: 'context-aware support',
  },
  {
    kicker: 'Guardrails',
    title: 'Safety checks run before the answer reaches the user.',
    copy:
      'Retrieval confidence, citation coverage, and guardrail status are evaluated together so the interface can flag uncertainty early and recommend verified next steps.',
    points: ['Groundedness scoring', 'Audit-friendly traces', 'Safe fallback messaging'],
    stat: '94%',
    statLabel: 'retrieval confidence preview',
  },
  {
    kicker: 'Profile',
    title: 'Personal context without turning the interface into clutter.',
    copy:
      'Profile data is designed around essentials: allergies, active prescriptions, known conditions, and preferences that help tailor safer assistant responses.',
    points: ['Allergy-aware context', 'Condition notes', 'Privacy-first controls'],
    stat: '1',
    statLabel: 'secure health profile',
  },
];

function RotatingText() {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setIndex((current) => (current + 1) % rotatingWords.length);
    }, 1900);

    return () => window.clearInterval(timer);
  }, []);

  return (
    <span className="rotating-word" aria-live="polite">
      <span key={rotatingWords[index]}>{rotatingWords[index]}</span>
    </span>
  );
}

function App() {
  return (
    <main className="landing-page">
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

      <CardNav
        logo={logo}
        logoAlt="MedGuard AI"
        items={navItems}
        baseColor="rgba(255, 255, 255, 0.92)"
        menuColor="#111111"
        buttonBgColor="#111111"
        buttonTextColor="#ffffff"
        ease="power3.out"
      />

      <section className="hero" id="assistant" aria-labelledby="hero-title">
        <div className="hero-copy">
          <p className="eyebrow">Citation-grounded medical intelligence</p>
          <h1 id="hero-title">
            Your Personal Medical <RotatingText />
          </h1>
          <p className="hero-description">
            A calm, evidence-first assistant experience for prescription
            guidance, retrieval-backed answers, and safer healthcare workflows.
          </p>
          <div className="hero-actions">
            <a className="primary-action" href="#demo">
              Start exploring
            </a>
            <a className="secondary-action" href="#safety">
              View safety model
            </a>
          </div>
        </div>

        <aside className="signal-panel" aria-label="Assistant status preview">
          <div className="panel-header">
            <span>Live guardrail status</span>
            <strong>Grounded</strong>
          </div>
          <div className="metric-grid">
            <div>
              <span>Retrieval confidence</span>
              <strong>94%</strong>
            </div>
            <div>
              <span>Citation coverage</span>
              <strong>8 refs</strong>
            </div>
          </div>
          <div className="answer-preview">
            <span className="preview-label">Latest response</span>
            <p>
              Evidence found across prescribing labels. Potential interaction
              detected and flagged before answer delivery.
            </p>
          </div>
        </aside>
      </section>

      <section className="features-section" id="features" aria-labelledby="features-title">
        <div className="section-heading">
          <p className="eyebrow">Features built around trust</p>
          <h2 id="features-title">What we provide, and how the system makes it safer.</h2>
          <p>
            Each feature is designed around the same principle: collect the right
            medical context, retrieve evidence, and make uncertainty visible before
            a user acts on the response.
          </p>
        </div>

        <div className="features-stack-shell">
          <ScrollStack
            itemDistance={82}
            itemScale={0.035}
            itemStackDistance={24}
            stackPosition="18%"
            scaleEndPosition="8%"
            baseScale={0.86}
            rotationAmount={0}
            blurAmount={0.4}
          >
            {features.map((feature) => (
              <ScrollStackItem key={feature.title} itemClassName="feature-card">
                <div className="feature-card-content">
                  <div className="feature-main">
                    <span>{feature.kicker}</span>
                    <h3>{feature.title}</h3>
                    <p>{feature.copy}</p>
                    <ul>
                      {feature.points.map((point) => (
                        <li key={point}>{point}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="feature-stat">
                    <strong>{feature.stat}</strong>
                    <span>{feature.statLabel}</span>
                  </div>
                </div>
              </ScrollStackItem>
            ))}
          </ScrollStack>
        </div>
      </section>

      <ElevenLabsVoiceAgent />
    </main>
  );
}

export default App;
