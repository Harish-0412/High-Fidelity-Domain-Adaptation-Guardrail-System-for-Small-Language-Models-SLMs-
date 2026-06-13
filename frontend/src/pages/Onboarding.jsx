
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DotGrid from '../components/ui/DotGrid';

const ONBOARDING_STEPS = [
  {
    title: 'Basic Info',
    fields: [
      { name: 'fullName', label: 'Full Name', type: 'text', required: true },
      { name: 'age', label: 'Age', type: 'number', required: true },
      { name: 'gender', label: 'Gender', type: 'select', options: ['Male', 'Female', 'Other', 'Prefer not to say'], required: true },
    ],
  },
  {
    title: 'Medical Profile',
    fields: [
      { name: 'bloodGroup', label: 'Blood Group', type: 'select', options: ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', 'Unknown'], required: true },
      { name: 'height', label: 'Height (cm)', type: 'number' },
      { name: 'weight', label: 'Weight (kg)', type: 'number' },
    ],
  },
  {
    title: 'Health History',
    fields: [
      { name: 'allergies', label: 'Known Allergies', type: 'textarea', placeholder: 'List any allergies you have (e.g., peanuts, penicillin)' },
      { name: 'conditions', label: 'Existing Medical Conditions', type: 'textarea', placeholder: 'List any existing conditions (e.g., diabetes, hypertension)' },
      { name: 'medications', label: 'Current Medications', type: 'textarea', placeholder: 'List any medications you are currently taking' },
    ],
  },
];

function Onboarding() {
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState({});
  const navigate = useNavigate();

  const currentStepData = ONBOARDING_STEPS[currentStep];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleNext = () => {
    if (currentStep < ONBOARDING_STEPS.length - 1) {
      setCurrentStep((prev) => prev + 1);
    } else {
      localStorage.setItem('userProfile', JSON.stringify(formData));
      navigate('/assistant');
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  const isStepValid = () => {
    const requiredFields = currentStepData.fields.filter((field) => field.required);
    return requiredFields.every((field) => formData[field.name]?.trim());
  };

  const progress = ((currentStep + 1) / ONBOARDING_STEPS.length) * 100;

  return (
    <div className="onboarding-page">
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
      <div className="onboarding-container">
        <div className="onboarding-header">
          <h1>Set Up Your Profile</h1>
          <p className="onboarding-subtitle">Let's personalize your medical assistant experience</p>
        </div>

        <div className="progress-bar-container">
          <div className="progress-bar" style={{ width: `${progress}%` }} />
        </div>
        <div className="step-indicators">
          {ONBOARDING_STEPS.map((step, index) => (
            <div
              key={index}
              className={`step-indicator ${index === currentStep ? 'active' : index < currentStep ? 'completed' : ''}`}
            >
              {index < currentStep ? '✓' : index + 1}
            </div>
          ))}
        </div>

        <div className="step-content">
          <h2 className="step-title">{currentStepData.title}</h2>
          <div className="step-fields">
            {currentStepData.fields.map((field) => (
              <div key={field.name} className="field-group">
                <label htmlFor={field.name} className="field-label">
                  {field.label}
                  {field.required && <span className="required">*</span>}
                </label>
                {field.type === 'select' ? (
                  <select
                    id={field.name}
                    name={field.name}
                    value={formData[field.name] || ''}
                    onChange={handleInputChange}
                    className="field-input"
                  >
                    <option value="">Select...</option>
                    {field.options.map((option) => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                ) : field.type === 'textarea' ? (
                  <textarea
                    id={field.name}
                    name={field.name}
                    value={formData[field.name] || ''}
                    onChange={handleInputChange}
                    placeholder={field.placeholder}
                    className="field-input field-textarea"
                  />
                ) : (
                  <input
                    id={field.name}
                    name={field.name}
                    type={field.type}
                    value={formData[field.name] || ''}
                    onChange={handleInputChange}
                    placeholder={field.placeholder}
                    className="field-input"
                  />
                )}
              </div>
            ))}
          </div>
          <div className="step-actions">
            {currentStep > 0 && (
              <button onClick={handleBack} className="secondary-action">
                Back
              </button>
            )}
            <button onClick={handleNext} className="primary-action" disabled={!isStepValid()}>
              {currentStep === ONBOARDING_STEPS.length - 1 ? 'Finish' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Onboarding;
