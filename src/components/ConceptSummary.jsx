import React, { useState, useEffect } from 'react';
import './ConceptSummary.css';

function ConceptSummary({ sectionTitle = 'transformers', setScore }) {
  const storageKey = `summary_${sectionTitle}`;
  
  const [summary, setSummary] = useState(() => {
    return localStorage.getItem(storageKey) || '';
  });

  useEffect(() => {
    localStorage.setItem(storageKey, summary);
  }, [summary, storageKey]);

  const [hasAwardedPoints, setHasAwardedPoints] = useState(false);

  // Calculate word count
  const wordCount = summary.trim().split(/\s+/).filter(word => word.length > 0).length;
  const isLongEnough = wordCount > 100;

  useEffect(() => {
    // Award points if threshold met and not yet awarded
    if (isLongEnough && !hasAwardedPoints && setScore) {
      setScore(prev => prev + 50);
      setHasAwardedPoints(true);
    }
  }, [isLongEnough, hasAwardedPoints, setScore]);

  const handleChange = (e) => {
    setSummary(e.target.value);
  };

  return (
    <div className={`concept-summary-section ${isLongEnough ? 'summary-complete' : ''}`}>
      <div className="summary-header">
        <h3 className="summary-title">📝 Your Summary of This Section</h3>
        <p className="summary-hint">
          Write, in your own words, what you learned about transformers here. 
          This helps solidify your understanding.
        </p>
      </div>
      
      <textarea
        className="summary-textarea"
        value={summary}
        onChange={handleChange}
        placeholder="In this section, I learned that transformers use self-attention to process sequences in parallel. The key insight is..."
        rows={8}
        aria-label="Your summary of transformer concepts"
      />
      
      <div className="summary-footer">
        <span className="summary-info">
          {isLongEnough ? '✅ Great summary! (+50 points)' : `${wordCount} / 100 words`} | 💾 Saved
        </span>
      </div>
    </div>
  );
}

export default ConceptSummary;
